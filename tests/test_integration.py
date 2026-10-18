"""
tests/test_integration.py
=========================
End-to-end integration test over the current (YOLO) pipeline, driven by the
bundled sample radiograph.

- The image path is resolved relative to the repository root; the sample and
  both YOLO weight files are untracked (see .gitignore), so the test skips
  with a clear reason when any of them is absent instead of fabricating a
  passing run.
- Image dimensions are derived from the actual decoded image, never hardcoded.
- Assertions stay at the smoke level (stage wiring + honest status values);
  they make no claim about clinical accuracy or model validation.
"""

from pathlib import Path

import pytest

from preprocessing.validation import validate_and_load_image
from preprocessing.enhancement import enhance_xray, enhance_rgb_xray
from modules.segmentation.backend import run_segmentation
from modules.segmentation.fdi_resolver import resolve_dental_arch
from modules.segmentation.visualization import create_segmentation_overlay
from modules.pathology.registry import evaluate_all_pathologies
from modules.pathology.yolo_detector import (
    register_yolo_pathology_detectors,
    resolve_pathology_weights_path,
)
from modules.segmentation.yolo_adapter import (
    resolve_yolo_inference_device,
    resolve_yolo_weights_path,
)
from utils.pdf_report import generate_dental_pdf_report


REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_RADIOGRAPH = REPO_ROOT / "models" / "samples" / "sample_opg.jpg"


def _missing_prerequisites():
    """Return a skip reason if sample or weights are absent, else None."""
    missing = []
    if not SAMPLE_RADIOGRAPH.exists():
        missing.append(f"sample radiograph not found at {SAMPLE_RADIOGRAPH}")
    if not resolve_yolo_weights_path().exists():
        missing.append("YOLOv8 segmentation weights not found")
    if not resolve_pathology_weights_path().exists():
        missing.append("YOLO11 pathology weights not found")
    if missing:
        return "integration test skipped: " + "; ".join(missing)
    return None


def test_full_pipeline_on_real_radiograph():
    skip_reason = _missing_prerequisites()
    if skip_reason is not None:
        pytest.skip(skip_reason)

    # 1. Validation & Loading (dimensions derived from the actual image).
    val_res = validate_and_load_image(str(SAMPLE_RADIOGRAPH))
    assert val_res.is_valid
    assert val_res.width == val_res.rgb_array.shape[1]
    assert val_res.height == val_res.rgb_array.shape[0]

    # 2. Enhancement
    gray_enh = enhance_xray(val_res.grayscale_array)
    rgb_enh = enhance_rgb_xray(val_res.rgb_array)
    assert gray_enh.shape == val_res.grayscale_array.shape
    assert rgb_enh.shape == val_res.rgb_array.shape

    # 3. Segmentation via the resolved backend (YOLO when weights present).
    raw_res, backend_name = run_segmentation(
        image_rgb=rgb_enh,
        backend="auto",
        conf_threshold=0.30,
    )
    assert backend_name == "yolov8"
    assert raw_res is not None
    assert len(raw_res.detections) > 20
    # Inference-device provenance must be concrete, never the old vague label.
    assert raw_res.device_used != "cpu/gpu (ultralytics auto)"
    assert raw_res.device_used in ("cpu",) or raw_res.device_used.startswith("cuda")
    assert resolve_yolo_inference_device() == "cpu"  # CPU-only smoke runner.

    # 4. FDI Resolver
    arch = resolve_dental_arch(raw_res.detections, (val_res.height, val_res.width))
    assert arch.total_detected_teeth >= 25
    assert len(arch.missing_teeth) <= 7
    # Verify no duplicate FDI assignments in confirmed teeth
    fdis = [t.fdi_number for t in arch.teeth]
    assert len(fdis) == len(set(fdis)), "Duplicate confirmed FDI tooth numbers detected!"

    # 5. Visual Overlay
    overlay = create_segmentation_overlay(
        image_rgb=val_res.rgb_array,
        teeth=arch.teeth,
        restorations=arch.restorations,
        unassigned=arch.unassigned,
    )
    assert overlay.shape == val_res.rgb_array.shape

    # 6. Pathology Status (smoke level: registry wiring + honest statuses only).
    register_yolo_pathology_detectors()
    pathology = evaluate_all_pathologies(val_res.rgb_array)
    assert set(pathology) == {
        "dental_caries",
        "periapical_lesion",
        "impacted_teeth",
        "periodontal_bone_loss",
    }
    for code, result in pathology.items():
        if code == "periodontal_bone_loss":
            assert result.status == "NOT_CONFIGURED"
        else:
            # Screened by the YOLO detector: either DETECTED findings or an
            # honest NEGATIVE — never a fabricated diagnosis.
            assert result.status in ("DETECTED", "NEGATIVE")

    # 7. PDF Report Generation
    pdf_bytes = generate_dental_pdf_report(
        original_rgb=val_res.rgb_array,
        enhanced_rgb=rgb_enh,
        segmentation_overlay_rgb=overlay,
        dental_arch=arch,
        pathology_results=pathology,
        filename=SAMPLE_RADIOGRAPH.name,
        analysis_id="DENT-INTEG-001",
    )
    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes.startswith(b"%PDF-")
    assert len(pdf_bytes) > 20000  # PDF containing embedded radiographs is >20KB
