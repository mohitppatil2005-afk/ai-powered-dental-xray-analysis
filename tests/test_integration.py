from pathlib import Path
import pytest
import numpy as np

from preprocessing.validation import validate_and_load_image
from preprocessing.enhancement import enhance_xray, enhance_rgb_xray
from modules.segmentation.model import (
    load_segmentation_model,
    run_segmentation_inference,
)
from modules.segmentation.fdi_resolver import resolve_dental_arch
from modules.segmentation.visualization import create_segmentation_overlay
from modules.pathology.registry import evaluate_all_pathologies
from utils.pdf_report import generate_dental_pdf_report


def test_full_pipeline_on_real_radiograph():
    image_path = Path(r"C:\Users\mohit\Downloads\AKUDENTAL\AKUDENTAL-main\AKUDENTAL\images\1.jpg")
    if not image_path.exists():
        pytest.skip("Test radiograph 1.jpg not found on disk.")

    # 1. Validation & Loading
    val_res = validate_and_load_image(str(image_path))
    assert val_res.is_valid
    assert val_res.width == 2943
    assert val_res.height == 1435

    # 2. Enhancement
    gray_enh = enhance_xray(val_res.grayscale_array)
    rgb_enh = enhance_rgb_xray(val_res.rgb_array)
    assert gray_enh.shape == val_res.grayscale_array.shape
    assert rgb_enh.shape == val_res.rgb_array.shape

    # 3. Model Loading & Inference
    model, device = load_segmentation_model()
    raw_res = run_segmentation_inference(
        model=model,
        image_rgb=rgb_enh,
        device=device,
        conf_threshold=0.30,
    )
    assert len(raw_res.detections) > 20

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

    # 6. Pathology Status
    pathology = evaluate_all_pathologies(val_res.rgb_array)
    assert len(pathology) == 4
    for p in pathology.values():
        assert p.is_evaluated is False
        assert p.status == "NOT_CONFIGURED"

    # 7. PDF Report Generation
    pdf_bytes = generate_dental_pdf_report(
        original_rgb=val_res.rgb_array,
        enhanced_rgb=rgb_enh,
        segmentation_overlay_rgb=overlay,
        dental_arch=arch,
        pathology_results=pathology,
        filename="1.jpg",
        analysis_id="DENT-INTEG-001",
    )
    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes.startswith(b"%PDF-")
    assert len(pdf_bytes) > 20000  # PDF containing embedded radiographs is >20KB
