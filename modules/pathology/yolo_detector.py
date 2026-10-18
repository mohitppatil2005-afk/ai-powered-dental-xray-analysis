"""
modules/pathology/yolo_detector.py
==================================
Adapter for the public pretrained YOLO11-nano dental pathology detector
(liodon-ai/dental-panoramic-detector, CC-BY-NC-4.0, ~5.2 MB).

Detected conditions (bounding-box object detection, NOT segmentation):
- caries -> dental_caries
- periapical_lesion -> periapical_lesion
- impacted_tooth -> impacted_teeth

Periodontal bone loss is intentionally NOT covered by this model and remains
explicitly unavailable.

Upstream reported (unverified) metrics: mAP50 0.622, mAP50-95 0.406,
Precision 0.630, Recall 0.614 on the DENTEX validation split. These are
screening aids, not diagnoses.
"""

import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from modules.pathology.registry import register_pathology_detector
from modules.pathology.types import (
    ConditionEvaluationResult,
    PathologyFinding,
    PathologyStatus,
)

# ---------------------------------------------------------------------------
# Public pretrained YOLO11 pathology detector configuration
# ---------------------------------------------------------------------------
PATHOLOGY_HF_REPO_ID = "liodon-ai/dental-panoramic-detector"
PATHOLOGY_WEIGHTS_APPROX_BYTES = 5_452_442  # best.pt ~5.2-5.5 MB

DEFAULT_PATHOLOGY_WEIGHTS_PATH = Path(
    os.getenv(
        "DENTAL_PATHO_YOLO_WEIGHTS",
        "models/weights/yolo11n_pathology/best.pt",
    )
)

# Recommended inference settings from the upstream model card.
PATHOLOGY_CONF_THRESHOLD = 0.45
PATHOLOGY_IOU_THRESHOLD = 0.35
PATHOLOGY_IMGSZ = 640

# Model class name -> internal condition code.
PATHOLOGY_CLASS_TO_CONDITION: Dict[str, str] = {
    "caries": "dental_caries",
    "periapical_lesion": "periapical_lesion",
    "impacted_tooth": "impacted_teeth",
}

CONDITION_TO_DISPLAY_NAME: Dict[str, str] = {
    "dental_caries": "Dental Caries",
    "periapical_lesion": "Periapical Lesion",
    "impacted_teeth": "Impacted Tooth",
}

# Upstream-reported limitations per condition (from the model card's own notes).
# These describe the upstream model card in general terms; they must never
# state numeric findings about the current image. Per-image numbers (counts,
# top confidence) are composed separately in the detector below.
_UPSTREAM_NOTES: Dict[str, str] = {
    "dental_caries": (
        "Upstream note (not this image): caries recall is reported as limited "
        "at panoramic resolution; treat any flag as a screening hint, not a count."
    ),
    "periapical_lesion": (
        "Upstream note (not this image): periapical-lesion detection is reported "
        "as limited by a small validation set; treat any flag as a prompt to "
        "look closer, not a diagnosis."
    ),
    "impacted_teeth": (
        "Upstream note (not this image): the model card describes "
        "impacted_tooth as its strongest class; treat any flag as a screening "
        "hint requiring professional review."
    ),
}

_VALIDATION_LIMITATION = (
    "Pretrained model trained on DENTEX + OralXrays-9 panoramic radiographs. "
    "Reported mAP50 0.622 on the DENTEX val split (46 images) - NOT independently "
    "reproduced in this repository. Outputs are screening flags requiring "
    "professional review. Absence of a detection does NOT imply absence of disease."
)

# Singleton YOLO pathology model.
_PATHO_MODEL = None
_PATHO_PATH: Optional[str] = None


def resolve_pathology_weights_path(
    explicit_path: Optional[Union[str, Path]] = None,
) -> Path:
    """Resolve pathology weights path; env var re-read on each call."""
    if explicit_path is not None:
        return Path(explicit_path)
    return Path(
        os.getenv("DENTAL_PATHO_YOLO_WEIGHTS", str(DEFAULT_PATHOLOGY_WEIGHTS_PATH))
    )


def pathology_model_available() -> bool:
    """True if pathology weights exist and ultralytics imports."""
    if not resolve_pathology_weights_path().exists():
        return False
    try:
        import ultralytics  # noqa: F401
    except Exception:
        return False
    return True


def download_pathology_weights(
    dest_path: Optional[Union[str, Path]] = None,
    force: bool = False,
) -> Path:
    """Download the ~5.2 MB YOLO11 pathology weights from Hugging Face."""
    target = resolve_pathology_weights_path(dest_path)
    if target.exists() and not force:
        return target

    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:
        raise RuntimeError(
            "huggingface_hub is required to download the pathology weights. "
            "Install it with: uv pip install huggingface_hub"
        ) from exc

    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        downloaded = hf_hub_download(
            repo_id=PATHOLOGY_HF_REPO_ID,
            filename="best.pt",
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download pathology weights from '{PATHOLOGY_HF_REPO_ID}': {exc}"
        ) from exc

    downloaded_path = Path(downloaded)
    if downloaded_path.resolve() != target.resolve():
        import shutil
        shutil.copy2(downloaded_path, target)
    return target


def load_pathology_model(
    weights_path: Optional[Union[str, Path]] = None,
    force_reload: bool = False,
):
    """Load and cache the Ultralytics YOLO pathology detection model."""
    global _PATHO_MODEL, _PATHO_PATH

    target = resolve_pathology_weights_path(weights_path)
    if not force_reload and _PATHO_MODEL is not None and _PATHO_PATH == str(target):
        return _PATHO_MODEL

    if not target.exists():
        raise FileNotFoundError(
            f"Pathology YOLO weights not found at: {target}. "
            "Run download_pathology_weights() or set DENTAL_PATHO_YOLO_WEIGHTS."
        )

    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError(
            "The 'ultralytics' package is required for the pathology detector. "
            "Install it with: uv pip install ultralytics"
        ) from exc

    try:
        model = YOLO(str(target))
    except Exception as exc:
        raise RuntimeError(f"Failed to load pathology weights from {target}: {exc}") from exc

    _PATHO_MODEL = model
    _PATHO_PATH = str(target)
    return model


def _associate_tooth(
    bbox: Tuple[int, int, int, int],
    dental_arch,
) -> Optional[str]:
    """
    Associate a pathology bounding box with the nearest resolved tooth centroid.

    Returns the FDI number string, or None if no arch/teeth are available.
    """
    if dental_arch is None or not getattr(dental_arch, "teeth", None):
        return None
    cx = (bbox[0] + bbox[2]) / 2.0
    cy = (bbox[1] + bbox[3]) / 2.0
    best_fdi = None
    best_dist = float("inf")
    for tooth in dental_arch.teeth:
        tx, ty = tooth.centroid
        dist = (cx - tx) ** 2 + (cy - ty) ** 2
        if dist < best_dist:
            best_dist = dist
            best_fdi = tooth.fdi_number
    return best_fdi


def run_pathology_inference(
    model,
    image_rgb: np.ndarray,
    dental_arch=None,
    conf_threshold: float = PATHOLOGY_CONF_THRESHOLD,
    iou_threshold: float = PATHOLOGY_IOU_THRESHOLD,
    imgsz: int = PATHOLOGY_IMGSZ,
) -> Dict[str, List[PathologyFinding]]:
    """
    Run YOLO pathology detection and bucket findings by condition code.

    Returns a dict mapping condition_code -> list of PathologyFinding.
    Conditions with no detections map to an empty list.
    """
    if image_rgb is None or image_rgb.size == 0:
        raise ValueError("Cannot run inference on an empty image.")

    results = model.predict(
        source=image_rgb,
        conf=conf_threshold,
        iou=iou_threshold,
        imgsz=imgsz,
        verbose=False,
    )
    result = results[0]

    findings: Dict[str, List[PathologyFinding]] = {
        code: [] for code in PATHOLOGY_CLASS_TO_CONDITION.values()
    }

    if result.boxes is None or len(result.boxes) == 0:
        return findings

    H, W = image_rgb.shape[:2]
    boxes_arr = result.boxes.xyxy.cpu().numpy()
    conf_arr = result.boxes.conf.cpu().numpy()
    cls_arr = result.boxes.cls.cpu().numpy()
    names = result.names  # dict {int: str}

    for i in range(len(cls_arr)):
        cls_name = str(names.get(int(cls_arr[i]), "")).strip().lower()
        condition_code = PATHOLOGY_CLASS_TO_CONDITION.get(cls_name)
        if condition_code is None:
            continue

        x1, y1, x2, y2 = boxes_arr[i]
        bbox = (
            max(0, min(int(round(x1)), W - 1)),
            max(0, min(int(round(y1)), H - 1)),
            max(0, min(int(round(x2)), W)),
            max(0, min(int(round(y2)), H)),
        )
        conf = float(conf_arr[i])
        associated_tooth = _associate_tooth(bbox, dental_arch)

        findings[condition_code].append(
            PathologyFinding(
                condition_code=condition_code,
                condition_name=CONDITION_TO_DISPLAY_NAME.get(condition_code, condition_code),
                confidence=conf,
                bbox=bbox,
                associated_tooth=associated_tooth,
                description=(
                    f"{CONDITION_TO_DISPLAY_NAME.get(condition_code, condition_code)} "
                    f"detected with {conf * 100:.0f}% confidence"
                    + (f" near tooth {associated_tooth}." if associated_tooth else ".")
                ),
                status=PathologyStatus.DETECTED,
            )
        )

    return findings


def make_pathology_detector(condition_code: str) -> Callable:
    """
    Build a detector callable for a single condition, compatible with the
    pathology registry's Callable[[np.ndarray], ConditionEvaluationResult].
    """
    if condition_code not in PATHOLOGY_CLASS_TO_CONDITION.values():
        raise ValueError(
            f"Condition '{condition_code}' is not covered by the YOLO pathology model."
        )

    display_name = CONDITION_TO_DISPLAY_NAME.get(condition_code, condition_code)

    def _detector(image_rgb: np.ndarray) -> ConditionEvaluationResult:
        try:
            model = load_pathology_model()
        except Exception as exc:
            return ConditionEvaluationResult(
                condition_code=condition_code,
                condition_name=display_name,
                is_configured=True,
                is_evaluated=False,
                status=PathologyStatus.UNVALIDATED,
                status_display="Model Load Failed",
                status_explanation=f"Could not load pathology detector: {exc}",
                validation_limitation=_VALIDATION_LIMITATION,
            )

        try:
            all_findings = run_pathology_inference(model, image_rgb)
        except Exception as exc:
            return ConditionEvaluationResult(
                condition_code=condition_code,
                condition_name=display_name,
                is_configured=True,
                is_evaluated=False,
                status=PathologyStatus.UNVALIDATED,
                status_display="Inference Failed",
                status_explanation=f"Pathology detector error: {exc}",
                validation_limitation=_VALIDATION_LIMITATION,
            )

        cond_findings = all_findings.get(condition_code, [])

        if cond_findings:
            top = max(f.confidence for f in cond_findings)
            status = PathologyStatus.DETECTED
            status_display = f"Detected ({len(cond_findings)} finding(s))"
            explanation = (
                f"On this image, {len(cond_findings)} region(s) for this condition "
                f"exceeded the screening threshold; highest confidence {top * 100:.0f}%. "
                + _UPSTREAM_NOTES.get(condition_code, "")
            )
        else:
            status = PathologyStatus.NEGATIVE
            status_display = "Not Detected (screening negative)"
            explanation = (
                "On this image, no region for this condition exceeded the "
                "screening confidence threshold. "
                "This does NOT rule out the condition; a negative model output is not a clinical negative."
            )

        return ConditionEvaluationResult(
            condition_code=condition_code,
            condition_name=display_name,
            is_configured=True,
            is_evaluated=True,
            status=status,
            status_display=status_display,
            status_explanation=explanation,
            findings=cond_findings,
            model_name="YOLO11n pathology detector (liodon-ai)",
            model_version=PATHOLOGY_HF_REPO_ID,
            validation_limitation=_VALIDATION_LIMITATION,
        )

    return _detector


def register_yolo_pathology_detectors() -> List[str]:
    """
    Register YOLO-backed detectors for all supported conditions with the
    pathology registry. Returns the list of condition codes registered.
    """
    registered: List[str] = []
    for condition_code in sorted(set(PATHOLOGY_CLASS_TO_CONDITION.values())):
        register_pathology_detector(
            condition_code, make_pathology_detector(condition_code)
        )
        registered.append(condition_code)
    return registered


def pathology_weights_info() -> dict:
    """Human-readable provenance for UI display."""
    path = resolve_pathology_weights_path()
    return {
        "repo": PATHOLOGY_HF_REPO_ID,
        "url": f"https://huggingface.co/{PATHOLOGY_HF_REPO_ID}",
        "license": "CC-BY-NC-4.0",
        "approx_size_mb": round(PATHOLOGY_WEIGHTS_APPROX_BYTES / 1e6, 1),
        "weights_path": str(path),
        "available": pathology_model_available(),
        "covered_conditions": sorted(set(PATHOLOGY_CLASS_TO_CONDITION.values())),
        "uncovered_conditions": ["periodontal_bone_loss"],
    }