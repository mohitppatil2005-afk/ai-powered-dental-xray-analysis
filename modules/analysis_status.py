"""
modules/analysis_status.py
==========================
High-level module status reporting for user interface and text summaries.
Statuses reflect the real availability of registered model backends at call
time (segmentation backend + pathology detector registry).
"""

from modules.pathology.registry import TARGET_PATHOLOGIES, get_registered_detector
from modules.segmentation.backend import resolve_segmentation_backend

# Conditions actually covered by the public pretrained YOLO pathology detector.
YOLO_PATHOLOGY_CONDITIONS = {"dental_caries", "periapical_lesion", "impacted_teeth"}


def get_module_status():
    """
    Build a live status map for every analysis module.

    Segmentation status reflects whichever backend (yolov8/maskrcnn) resolves
    right now; pathology conditions are 'configured' only if a detector is
    registered for them. Periodontal bone loss has no public weights and is
    always reported unavailable.
    """
    seg_info = resolve_segmentation_backend()
    seg_configured = seg_info.name is not None
    seg_description = (
        f"Instance segmentation ({seg_info.source}) with Anatomical FDI Resolver."
        if seg_configured
        else "No segmentation backend configured (no weights/checkpoint found)."
    )

    def _patho_status(code):
        detector_registered = get_registered_detector(code) is not None
        return {
            "configured": detector_registered,
            "description": TARGET_PATHOLOGIES[code]["description"],
            "validation_status": (
                "Public pretrained YOLO11 screening detector (upstream-reported, "
                "not independently validated here); outputs are screening flags, not diagnoses."
                if detector_registered
                else TARGET_PATHOLOGIES[code]["validation_limitation"]
            ),
        }

    statuses = {
        "Tooth segmentation & FDI numbering": {
            "configured": seg_configured,
            "description": seg_description,
            "validation_status": (
                "Public pretrained model — upstream-reported metrics only, "
                "not independently validated in this repository."
                if seg_configured
                else "No model configured."
            ),
        },
        "Dental caries": _patho_status("dental_caries"),
        "Periodontal bone loss": {
            "configured": False,
            "description": TARGET_PATHOLOGIES["periodontal_bone_loss"]["description"],
            "validation_status": (
                "No public pretrained weights available for panoramic bone-loss "
                "detection; this capability is explicitly unavailable."
            ),
        },
        "Periapical abnormalities": _patho_status("periapical_lesion"),
        "Impacted teeth": _patho_status("impacted_teeth"),
    }

    for details in statuses.values():
        details["status"] = (
            f"Model configured — {details['validation_status']}"
            if details["configured"]
            else f"Model not configured — {details['validation_status']}"
        )
    return statuses
