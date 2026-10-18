"""
modules/analysis_status.py
==========================
High-level module status reporting for user interface and text summaries.
"""

from modules.pathology.registry import TARGET_PATHOLOGIES


ANALYSIS_MODULES = {
    "Tooth segmentation & FDI numbering": {
        "configured": True,
        "description": "Instance segmentation (Mask R-CNN ResNet-50) with Anatomical FDI Resolver.",
        "validation_status": "Research prototype — validation metrics pending independent review.",
    },
    "Dental caries": {
        "configured": False,
        "description": TARGET_PATHOLOGIES["dental_caries"]["description"],
        "validation_status": TARGET_PATHOLOGIES["dental_caries"]["validation_limitation"],
    },
    "Periodontal bone loss": {
        "configured": False,
        "description": TARGET_PATHOLOGIES["periodontal_bone_loss"]["description"],
        "validation_status": TARGET_PATHOLOGIES["periodontal_bone_loss"]["validation_limitation"],
    },
    "Periapical abnormalities": {
        "configured": False,
        "description": TARGET_PATHOLOGIES["periapical_lesion"]["description"],
        "validation_status": TARGET_PATHOLOGIES["periapical_lesion"]["validation_limitation"],
    },
    "Impacted teeth": {
        "configured": False,
        "description": TARGET_PATHOLOGIES["impacted_teeth"]["description"],
        "validation_status": TARGET_PATHOLOGIES["impacted_teeth"]["validation_limitation"],
    },
}


def get_module_status():
    return {
        name: {
            **details,
            "status": (
                f"Model configured — {details['validation_status']}"
                if details["configured"]
                else f"Model not configured — {details['validation_status']}"
            ),
        }
        for name, details in ANALYSIS_MODULES.items()
    }
