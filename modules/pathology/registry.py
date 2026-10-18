"""
modules/pathology/registry.py
=============================
Registry of supported dental pathology conditions and their configuration metadata.
"""

from typing import Callable, Dict, Optional
import numpy as np

from modules.pathology.types import (
    ConditionEvaluationResult,
    PathologyStatus,
)


TARGET_PATHOLOGIES = {
    "dental_caries": {
        "code": "dental_caries",
        "name": "Dental Caries (Cavities)",
        "description": "Detection of coronal and proximal carious decay lesions.",
        "validation_limitation": (
            "AKUDENTAL dataset contains tooth anatomical polygons only and does not supply "
            "caries labels. No validated local detector is configured."
        ),
    },
    "periodontal_bone_loss": {
        "code": "periodontal_bone_loss",
        "name": "Periodontal Bone Loss (PBL)",
        "description": "Evaluation of alveolar bone crest resorption and bone support reduction.",
        "validation_limitation": (
            "Requires CEJ-to-alveolar crest ground-truth measurements or bitewing radiographs. "
            "No validated local model is configured."
        ),
    },
    "periapical_lesion": {
        "code": "periapical_lesion",
        "name": "Periapical Lesions (Apical Periodontitis)",
        "description": "Detection of radiolucent inflammatory lesions at root apices.",
        "validation_limitation": (
            "Requires differential diagnostic validation against normal anatomical landmarks "
            "(mental foramen, incisive canal, maxillary sinus). Not annotated in AKUDENTAL."
        ),
    },
    "impacted_teeth": {
        "code": "impacted_teeth",
        "name": "Impacted Teeth",
        "description": "Identification of unerupted or aberrantly angulated permanent teeth.",
        "validation_limitation": (
            "Requires angulation and depth validation relative to occlusal plane and ramus. "
            "No validated local model is configured."
        ),
    },
}

# Pluggable detector registry: condition_code -> Callable[[np.ndarray], ConditionEvaluationResult]
_DETECTOR_REGISTRY: Dict[str, Callable] = {}


def register_pathology_detector(condition_code: str, detector_func: Callable) -> None:
    """Register a custom detector function for a specific pathology condition."""
    if condition_code not in TARGET_PATHOLOGIES:
        raise ValueError(f"Unknown pathology condition '{condition_code}'.")
    _DETECTOR_REGISTRY[condition_code] = detector_func


def get_registered_detector(condition_code: str) -> Optional[Callable]:
    """Retrieve registered detector for a condition, or None if not configured."""
    return _DETECTOR_REGISTRY.get(condition_code)


def evaluate_condition(
    condition_code: str,
    image_rgb: np.ndarray,
) -> ConditionEvaluationResult:
    """
    Evaluate a specific condition using its registered detector, or return
    an explicit NOT_CONFIGURED result with honest clinical explanations.
    """
    meta = TARGET_PATHOLOGIES.get(condition_code)
    if not meta:
        raise ValueError(f"Unknown condition code '{condition_code}'.")

    detector = get_registered_detector(condition_code)
    if detector is not None:
        try:
            return detector(image_rgb)
        except Exception as exc:
            return ConditionEvaluationResult(
                condition_code=condition_code,
                condition_name=meta["name"],
                is_configured=True,
                is_evaluated=False,
                status=PathologyStatus.UNVALIDATED,
                status_display="Inference Failed",
                status_explanation=f"Detector error during execution: {exc}",
                validation_limitation=meta["validation_limitation"],
            )

    # When no detector is configured, return explicit unvalidated status
    return ConditionEvaluationResult(
        condition_code=condition_code,
        condition_name=meta["name"],
        is_configured=False,
        is_evaluated=False,
        status=PathologyStatus.NOT_CONFIGURED,
        status_display="Not Configured — Unvalidated",
        status_explanation=(
            "No validated model is configured for this condition. "
            "Absence of a detection does NOT imply absence of disease."
        ),
        validation_limitation=meta["validation_limitation"],
    )


def evaluate_all_pathologies(image_rgb: np.ndarray) -> Dict[str, ConditionEvaluationResult]:
    """Run evaluation across all four target pathologies."""
    return {
        code: evaluate_condition(code, image_rgb)
        for code in TARGET_PATHOLOGIES
    }
