"""
modules/pathology
=================
Modular dental pathology detection and condition registry.
"""

from modules.pathology.types import (
    PathologyFinding,
    PathologyStatus,
    ConditionEvaluationResult,
)
from modules.pathology.registry import (
    TARGET_PATHOLOGIES,
    evaluate_all_pathologies,
    evaluate_condition,
    register_pathology_detector,
)

__all__ = [
    "PathologyFinding",
    "PathologyStatus",
    "ConditionEvaluationResult",
    "TARGET_PATHOLOGIES",
    "evaluate_all_pathologies",
    "evaluate_condition",
    "register_pathology_detector",
]
