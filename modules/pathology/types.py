"""
modules/pathology/types.py
==========================
Data models and enumeration types for dental pathology analysis.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple
import numpy as np


class PathologyStatus(str, Enum):
    NOT_CONFIGURED = "NOT_CONFIGURED"
    UNVALIDATED = "UNVALIDATED"
    NEGATIVE = "NEGATIVE"
    DETECTED = "DETECTED"
    UNCERTAIN = "UNCERTAIN"


@dataclass
class PathologyFinding:
    """Individual localized pathology finding."""
    condition_code: str
    condition_name: str
    confidence: float
    bbox: Optional[Tuple[int, int, int, int]] = None  # (x1, y1, x2, y2)
    associated_tooth: Optional[str] = None           # E.g. "16", "38"
    description: str = ""
    status: PathologyStatus = PathologyStatus.DETECTED
    mask: Optional[np.ndarray] = None               # (H, W) bone probability map


@dataclass
class ConditionEvaluationResult:
    """Evaluation result for an individual target pathology condition."""
    condition_code: str
    condition_name: str
    is_configured: bool
    is_evaluated: bool
    status: PathologyStatus
    status_display: str
    status_explanation: str
    findings: List[PathologyFinding] = field(default_factory=list)
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    validation_limitation: str = ""
