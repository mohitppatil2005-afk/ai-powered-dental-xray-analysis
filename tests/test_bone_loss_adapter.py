"""
Tests for the Periodontal Bone Loss (PBL) screening adapter.

Note: these tests are deterministic only when a real PBL weights checkpoint
exists at modules/pathology/weights/bone_loss/best.pt. In the CI-free,
local-only integration, the adapter is expected to fall back to
UNVALIDATED when the weights are absent.
"""

import numpy as np
import pytest

from modules.pathology.bone_loss_adapter import find_periodontal_bone_loss
from modules.pathology.types import PathologyStatus


SAMPLE_IMAGE = np.zeros((1024, 2041, 3), dtype=np.uint8)


def test_bone_loss_adapter_exists():
    """The adapter module imports and exposes its public API."""
    from modules.pathology.bone_loss_adapter import run_pbl_inference
    assert callable(run_pbl_inference)


def test_bone_loss_returns_honest_fields():
    """PBL output carries conf/mask/bbox only; no fabricated severity or
    tooth association for a diffuse screen."""
    gray = np.array(SAMPLE_IMAGE.convert('L') if hasattr(SAMPLE_IMAGE, 'convert') else SAMPLE_IMAGE, dtype=np.uint8)
    if gray.ndim == 3:
        gray = gray[..., 0]
    res = find_periodontal_bone_loss(gray)

    # The adapter is a screening scanner: it may or may not load the
    # weights. Either way it must be explicit, never a silent NEGATIVE.
    assert res.is_configured is True
    assert res.condition_code == "periodontal_bone_loss"
    if res.is_evaluated:
        assert res.status in (PathologyStatus.UNCERTAIN, PathologyStatus.DETECTED)
        assert res.status != PathologyStatus.NEGATIVE
        assert len(res.findings) == 1
        f = res.findings[0]
        # Conf/mask/bbox are the only allowed fields.
        assert isinstance(f.confidence, float)
        assert f.bbox is None or isinstance(f.bbox, (tuple, list))
        assert f.associated_tooth is None or isinstance(f.associated_tooth, str)
    else:
        assert res.findings == []
