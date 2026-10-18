"""
tests/test_yolo_backends.py
===========================
Tests for the public pretrained YOLO backend integration:
- Universal (Tooth_1..32) -> FDI mapping and model class ids.
- Segmentation backend selection (auto/yolov8/maskrcnn/none + env vars).
- Download idempotency (existing weights are never re-fetched).
- YOLO pathology detector registry integration and honest bone-loss exclusion.
- Live analysis_status reporting against the real registry state.
"""

import numpy as np
import pytest

from modules.segmentation.config import (
    MODEL_CLASSES,
    RESTORATION_CLASSES,
    fdi_to_model_class_id,
    universal_tooth_to_fdi,
)
from modules.segmentation.backend import (
    resolve_segmentation_backend,
)
from modules.segmentation.yolo_adapter import (
    download_yolo_weights,
    resolve_yolo_inference_device,
    yolo_weights_info,
)
from modules.pathology import registry as pathology_registry
from modules.pathology.registry import (
    evaluate_condition,
    get_registered_detector,
)
from modules.pathology.yolo_detector import (
    PATHOLOGY_CLASS_TO_CONDITION,
    pathology_weights_info,
    register_yolo_pathology_detectors,
)
from modules.pathology.types import PathologyStatus
from modules.analysis_status import get_module_status


# ---------------------------------------------------------------------------
# Universal -> FDI mapping
# ---------------------------------------------------------------------------

# Expected mapping at quadrant boundaries (universal index -> FDI).
_BOUNDARY_MAPPING = {
    1: "18",   # Upper right 3rd molar
    8: "11",   # Upper right central incisor
    9: "21",   # Upper left central incisor
    16: "28",  # Upper left 3rd molar
    17: "38",  # Lower left 3rd molar
    24: "31",  # Lower left central incisor
    25: "41",  # Lower right central incisor
    32: "48",  # Lower right 3rd molar
}


@pytest.mark.parametrize("universal_idx,expected_fdi", sorted(_BOUNDARY_MAPPING.items()))
def test_universal_to_fdi_boundary_mapping(universal_idx, expected_fdi):
    assert universal_tooth_to_fdi(universal_idx) == expected_fdi


def test_universal_to_fdi_covers_all_32_teeth_exactly_once():
    mapped = [universal_tooth_to_fdi(i) for i in range(1, 33)]
    assert all(m is not None for m in mapped)
    assert len(set(mapped)) == 32, "FDI mapping must be a bijection over 1..32"

    # Quadrant counts must be exactly 8 each.
    for prefix in ("1", "2", "3", "4"):
        assert sum(1 for m in mapped if m.startswith(prefix)) == 8


def test_universal_to_fdi_out_of_range_returns_none():
    assert universal_tooth_to_fdi(0) is None
    assert universal_tooth_to_fdi(33) is None
    assert universal_tooth_to_fdi(-5) is None


def test_fdi_to_model_class_id_matches_config():
    # Every mapped FDI must resolve to a MODEL_CLASSES entry that is a tooth
    # (not a restoration class).
    for i in range(1, 33):
        fdi = universal_tooth_to_fdi(i)
        class_id = fdi_to_model_class_id(fdi)
        assert class_id > 0
        class_name = MODEL_CLASSES[class_id]
        assert class_name == fdi
        assert class_name not in RESTORATION_CLASSES

    assert fdi_to_model_class_id("99") == -1
    assert fdi_to_model_class_id("Implant") == MODEL_CLASSES.index("Implant")


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------

@pytest.fixture
def _clear_backend_env(monkeypatch):
    for var in ("DENTAL_SEG_BACKEND", "DENTAL_SEG_WEIGHTS", "DENTAL_SEG_YOLO_WEIGHTS"):
        monkeypatch.delenv(var, raising=False)


def test_backend_auto_prefers_yolov8_when_weights_exist(tmp_path, monkeypatch, _clear_backend_env):
    fake_yolo = tmp_path / "yolo_best.pt"
    fake_yolo.write_bytes(b"fake")
    monkeypatch.setenv("DENTAL_SEG_YOLO_WEIGHTS", str(fake_yolo))

    info = resolve_segmentation_backend()
    assert info.name == "yolov8"
    assert info.supports_restorations is False
    assert info.weights_path == str(fake_yolo)


def test_backend_auto_falls_back_to_maskrcnn(tmp_path, monkeypatch, _clear_backend_env):
    missing_yolo = tmp_path / "missing_yolo.pt"
    fake_ckpt = tmp_path / "best.pth"
    fake_ckpt.write_bytes(b"fake")
    monkeypatch.setenv("DENTAL_SEG_YOLO_WEIGHTS", str(missing_yolo))
    monkeypatch.setenv("DENTAL_SEG_WEIGHTS", str(fake_ckpt))

    info = resolve_segmentation_backend()
    assert info.name == "maskrcnn"
    assert info.supports_restorations is True


def test_backend_auto_none_when_no_weights(tmp_path, monkeypatch, _clear_backend_env):
    monkeypatch.setenv("DENTAL_SEG_YOLO_WEIGHTS", str(tmp_path / "nope.pt"))
    monkeypatch.setenv("DENTAL_SEG_WEIGHTS", str(tmp_path / "nope.pth"))

    info = resolve_segmentation_backend()
    assert info.name is None


def test_backend_none_disables_segmentation(tmp_path, monkeypatch, _clear_backend_env):
    fake_yolo = tmp_path / "yolo_best.pt"
    fake_yolo.write_bytes(b"fake")
    monkeypatch.setenv("DENTAL_SEG_YOLO_WEIGHTS", str(fake_yolo))

    info = resolve_segmentation_backend("none")
    assert info.name is None


def test_explicit_backend_overrides_env_var(tmp_path, monkeypatch, _clear_backend_env):
    fake_yolo = tmp_path / "yolo_best.pt"
    fake_yolo.write_bytes(b"fake")
    monkeypatch.setenv("DENTAL_SEG_YOLO_WEIGHTS", str(fake_yolo))

    # Env says auto (yolov8 would win), but explicit "none" wins.
    info = resolve_segmentation_backend("none")
    assert info.name is None


def test_invalid_backend_choice_falls_back_to_auto(tmp_path, monkeypatch, _clear_backend_env):
    fake_yolo = tmp_path / "yolo_best.pt"
    fake_yolo.write_bytes(b"fake")
    monkeypatch.setenv("DENTAL_SEG_YOLO_WEIGHTS", str(fake_yolo))

    info = resolve_segmentation_backend("bogus")
    assert info.name == "yolov8"


# ---------------------------------------------------------------------------
# Download idempotency
# ---------------------------------------------------------------------------

def test_download_yolo_weights_idempotent(tmp_path, _clear_backend_env):
    existing = tmp_path / "best.pt"
    existing.write_bytes(b"already-here")

    # With weights already present and force=False, no network call is made;
    # the existing path is returned unchanged.
    result = download_yolo_weights(dest_path=existing, force=False)
    assert result == existing
    assert existing.read_bytes() == b"already-here"


# ---------------------------------------------------------------------------
# Inference-device provenance
# ---------------------------------------------------------------------------


def test_resolve_yolo_inference_device_prefers_model_device():
    class _CudaLike:
        def __str__(self):
            return "cuda:0"

    class _CpuLike:
        def __str__(self):
            return "cpu"

    class _FakeModel:
        def __init__(self, device):
            self.device = device

    # A model that really sits on CUDA must be reported as CUDA.
    assert resolve_yolo_inference_device(_FakeModel(_CudaLike())) == "cuda:0"
    # A model on CPU must be reported as CPU.
    assert resolve_yolo_inference_device(_FakeModel(_CpuLike())) == "cpu"


def test_resolve_yolo_inference_device_never_claims_gpu_on_cpu_host():
    # No model device info and no usable CUDA on this host -> "cpu".
    assert resolve_yolo_inference_device(None) == "cpu"

    class _NoDevice:
        pass

    assert resolve_yolo_inference_device(_NoDevice()) == "cpu"


def test_yolo_weights_info_shape(_clear_backend_env):
    info = yolo_weights_info()
    for key in ("backend", "repo", "url", "license", "approx_size_mb", "weights_path", "available"):
        assert key in info
    assert info["backend"] == "yolov8"
    assert info["available"] in (True, False)


# ---------------------------------------------------------------------------
# Pathology detector registry integration
# ---------------------------------------------------------------------------

@pytest.fixture
def _isolate_pathology_registry():
    """Snapshot and restore the global detector registry around a test."""
    saved = dict(pathology_registry._DETECTOR_REGISTRY)
    yield
    pathology_registry._DETECTOR_REGISTRY.clear()
    pathology_registry._DETECTOR_REGISTRY.update(saved)


def test_register_yolo_pathology_detectors_registers_expected_conditions(_isolate_pathology_registry):
    pathology_registry._DETECTOR_REGISTRY.clear()

    registered = register_yolo_pathology_detectors()

    assert set(registered) == set(PATHOLOGY_CLASS_TO_CONDITION.values())
    for code in registered:
        assert get_registered_detector(code) is not None

    # Periodontal bone loss must never be registered (no public weights exist).
    assert "periodontal_bone_loss" not in registered
    assert get_registered_detector("periodontal_bone_loss") is None


def test_bone_loss_evaluation_stays_not_configured(_isolate_pathology_registry):
    pathology_registry._DETECTOR_REGISTRY.clear()
    register_yolo_pathology_detectors()

    img = np.zeros((64, 64, 3), dtype=np.uint8)
    res = evaluate_condition("periodontal_bone_loss", img)
    assert res.is_configured is False
    assert res.status == PathologyStatus.NOT_CONFIGURED


def test_pathology_weights_info_reports_uncovered_bone_loss():
    info = pathology_weights_info()
    assert "periodontal_bone_loss" in info["uncovered_conditions"]
    assert "periodontal_bone_loss" not in info["covered_conditions"]


# ---------------------------------------------------------------------------
# analysis_status live reporting
# ---------------------------------------------------------------------------

def test_analysis_status_reflects_registry_state(_isolate_pathology_registry):
    pathology_registry._DETECTOR_REGISTRY.clear()

    statuses = get_module_status()
    assert statuses["Dental caries"]["configured"] is False
    assert statuses["Periapical abnormalities"]["configured"] is False
    assert statuses["Impacted teeth"]["configured"] is False

    # Bone loss is always unavailable regardless of registration.
    register_yolo_pathology_detectors()
    statuses = get_module_status()
    assert statuses["Periodontal bone loss"]["configured"] is False
    assert "unavailable" in statuses["Periodontal bone loss"]["validation_status"].lower()

    # Registered conditions flip to configured with an honest screening caveat.
    assert statuses["Dental caries"]["configured"] is True
    assert "screening" in statuses["Dental caries"]["validation_status"].lower()

    # Every module must carry a non-empty status string.
    for details in statuses.values():
        assert details["status"]
