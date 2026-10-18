import numpy as np
import pytest

from modules.pathology.types import (
    ConditionEvaluationResult,
    PathologyFinding,
    PathologyStatus,
)
from modules.pathology.registry import (
    TARGET_PATHOLOGIES,
    evaluate_all_pathologies,
    evaluate_condition,
    register_pathology_detector,
)
from modules.pathology import registry as _pathology_registry


@pytest.fixture(autouse=True)
def _isolated_detector_registry():
    """Keep pathology tests hermetic: other test modules (integration,
    YOLO backends) register real detectors in the global registry, so
    snapshot and restore it around every test here."""
    snapshot = dict(_pathology_registry._DETECTOR_REGISTRY)
    _pathology_registry._DETECTOR_REGISTRY.clear()
    try:
        yield
    finally:
        _pathology_registry._DETECTOR_REGISTRY.clear()
        _pathology_registry._DETECTOR_REGISTRY.update(snapshot)


def test_registered_pathologies_count():
    assert len(TARGET_PATHOLOGIES) == 4
    assert "dental_caries" in TARGET_PATHOLOGIES
    assert "periodontal_bone_loss" in TARGET_PATHOLOGIES
    assert "periapical_lesion" in TARGET_PATHOLOGIES
    assert "impacted_teeth" in TARGET_PATHOLOGIES


def test_unconfigured_condition_evaluation():
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    res = evaluate_condition("dental_caries", dummy_img)

    assert isinstance(res, ConditionEvaluationResult)
    assert res.condition_code == "dental_caries"
    assert res.is_configured is False
    assert res.is_evaluated is False
    assert res.status == PathologyStatus.NOT_CONFIGURED
    # Critical: Must not report negative or confirmed absence
    assert res.status != PathologyStatus.NEGATIVE
    assert "not configured" in res.status_display.lower()
    assert len(res.findings) == 0


def test_evaluate_all_pathologies():
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    results = evaluate_all_pathologies(dummy_img)

    assert len(results) == 4
    for code, res in results.items():
        assert res.is_evaluated is False
        assert res.status == PathologyStatus.NOT_CONFIGURED
        assert res.validation_limitation != ""


def test_custom_detector_registration():
    code = "impacted_teeth"

    def mock_detector(img):
        return ConditionEvaluationResult(
            condition_code=code,
            condition_name="Impacted Teeth",
            is_configured=True,
            is_evaluated=True,
            status=PathologyStatus.DETECTED,
            status_display="Detected",
            status_explanation="Mock detection",
            findings=[
                PathologyFinding(
                    condition_code=code,
                    condition_name="Impacted Teeth",
                    confidence=0.88,
                    associated_tooth="38",
                    description="Horizontal impaction",
                )
            ],
            model_name="MockDetector",
        )

    register_pathology_detector(code, mock_detector)
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    res = evaluate_condition(code, dummy_img)

    assert res.is_configured is True
    assert res.is_evaluated is True
    assert res.status == PathologyStatus.DETECTED
    assert len(res.findings) == 1
    assert res.findings[0].associated_tooth == "38"

    # Reset registry to None for this condition
    register_pathology_detector(code, None)
    res_reset = evaluate_condition(code, dummy_img)
    assert res_reset.is_configured is False


def test_bone_loss_has_no_detector_registered():
    """Guard: the caries/periapical/impacted detector must never be reused
    as a bone-loss detector. Bone loss stays NOT_CONFIGURED until a genuine,
    verified PBL model with downloadable weights is integrated."""
    from modules.pathology.yolo_detector import (
        PATHOLOGY_CLASS_TO_CONDITION,
        register_yolo_pathology_detectors,
    )
    from modules.pathology import registry as reg

    saved = dict(reg._DETECTOR_REGISTRY)
    reg._DETECTOR_REGISTRY.clear()
    try:
        registered = register_yolo_pathology_detectors()
        assert "periodontal_bone_loss" not in registered
        assert "periodontal_bone_loss" not in PATHOLOGY_CLASS_TO_CONDITION.values()
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        res = evaluate_condition("periodontal_bone_loss", dummy_img)
        assert res.is_configured is False
        assert res.status == PathologyStatus.NOT_CONFIGURED
        assert res.findings == []
    finally:
        reg._DETECTOR_REGISTRY.clear()
        reg._DETECTOR_REGISTRY.update(saved)
