import numpy as np
import os
import pytest
import tempfile
import torch

from modules.segmentation.config import (
    DEFAULT_CHECKPOINT_PATH,
    MODEL_CLASSES,
    NUM_CLASSES,
    RESTORATION_CLASSES,
    get_anatomical_name,
)
from modules.segmentation.model import (
    build_maskrcnn_model,
    load_segmentation_model,
    run_segmentation_inference,
)
from modules.segmentation.visualization import create_segmentation_overlay
from modules.segmentation.fdi_resolver import ResolvedTooth


def test_config_classes():
    assert NUM_CLASSES == 36
    assert len(MODEL_CLASSES) == 36
    assert MODEL_CLASSES[0] == '__background__'
    assert '11' in MODEL_CLASSES
    assert '48' in MODEL_CLASSES
    assert 'Bridge' in RESTORATION_CLASSES
    assert 'Filling-Crown' in RESTORATION_CLASSES
    assert 'Implant' in RESTORATION_CLASSES


def test_anatomical_naming():
    assert "Central Incisor" in get_anatomical_name("11")
    assert "Wisdom" in get_anatomical_name("18") or "3rd Molar" in get_anatomical_name("18")
    assert "Restoration: Bridge" == get_anatomical_name("Bridge")


def test_build_model_architecture():
    model = build_maskrcnn_model(num_classes=36)
    assert isinstance(model, torch.nn.Module)
    # Check predictor head shape
    cls_weight = model.roi_heads.box_predictor.cls_score.weight
    assert cls_weight.shape[0] == 36


def test_inference_coordinate_preservation():
    # Construct a test model
    model = build_maskrcnn_model(num_classes=36)
    model.eval()

    H, W = 320, 640
    dummy_rgb = np.random.randint(0, 256, (H, W, 3), dtype=np.uint8)

    result = run_segmentation_inference(
        model=model,
        image_rgb=dummy_rgb,
        device=torch.device("cpu"),
        conf_threshold=0.99,  # High threshold on untrained head
    )

    assert result.image_shape == (H, W)
    for det in result.detections:
        # Every mask must match the original image dimensions exactly
        assert det.mask.shape == (H, W)
        x1, y1, x2, y2 = det.bbox
        assert 0 <= x1 <= W
        assert 0 <= x2 <= W
        assert 0 <= y1 <= H
        assert 0 <= y2 <= H


def test_inference_invalid_inputs():
    model = build_maskrcnn_model(num_classes=36)
    model.eval()

    with pytest.raises(ValueError, match="empty image"):
        run_segmentation_inference(model, np.array([]))

    with pytest.raises(ValueError, match="Expected 3-channel"):
        run_segmentation_inference(model, np.zeros((100, 100), dtype=np.uint8))


def test_visualization_overlay():
    H, W = 300, 600
    img_rgb = np.full((H, W, 3), 100, dtype=np.uint8)

    # Create a resolved tooth
    mask = np.zeros((H, W), dtype=bool)
    mask[100:150, 200:250] = True
    tooth = ResolvedTooth(
        instance_id=1,
        fdi_number='11',
        anatomical_name='Central Incisor',
        quadrant='UR',
        is_restoration=False,
        restoration_type=None,
        confidence=0.95,
        mask=mask,
        bbox=(200, 100, 250, 150),
        centroid=(225, 125),
        area=2500,
        status='CONFIRMED',
        status_note='Verified',
    )

    overlay = create_segmentation_overlay(
        image_rgb=img_rgb,
        teeth=[tooth],
        show_labels=True,
        show_contours=True,
    )

    assert overlay.shape == (H, W, 3)
    assert overlay.dtype == np.uint8
    # Pixel inside mask should be modified by the overlay
    assert not np.array_equal(overlay[125, 225], img_rgb[125, 225])
    # Pixel far outside mask should remain unchanged
    assert np.array_equal(overlay[10, 10], img_rgb[10, 10])


# ── Checkpoint loading tests ─────────────────────────────────────────────────


from pathlib import Path


def test_default_checkpoint_path_does_not_exist():
    """Document the current state: no trained checkpoint is shipped with the repo.

    The .gitignore excludes *.pth/*.pt, and no checkpoint artifact is tracked in
    git. This test verifies the default path is absent so the unavailable-state
    behavior is exercised by default.
    """
    assert not DEFAULT_CHECKPOINT_PATH.exists(), (
        f"Unexpected: default checkpoint exists at {DEFAULT_CHECKPOINT_PATH}\n"
        "A shipped checkpoint would change the unavailable-state behavior.\n"
    )


def test_missing_checkpoint_raises_file_not_found():
    """Regression: load_segmentation_model() must raise FileNotFoundError
    (not a silent failure or random-weight substitution) when the checkpoint
    path does not exist.
    """
    nonexistent = Path(tempfile.gettempdir()) / "nonexistent_dental_checkpoint.pth"
    assert not nonexistent.exists()

    with pytest.raises(FileNotFoundError, match="not found"):
        load_segmentation_model(checkpoint_path=str(nonexistent))


def test_incompatible_checkpoint_raises_runtime_error():
    """Regression: an incompatible state dict (wrong class count / keys) must
    raise RuntimeError, not produce silent garbage output.
    """
    model = build_maskrcnn_model(num_classes=36)
    model.eval()

    # Create a state dict with wrong-shaped weights (e.g. 10 classes instead of 36)
    wrong_state_dict = {
        k.replace("roi_heads.box_predictor.cls_score.", "")
        if "roi_heads.box_predictor.cls_score" in k else k: v
        for k, v in model.state_dict().items()
    }
    # Overwrite the classifier weight to a wrong shape to force a mismatch
    wrong_state_dict["roi_heads.box_predictor.cls_score.weight"] = torch.zeros(
        10, model.roi_heads.box_predictor.cls_score.weight.shape[1]
    )

    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
        tmp_path = f.name
        torch.save(wrong_state_dict, tmp_path)

    try:
        with pytest.raises(RuntimeError, match="Failed to load checkpoint"):
            load_segmentation_model(checkpoint_path=tmp_path)
    finally:
        os.unlink(tmp_path)


def test_checkpoint_env_var_override(monkeypatch):
    """Regression: DENTAL_SEG_WEIGHTS env var overrides the default path.

    Uses a nonexistent path to verify the override is honored (FileNotFoundError
    references the env var path, not the default).
    """
    fake_path = "/tmp/nonexistent_dental_weights_override.pth"
    monkeypatch.setenv("DENTAL_SEG_WEIGHTS", fake_path)

    # Clear the cached model so the new path is used
    import modules.segmentation.model as model_module
    model_module._LOADED_MODEL = None
    model_module._LOADED_PATH = None

    try:
        with pytest.raises(FileNotFoundError, match=fake_path):
            load_segmentation_model()
    finally:
        monkeypatch.delenv("DENTAL_SEG_WEIGHTS", raising=False)
        model_module._LOADED_MODEL = None
        model_module._LOADED_PATH = None


def test_successful_load_with_valid_untrained_checkpoint():
    """Regression: a structurally valid but untrained checkpoint loads successfully.

    This verifies that load_segmentation_model() can load a state dict that
    matches the architecture, even though the weights are untrained (random).
    It does NOT imply segmentation accuracy — only that the loading path works.
    """
    model = build_maskrcnn_model(num_classes=36)
    model.eval()

    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
        tmp_path = f.name
        torch.save(model.state_dict(), tmp_path)

    try:
        loaded_model, device = load_segmentation_model(checkpoint_path=tmp_path)
        assert loaded_model is not None
        assert isinstance(loaded_model, torch.nn.Module)
        assert device.type == "cpu"
    finally:
        os.unlink(tmp_path)
        # Clear cache
        import modules.segmentation.model as model_module
        model_module._LOADED_MODEL = None
        model_module._LOADED_PATH = None
        model_module._LOADED_DEVICE = None
