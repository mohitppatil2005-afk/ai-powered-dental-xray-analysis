import numpy as np
import pytest
import torch

from modules.segmentation.config import (
    MODEL_CLASSES,
    NUM_CLASSES,
    RESTORATION_CLASSES,
    get_anatomical_name,
)
from modules.segmentation.model import (
    build_maskrcnn_model,
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
