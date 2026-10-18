import numpy as np
import pytest

from preprocessing.enhancement import enhance_xray, enhance_rgb_xray


def test_grayscale_image():
    image = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
    result = enhance_xray(image)

    assert result.shape == image.shape
    assert result.dtype == np.uint8


def test_color_image():
    image = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
    result = enhance_xray(image)

    assert result.shape == (128, 128)
    assert result.dtype == np.uint8


def test_empty_image():
    with pytest.raises(ValueError):
        enhance_xray(np.array([], dtype=np.uint8))


def test_unsupported_channels():
    image = np.zeros((32, 32, 4), dtype=np.uint8)
    with pytest.raises(ValueError):
        enhance_xray(image)


def test_configurable_parameters():
    image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    res1 = enhance_xray(image, clip_limit=1.5, tile_grid_size=(4, 4))
    res2 = enhance_xray(image, clip_limit=4.0, tile_grid_size=(16, 16))

    assert res1.shape == (100, 100)
    assert res2.shape == (100, 100)
    # Different clip limits produce different contrast curves
    assert not np.array_equal(res1, res2)


def test_invalid_parameters():
    image = np.zeros((64, 64), dtype=np.uint8)
    with pytest.raises(ValueError, match="clip_limit must be positive"):
        enhance_xray(image, clip_limit=-1.0)
    with pytest.raises(ValueError, match="tile_grid_size must be a tuple"):
        enhance_xray(image, tile_grid_size=(0, 8))


def test_enhance_rgb_xray():
    rgb = np.random.randint(0, 256, (64, 128, 3), dtype=np.uint8)
    enhanced = enhance_rgb_xray(rgb, clip_limit=2.5)

    assert enhanced.shape == (64, 128, 3)
    assert enhanced.dtype == np.uint8
