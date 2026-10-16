
import numpy as np
import pytest

from preprocessing.enhancement import enhance_xray


def test_grayscale_image():
    image = np.random.randint(
        0, 256, (256, 256), dtype=np.uint8
    )

    result = enhance_xray(image)

    assert result.shape == image.shape
    assert result.dtype == np.uint8


def test_color_image():
    image = np.random.randint(
        0, 256, (128, 128, 3), dtype=np.uint8
    )

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
