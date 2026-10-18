from io import BytesIO
import numpy as np
from PIL import Image
import pytest

from preprocessing.validation import validate_and_load_image, ImageValidationResult


def test_valid_numpy_grayscale():
    arr = np.random.randint(0, 256, (300, 600), dtype=np.uint8)
    res = validate_and_load_image(arr)

    assert res.is_valid
    assert res.error_message is None
    assert res.width == 600
    assert res.height == 300
    assert res.rgb_array.shape == (300, 600, 3)
    assert res.grayscale_array.shape == (300, 600)


def test_valid_numpy_rgb():
    arr = np.random.randint(0, 256, (300, 700, 3), dtype=np.uint8)
    res = validate_and_load_image(arr)

    assert res.is_valid
    assert res.width == 700
    assert res.height == 300


def test_empty_input():
    res = validate_and_load_image(np.array([], dtype=np.uint8))
    assert not res.is_valid
    assert "empty" in res.error_message.lower()


def test_invalid_dtype():
    arr = np.random.rand(300, 600).astype(np.float32)
    res = validate_and_load_image(arr)
    assert not res.is_valid
    assert "uint8" in res.error_message.lower()


def test_low_resolution_rejection():
    arr = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    res = validate_and_load_image(arr)
    assert not res.is_valid
    assert "too low" in res.error_message.lower()


def test_aspect_ratio_warning():
    # Square image (aspect ratio 1.0 < 1.1)
    arr = np.random.randint(0, 256, (400, 400), dtype=np.uint8)
    res = validate_and_load_image(arr)
    assert res.is_valid  # Still valid, but produces warning
    assert len(res.warnings) > 0
    assert "aspect ratio" in res.warnings[0].lower()


def test_bytes_stream_loading():
    # Create valid in-memory PNG bytes
    pil_img = Image.new("RGB", (600, 300), color=(128, 128, 128))
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)

    res = validate_and_load_image(buf)
    assert res.is_valid
    assert res.width == 600
    assert res.height == 300
    assert res.format == "PNG"


def test_corrupted_bytes():
    res = validate_and_load_image(b"not_an_image_file_content")
    assert not res.is_valid
    assert "failed to decode" in res.error_message.lower()
