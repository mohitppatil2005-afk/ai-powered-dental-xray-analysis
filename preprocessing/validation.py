"""
preprocessing/validation.py
===========================
Robust validation and normalization for panoramic dental X-ray images.

Checks:
- File format and integrity (corrupted headers, truncation).
- Minimum and maximum resolution constraints.
- Aspect ratio plausibility for panoramic radiographs.
- Bit-depth and channel normalization to 8-bit RGB and grayscale.
- Non-destructive preservation of original image data.
"""

from dataclasses import dataclass, field
from io import BytesIO
from typing import List, Optional, Tuple, Union
import numpy as np
from PIL import Image, ImageOps


SUPPORTED_FORMATS = {"PNG", "JPEG", "JPG", "TIFF", "TIF"}
MIN_IMAGE_DIMENSION = 256
MAX_IMAGE_DIMENSION = 8192
MIN_PANORAMIC_ASPECT_RATIO = 1.1   # Panoramic scans are wider than tall
MAX_PANORAMIC_ASPECT_RATIO = 3.5


@dataclass
class ImageValidationResult:
    is_valid: bool
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    original_pil: Optional[Image.Image] = None
    rgb_array: Optional[np.ndarray] = None
    grayscale_array: Optional[np.ndarray] = None
    width: int = 0
    height: int = 0
    format: str = ""


def validate_and_load_image(
    source: Union[str, bytes, BytesIO, Image.Image, np.ndarray],
    filename: Optional[str] = None,
) -> ImageValidationResult:
    """
    Validate, load, and normalize a dental radiograph safely.

    Preserves original image data without compression loss.
    Returns structured ImageValidationResult.
    """
    warnings: List[str] = []

    # 1. Handle np.ndarray directly
    if isinstance(source, np.ndarray):
        if source.size == 0:
            return ImageValidationResult(is_valid=False, error_message="Input array is empty.")
        if source.dtype != np.uint8:
            return ImageValidationResult(
                is_valid=False,
                error_message=f"Unsupported array dtype '{source.dtype}'. Expected uint8."
            )
        if source.ndim == 2:
            gray = source
            rgb = np.stack([gray, gray, gray], axis=-1)
        elif source.ndim == 3 and source.shape[2] == 3:
            rgb = source
            gray = (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]).astype(np.uint8)
        elif source.ndim == 3 and source.shape[2] == 4:
            rgb = source[:, :, :3]
            gray = (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]).astype(np.uint8)
            warnings.append("Alpha channel stripped from 4-channel image.")
        else:
            return ImageValidationResult(
                is_valid=False,
                error_message=f"Unsupported array shape '{source.shape}'. Expected 2D or 3-channel."
            )
        pil_img = Image.fromarray(rgb)
        h, w = rgb.shape[:2]
        return _validate_dimensions_and_return(pil_img, rgb, gray, w, h, "NUMPY", warnings)

    # 2. Handle PIL.Image directly
    if isinstance(source, Image.Image):
        try:
            pil_img = ImageOps.exif_transpose(source).convert("RGB")
            rgb = np.array(pil_img, dtype=np.uint8)
            gray = np.array(pil_img.convert("L"), dtype=np.uint8)
            w, h = pil_img.size
            fmt = source.format or "PIL"
            return _validate_dimensions_and_return(pil_img, rgb, gray, w, h, fmt, warnings)
        except Exception as exc:
            return ImageValidationResult(is_valid=False, error_message=f"Corrupted PIL image: {exc}")

    # 3. Handle file path, bytes, or BytesIO
    try:
        if isinstance(source, (bytes, bytearray)):
            stream = BytesIO(source)
        elif isinstance(source, BytesIO):
            stream = source
        elif isinstance(source, str):
            stream = open(source, "rb")
        elif hasattr(source, "read"):
            # Streamlit UploadedFile or file-like object
            content = source.read()
            stream = BytesIO(content)
        else:
            return ImageValidationResult(
                is_valid=False,
                error_message=f"Unsupported input type '{type(source).__name__}'."
            )

        with Image.open(stream) as raw_img:
            img_format = (raw_img.format or "").upper()
            if img_format and img_format not in SUPPORTED_FORMATS:
                warnings.append(
                    f"Format '{img_format}' is unusual for panoramic radiography. "
                    f"Recommended: {', '.join(sorted(SUPPORTED_FORMATS))}."
                )

            # Transpose according to EXIF tags to preserve true orientation
            pil_img = ImageOps.exif_transpose(raw_img).convert("RGB")
            w, h = pil_img.size
            rgb = np.array(pil_img, dtype=np.uint8)
            gray = np.array(pil_img.convert("L"), dtype=np.uint8)

        return _validate_dimensions_and_return(pil_img, rgb, gray, w, h, img_format, warnings)

    except Exception as exc:
        return ImageValidationResult(
            is_valid=False,
            error_message=f"Failed to decode or parse image file: {exc}"
        )


def _validate_dimensions_and_return(
    pil_img: Image.Image,
    rgb: np.ndarray,
    gray: np.ndarray,
    width: int,
    height: int,
    fmt: str,
    warnings: List[str],
) -> ImageValidationResult:
    """Validate image dimensions and aspect ratio."""
    if width < MIN_IMAGE_DIMENSION or height < MIN_IMAGE_DIMENSION:
        return ImageValidationResult(
            is_valid=False,
            error_message=(
                f"Image resolution ({width}×{height}) is too low for reliable dental analysis. "
                f"Minimum required is {MIN_IMAGE_DIMENSION}×{MIN_IMAGE_DIMENSION} pixels."
            ),
        )

    if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
        return ImageValidationResult(
            is_valid=False,
            error_message=(
                f"Image resolution ({width}×{height}) exceeds maximum allowable dimension "
                f"({MAX_IMAGE_DIMENSION} pixels)."
            ),
        )

    aspect_ratio = width / height
    if aspect_ratio < MIN_PANORAMIC_ASPECT_RATIO or aspect_ratio > MAX_PANORAMIC_ASPECT_RATIO:
        warnings.append(
            f"Aspect ratio ({aspect_ratio:.2f}) differs from standard panoramic radiographs "
            f"(expected ~{MIN_PANORAMIC_ASPECT_RATIO}–{MAX_PANORAMIC_ASPECT_RATIO}). "
            "Ensure the radiograph is a full panoramic (OPG) projection."
        )

    return ImageValidationResult(
        is_valid=True,
        warnings=warnings,
        original_pil=pil_img,
        rgb_array=rgb,
        grayscale_array=gray,
        width=width,
        height=height,
        format=fmt,
    )
