"""
preprocessing/enhancement.py
============================
Contrast-Limited Adaptive Histogram Equalization (CLAHE) for dental radiographs.

Provides configurable, reproducible contrast enhancement to improve tooth
boundary and bone trabecular visibility without introducing synthetic artifacts.
"""

from typing import Tuple, Union
import cv2
import numpy as np


def enhance_xray(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    Enhance a dental X-ray using CLAHE.

    Accepts:
    - 2D grayscale array (H, W), dtype uint8.
    - 3-channel BGR/RGB array (H, W, 3), dtype uint8.

    Returns:
    - Enhanced 2D grayscale array (H, W), dtype uint8.
    """
    if image is None or image.size == 0:
        raise ValueError("Input image is empty.")

    if image.dtype != np.uint8:
        raise ValueError("Input image must have uint8 pixel values.")

    if clip_limit <= 0:
        raise ValueError(f"clip_limit must be positive, got {clip_limit}.")

    if (
        not isinstance(tile_grid_size, (tuple, list))
        or len(tile_grid_size) != 2
        or tile_grid_size[0] <= 0
        or tile_grid_size[1] <= 0
    ):
        raise ValueError(
            f"tile_grid_size must be a tuple of two positive integers, got {tile_grid_size}."
        )

    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError("Expected a grayscale or 3-channel image.")

    clahe = cv2.createCLAHE(
        clipLimit=float(clip_limit),
        tileGridSize=(int(tile_grid_size[0]), int(tile_grid_size[1])),
    )

    enhanced = clahe.apply(gray)
    return enhanced


def enhance_rgb_xray(
    image_rgb: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    Apply CLAHE to a 3-channel RGB image in LAB color space.

    Enhances luminance (L channel) while preserving chrominance and spatial coordinates.
    Returns enhanced 3-channel RGB array (H, W, 3), dtype uint8.
    """
    if image_rgb is None or image_rgb.size == 0:
        raise ValueError("Input image is empty.")

    if image_rgb.dtype != np.uint8 or image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("Input must be a 3-channel uint8 RGB image.")

    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=float(clip_limit),
        tileGridSize=(int(tile_grid_size[0]), int(tile_grid_size[1])),
    )
    enhanced_l = clahe.apply(l_channel)

    enhanced_lab = cv2.merge([enhanced_l, a_channel, b_channel])
    enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
    return enhanced_rgb
