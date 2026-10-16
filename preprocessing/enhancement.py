
import cv2
import numpy as np


def enhance_xray(image: np.ndarray) -> np.ndarray:
    """
    Enhance a dental X-ray using CLAHE.

    Accepts a grayscale or 3-channel BGR image.
    Returns an enhanced grayscale image.
    """
    if image is None or image.size == 0:
        raise ValueError("Input image is empty.")

    if image.dtype != np.uint8:
        raise ValueError("Input image must have uint8 pixel values.")

    if image.ndim == 2:
        gray = image

    elif image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    else:
        raise ValueError(
            "Expected a grayscale or 3-channel image."
        )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray)

    return enhanced
