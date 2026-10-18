"""
modules/segmentation/visualization.py
=====================================
Clean, legible visual overlays for tooth instance masks and FDI numbering.

Designed for clinical readability:
- Quadrant-specific distinct color hues.
- Translucent mask blending (alpha = 0.40) to preserve underlying trabecular bone
  and anatomical details.
- Anti-aliased contour boundaries without thick, image-obscuring bounding boxes.
- Contrasting, legible text pills at centroids.
"""

from typing import List, Optional, Tuple
import cv2
import numpy as np
from PIL import Image

from modules.segmentation.config import QUADRANT_COLORS
from modules.segmentation.fdi_resolver import ResolvedTooth


def create_segmentation_overlay(
    image_rgb: np.ndarray,
    teeth: List[ResolvedTooth],
    restorations: Optional[List[ResolvedTooth]] = None,
    unassigned: Optional[List[ResolvedTooth]] = None,
    alpha: float = 0.35,
    show_labels: bool = True,
    show_contours: bool = True,
) -> np.ndarray:
    """
    Render non-destructive tooth segmentation masks and FDI labels onto an image.

    Returns:
    - RGB uint8 numpy array with overlays applied.
    """
    if image_rgb is None or image_rgb.size == 0:
        raise ValueError("Cannot render overlay on an empty image.")

    H, W = image_rgb.shape[:2]
    canvas = image_rgb.copy()
    overlay_mask = np.zeros_like(image_rgb, dtype=np.uint8)
    contour_image = np.zeros_like(image_rgb, dtype=np.uint8)

    all_instances: List[ResolvedTooth] = []
    if teeth:
        all_instances.extend(teeth)
    if restorations:
        all_instances.extend(restorations)
    if unassigned:
        all_instances.extend(unassigned)

    # 1. Draw colored mask areas
    for inst in all_instances:
        if inst.mask is None or inst.mask.shape[:2] != (H, W):
            continue

        quad = inst.quadrant
        color = QUADRANT_COLORS.get(quad, QUADRANT_COLORS['XX'])

        # Fill mask
        overlay_mask[inst.mask] = color

        # Draw contour boundary
        if show_contours:
            mask_u8 = inst.mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(contour_image, contours, -1, color, 2, cv2.LINE_AA)

    # Alpha blend masks with original radiograph
    has_mask = np.any(overlay_mask > 0, axis=-1)
    blended = canvas.copy()
    blended[has_mask] = cv2.addWeighted(canvas, 1.0 - alpha, overlay_mask, alpha, 0)[has_mask]

    # Overlay sharp contour borders
    has_contour = np.any(contour_image > 0, axis=-1)
    blended[has_contour] = contour_image[has_contour]

    # 2. Draw clean FDI text labels
    if show_labels:
        for inst in all_instances:
            label_text = ""
            if inst.fdi_number:
                label_text = str(inst.fdi_number)
            elif inst.is_restoration and inst.restoration_type:
                # Abbreviate restoration for readability
                short_names = {
                    'Bridge': 'BRG',
                    'Filling-Crown': 'FIL/CRN',
                    'Implant': 'IMP',
                }
                label_text = short_names.get(inst.restoration_type, inst.restoration_type[:3].upper())
            elif inst.status == 'UNCERTAIN':
                label_text = "?"

            if not label_text:
                continue

            cx, cy = inst.centroid
            # Clamp centroid so label stays within image bounds
            cx = max(20, min(W - 30, cx))
            cy = max(20, min(H - 20, cy))

            _draw_label_pill(blended, label_text, (cx, cy))

    return blended


def _draw_label_pill(
    image: np.ndarray,
    text: str,
    position: Tuple[int, int],
    font_scale: float = 0.50,
    thickness: int = 1,
) -> None:
    """Draw a readable text pill with a dark background and crisp white lettering."""
    font = cv2.FONT_HERSHEY_DUPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    cx, cy = position
    x1 = cx - text_w // 2 - 4
    y1 = cy - text_h // 2 - 3
    x2 = cx + text_w // 2 + 4
    y2 = cy + text_h // 2 + baseline + 2

    H, W = image.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(W, x2)
    y2 = min(H, y2)

    # Dark rounded-rectangle background
    cv2.rectangle(image, (x1, y1), (x2, y2), (20, 20, 20), -1)
    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 1, cv2.LINE_AA)

    # White text centered
    text_x = x1 + 4
    text_y = y1 + text_h + 1
    cv2.putText(
        image,
        text,
        (text_x, text_y),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        lineType=cv2.LINE_AA,
    )
