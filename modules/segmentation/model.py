"""
modules/segmentation/model.py
=============================
Inference wrapper for the 36-class Torchvision Mask R-CNN dental tooth
instance segmentation model.

Responsibilities:
- Lazy, cached model loading with error handling.
- Flexible device dispatch (CUDA if available, automatic CPU fallback).
- Coordinate preservation: ensures masks and bounding boxes perfectly match
  the original input dimensions.
"""

from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torchvision.transforms.functional as F

from modules.segmentation.config import (
    DEFAULT_CHECKPOINT_PATH,
    MODEL_CLASSES,
    NUM_CLASSES,
    RESTORATION_CLASSES,
    get_anatomical_name,
)
from preprocessing.enhancement import enhance_rgb_xray


@dataclass
class RawDetection:
    """Individual raw instance detected by the neural network."""
    mask: np.ndarray              # 2D boolean array (H, W)
    bbox: Tuple[int, int, int, int] # (x1, y1, x2, y2)
    centroid: Tuple[int, int]     # (cx, cy)
    area: int
    score: float
    class_id: int
    class_name: str
    is_restoration: bool


@dataclass
class SegmentationInferenceResult:
    """Complete output of the segmentation model before anatomical post-processing."""
    detections: List[RawDetection] = field(default_factory=list)
    image_shape: Tuple[int, int] = (0, 0) # (H, W)
    inference_time: float = 0.0
    device_used: str = "cpu"
    checkpoint_loaded: str = ""


# In-memory singleton cache to prevent reloading weights on every interaction
_LOADED_MODEL = None
_LOADED_PATH = None
_LOADED_DEVICE = None


def get_default_device() -> torch.device:
    """Select CUDA if available and functional, otherwise fall back to CPU."""
    if torch.cuda.is_available():
        try:
            # Probe device memory to confirm it is usable
            _ = torch.zeros(1).cuda()
            return torch.device("cuda")
        except Exception:
            return torch.device("cpu")
    return torch.device("cpu")


def build_maskrcnn_model(num_classes: int = NUM_CLASSES) -> torch.nn.Module:
    """
    Construct the Mask R-CNN ResNet-50+FPN network architecture matching
    the trained 36-class checkpoint.
    """
    model = maskrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None,
        box_nms_thresh=0.5,
        box_score_thresh=0.05,
        box_detections_per_img=45,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    return model


def load_segmentation_model(
    checkpoint_path: Optional[Union[str, Path]] = None,
    device: Optional[torch.device] = None,
    force_reload: bool = False,
) -> Tuple[torch.nn.Module, torch.device]:
    """
    Load the 36-class Mask R-CNN checkpoint into memory.

    Caches the loaded model. Handles missing weights gracefully.
    """
    global _LOADED_MODEL, _LOADED_PATH, _LOADED_DEVICE

    target_path = Path(checkpoint_path) if checkpoint_path else DEFAULT_CHECKPOINT_PATH
    target_device = device if device is not None else get_default_device()

    if not force_reload and _LOADED_MODEL is not None:
        if str(_LOADED_PATH) == str(target_path) and _LOADED_DEVICE == target_device:
            return _LOADED_MODEL, _LOADED_DEVICE

    if not target_path.exists():
        raise FileNotFoundError(
            f"Segmentation model checkpoint not found at: {target_path}. "
            "Please verify that the checkpoint exists or set DENTAL_SEG_WEIGHTS."
        )

    model = build_maskrcnn_model(num_classes=NUM_CLASSES)
    try:
        state_dict = torch.load(target_path, map_location=target_device, weights_only=True)
        model.load_state_dict(state_dict)
    except Exception as exc:
        raise RuntimeError(f"Failed to load checkpoint state dict from {target_path}: {exc}")

    model.to(target_device)
    model.eval()

    _LOADED_MODEL = model
    _LOADED_PATH = target_path
    _LOADED_DEVICE = target_device

    return model, target_device


def run_segmentation_inference(
    model: torch.nn.Module,
    image_rgb: np.ndarray,
    device: Optional[torch.device] = None,
    conf_threshold: float = 0.30,
    mask_threshold: float = 0.50,
) -> SegmentationInferenceResult:
    """
    Execute instance segmentation on a full-resolution RGB radiograph.

    Preserves original image dimensions.
    """
    if image_rgb is None or image_rgb.size == 0:
        raise ValueError("Cannot run inference on an empty image.")
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"Expected 3-channel RGB image, got shape {image_rgb.shape}.")

    H, W = image_rgb.shape[:2]
    run_device = device or get_default_device()

    # Preprocess with CLAHE contrast enhancement matching training protocol
    try:
        enhanced_rgb = enhance_rgb_xray(image_rgb, clip_limit=2.0, tile_grid_size=(8, 8))
    except Exception:
        # Fallback to direct RGB if CLAHE encounters an unexpected issue
        enhanced_rgb = image_rgb

    tensor_img = F.to_tensor(enhanced_rgb).to(run_device)

    start_time = time.time()
    with torch.no_grad():
        outputs = model([tensor_img])
    inference_duration = time.time() - start_time

    output = outputs[0]
    boxes = output['boxes'].detach().cpu().numpy()
    scores = output['scores'].detach().cpu().numpy()
    labels = output['labels'].detach().cpu().numpy()
    raw_masks = output['masks'].detach().cpu().numpy()[:, 0]  # (N, H, W)

    # Filter by confidence threshold
    keep = scores >= conf_threshold
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]
    raw_masks = raw_masks[keep]

    detections: List[RawDetection] = []

    for i in range(len(scores)):
        score = float(scores[i])
        cls_id = int(labels[i])
        cls_name = MODEL_CLASSES[cls_id] if cls_id < len(MODEL_CLASSES) else "unknown"
        is_restoration = cls_name in RESTORATION_CLASSES

        # Binary thresholding on sigmoid mask output
        bin_mask = (raw_masks[i] > mask_threshold)
        mask_area = int(np.sum(bin_mask))
        if mask_area == 0:
            continue

        # Bounding box coordinates clamped to image dimensions
        x1, y1, x2, y2 = boxes[i]
        x1 = max(0, min(int(round(x1)), W - 1))
        y1 = max(0, min(int(round(y1)), H - 1))
        x2 = max(0, min(int(round(x2)), W))
        y2 = max(0, min(int(round(y2)), H))

        # Centroid from moments
        mask_u8 = bin_mask.astype(np.uint8)
        M = cv2.moments(mask_u8)
        if M['m00'] > 0:
            cx = int(round(M['m10'] / M['m00']))
            cy = int(round(M['m01'] / M['m00']))
        else:
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

        detections.append(
            RawDetection(
                mask=bin_mask,
                bbox=(x1, y1, x2, y2),
                centroid=(cx, cy),
                area=mask_area,
                score=score,
                class_id=cls_id,
                class_name=cls_name,
                is_restoration=is_restoration,
            )
        )

    return SegmentationInferenceResult(
        detections=detections,
        image_shape=(H, W),
        inference_time=inference_duration,
        device_used=str(run_device),
        checkpoint_loaded=str(_LOADED_PATH or "default"),
    )
