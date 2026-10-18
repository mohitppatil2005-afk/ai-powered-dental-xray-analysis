"""
modules/segmentation/yolo_adapter.py
====================================
Adapter for the public pretrained YOLOv8-nano 32-tooth instance segmentation
model (promitbhattacharjee/dental-panoramic-xray-segmentation, AGPL-3.0).

Responsibilities:
- Weights resolution (DENTAL_SEG_YOLO_WEIGHTS env var, re-read per call),
  availability probing, and explicit on-demand download via huggingface_hub.
- Lazy, cached model loading through the `ultralytics` package.
- Conversion of Ultralytics outputs (per-tooth instance masks in "universal"
  Tooth_1..32 numbering) into the pipeline's existing RawDetection format
  with FDI class labels, so the downstream FDI resolver is preserved.

Honest limitations:
- This backend does NOT detect restorations (Bridge / Filling-Crown / Implant).
- Upstream reported metrics (mask mAP50 94.4%) are unverified in this repo.
"""

import os
import time
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

from modules.segmentation.config import (
    DEFAULT_YOLO_WEIGHTS_PATH,
    YOLO_HF_REPO_ID,
    YOLO_HF_WEIGHTS_FILENAME,
    YOLO_WEIGHTS_APPROX_BYTES,
    fdi_to_model_class_id,
    universal_tooth_to_fdi,
)
from modules.segmentation.model import RawDetection, SegmentationInferenceResult

# In-memory singleton cache for the YOLO model.
_YOLO_MODEL = None
_YOLO_PATH: Optional[str] = None


def resolve_yolo_inference_device(model=None) -> str:
    """Report the device the YOLO model actually runs inference on.

    Prefers the loaded model's own ``device`` attribute (the Ultralytics
    runtime moves the model to CUDA when available); falls back to a local
    ``torch.cuda.is_available()`` probe. Never claims GPU use unless the
    actual inference device is a GPU. Imported lazily so this module stays
    importable without heavy dependencies.
    """
    try:
        device = getattr(model, "device", None)
        if device is not None:
            device_str = str(device).lower()
            if device_str.startswith("cuda") or device_str == "cpu":
                return device_str
    except Exception:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def resolve_yolo_weights_path(explicit_path: Optional[Union[str, Path]] = None) -> Path:
    """Resolve the YOLO weights path; the env var is re-read on each call."""
    if explicit_path is not None:
        return Path(explicit_path)
    return Path(
        os.getenv("DENTAL_SEG_YOLO_WEIGHTS", str(DEFAULT_YOLO_WEIGHTS_PATH))
    )


def yolo_backend_available() -> bool:
    """True if the YOLO weights file exists and the ultralytics package imports."""
    if not resolve_yolo_weights_path().exists():
        return False
    try:
        import ultralytics  # noqa: F401
    except Exception:
        return False
    return True


def download_yolo_weights(
    dest_path: Optional[Union[str, Path]] = None,
    force: bool = False,
) -> Path:
    """
    Download the pretrained 32-tooth YOLOv8 weights (~7 MB) from Hugging Face.

    Requires `huggingface_hub`. Raises RuntimeError with guidance on failure.
    Returns the local path to the weights file.
    """
    target = resolve_yolo_weights_path(dest_path)
    if target.exists() and not force:
        return target

    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:
        raise RuntimeError(
            "huggingface_hub is required to download the YOLO weights. "
            "Install it with: uv pip install huggingface_hub"
        ) from exc

    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        downloaded = hf_hub_download(
            repo_id=YOLO_HF_REPO_ID,
            filename=YOLO_HF_WEIGHTS_FILENAME,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download YOLO weights from '{YOLO_HF_REPO_ID}': {exc}"
        ) from exc

    downloaded_path = Path(downloaded)
    if downloaded_path.resolve() != target.resolve():
        # Copy from the HF cache into the project-configured location.
        import shutil
        shutil.copy2(downloaded_path, target)
    return target


def load_yolo_model(
    weights_path: Optional[Union[str, Path]] = None,
    force_reload: bool = False,
):
    """
    Load and cache the Ultralytics YOLO segmentation model.

    Raises FileNotFoundError if weights are absent, RuntimeError if the
    ultralytics package is missing or loading fails.
    """
    global _YOLO_MODEL, _YOLO_PATH

    target = resolve_yolo_weights_path(weights_path)
    if not force_reload and _YOLO_MODEL is not None and _YOLO_PATH == str(target):
        return _YOLO_MODEL

    if not target.exists():
        raise FileNotFoundError(
            f"YOLO tooth segmentation weights not found at: {target}. "
            "Run download_yolo_weights() or set DENTAL_SEG_YOLO_WEIGHTS."
        )

    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError(
            "The 'ultralytics' package is required for the YOLO backend. "
            "Install it with: uv pip install ultralytics"
        ) from exc

    try:
        model = YOLO(str(target))
    except Exception as exc:
        raise RuntimeError(f"Failed to load YOLO weights from {target}: {exc}") from exc

    _YOLO_MODEL = model
    _YOLO_PATH = str(target)
    return model


def _masks_to_full_resolution(
    mask_data,
    image_shape: Tuple[int, int],
) -> List[np.ndarray]:
    """Upscale raw (N, h, w) mask probabilities to full (H, W) boolean masks."""
    H, W = image_shape
    masks_np = (
        mask_data.detach().cpu().numpy()
        if hasattr(mask_data, "detach")
        else np.asarray(mask_data)
    )
    full_masks: List[np.ndarray] = []
    for i in range(masks_np.shape[0]):
        m = masks_np[i]
        if m.shape != (H, W):
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR)
        full_masks.append(m > 0.5)
    return full_masks


def run_yolo_inference(
    model,
    image_rgb: np.ndarray,
    conf_threshold: float = 0.25,
    imgsz: int = 1024,
) -> SegmentationInferenceResult:
    """
    Run YOLOv8 tooth instance segmentation on a full-resolution RGB radiograph.

    Returns a SegmentationInferenceResult whose RawDetections carry FDI tooth
    codes as class_name/class_id (indices into MODEL_CLASSES), masks at the
    original image resolution, and is_restoration=False (this backend does not
    model restorations).
    """
    if image_rgb is None or image_rgb.size == 0:
        raise ValueError("Cannot run inference on an empty image.")
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"Expected 3-channel RGB image, got shape {image_rgb.shape}.")

    H, W = image_rgb.shape[:2]

    start_time = time.time()
    results = model.predict(
        source=image_rgb,
        conf=conf_threshold,
        imgsz=imgsz,
        retina_masks=True,
        verbose=False,
    )
    inference_duration = time.time() - start_time

    result = results[0]
    detections: List[RawDetection] = []

    boxes_arr = (
        result.boxes.xyxy.cpu().numpy()
        if result.boxes is not None and len(result.boxes) > 0
        else np.zeros((0, 4))
    )
    conf_arr = (
        result.boxes.conf.cpu().numpy()
        if result.boxes is not None and len(result.boxes) > 0
        else np.zeros((0,))
    )
    cls_arr = (
        result.boxes.cls.cpu().numpy()
        if result.boxes is not None and len(result.boxes) > 0
        else np.zeros((0,))
    )

    full_masks: List[np.ndarray] = []
    if result.masks is not None and len(cls_arr) > 0:
        full_masks = _masks_to_full_resolution(result.masks.data, (H, W))

    for i in range(len(cls_arr)):
        universal_idx = int(cls_arr[i]) + 1  # class 0 -> Tooth_1
        fdi_code = universal_tooth_to_fdi(universal_idx)
        if fdi_code is None:
            continue

        if i < len(full_masks):
            bin_mask = full_masks[i]
        else:
            bin_mask = np.zeros((H, W), dtype=bool)

        mask_area = int(np.sum(bin_mask))
        if mask_area == 0:
            continue

        x1, y1, x2, y2 = boxes_arr[i]
        x1 = max(0, min(int(round(x1)), W - 1))
        y1 = max(0, min(int(round(y1)), H - 1))
        x2 = max(0, min(int(round(x2)), W))
        y2 = max(0, min(int(round(y2)), H))

        mask_u8 = bin_mask.astype(np.uint8)
        M = cv2.moments(mask_u8)
        if M["m00"] > 0:
            cx = int(round(M["m10"] / M["m00"]))
            cy = int(round(M["m01"] / M["m00"]))
        else:
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

        class_id = fdi_to_model_class_id(fdi_code)

        detections.append(
            RawDetection(
                mask=bin_mask,
                bbox=(x1, y1, x2, y2),
                centroid=(cx, cy),
                area=mask_area,
                score=float(conf_arr[i]),
                class_id=class_id,
                class_name=fdi_code,
                is_restoration=False,
            )
        )

    return SegmentationInferenceResult(
        detections=detections,
        image_shape=(H, W),
        inference_time=inference_duration,
        device_used=resolve_yolo_inference_device(model),
        checkpoint_loaded=_YOLO_PATH or str(resolve_yolo_weights_path()),
    )


def yolo_weights_info() -> dict:
    """Human-readable provenance for UI display."""
    path = resolve_yolo_weights_path()
    return {
        "backend": "yolov8",
        "repo": YOLO_HF_REPO_ID,
        "url": f"https://huggingface.co/{YOLO_HF_REPO_ID}",
        "license": "AGPL-3.0",
        "approx_size_mb": round(YOLO_WEIGHTS_APPROX_BYTES / 1e6, 1),
        "weights_path": str(path),
        "available": yolo_backend_available(),
    }