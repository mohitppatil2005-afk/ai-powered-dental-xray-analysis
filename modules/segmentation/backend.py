"""
modules/segmentation/backend.py
===============================
Unified segmentation backend selection.

Two backends are supported:
- "maskrcnn": the project's original 36-class Torchvision Mask R-CNN
  (teeth + restorations) — requires a user-trained checkpoint.
- "yolov8": public pretrained 32-tooth instance segmentation
  (teeth only, no restorations) with downloadable weights.

Selection honors the DENTAL_SEG_BACKEND env var (re-read per call):
- "auto" (default): prefer yolov8 if its weights are present; otherwise
  fall back to maskrcnn if its checkpoint is present; otherwise None.
- "maskrcnn" / "yolov8": force that backend (still requires weights).
- "none": disable segmentation entirely.

Both backends return the same SegmentationInferenceResult format so the
downstream FDI resolver and visualization are shared unchanged.
"""

import os
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from modules.segmentation.config import VALID_SEG_BACKENDS
from modules.segmentation.model import (
    SegmentationInferenceResult,
    load_segmentation_model,
    run_segmentation_inference,
)
from modules.segmentation.yolo_adapter import (
    load_yolo_model,
    resolve_yolo_weights_path,
    run_yolo_inference,
    yolo_backend_available,
)


@dataclass
class SegmentationBackendInfo:
    """Resolved backend identity and availability."""
    name: Optional[str]              # "maskrcnn", "yolov8", or None
    weights_path: Optional[str]
    supports_restorations: bool
    source: str                      # human-readable provenance string


def _maskrcnn_weights_available() -> bool:
    """True if the local Mask R-CNN checkpoint exists."""
    from modules.segmentation.model import DEFAULT_CHECKPOINT_PATH
    import os as _os
    path = _os.getenv("DENTAL_SEG_WEIGHTS", str(DEFAULT_CHECKPOINT_PATH))
    from pathlib import Path
    return Path(path).exists()


def resolve_segmentation_backend(
    requested: Optional[str] = None,
) -> SegmentationBackendInfo:
    """
    Determine which segmentation backend to use.

    `requested` overrides the DENTAL_SEG_BACKEND env var if provided.
    """
    choice = (requested or os.getenv("DENTAL_SEG_BACKEND", "auto")).strip().lower()
    if choice not in VALID_SEG_BACKENDS:
        choice = "auto"

    yolo_path = str(resolve_yolo_weights_path())

    if choice == "none":
        return SegmentationBackendInfo(None, None, False, "Segmentation disabled.")

    if choice == "yolov8":
        return SegmentationBackendInfo(
            name="yolov8" if yolo_backend_available() else None,
            weights_path=yolo_path,
            supports_restorations=False,
            source="YOLOv8n-seg 32-tooth (promitbhattacharjee, AGPL-3.0)",
        )

    if choice == "maskrcnn":
        from modules.segmentation.model import DEFAULT_CHECKPOINT_PATH
        import os as _os
        path = _os.getenv("DENTAL_SEG_WEIGHTS", str(DEFAULT_CHECKPOINT_PATH))
        return SegmentationBackendInfo(
            name="maskrcnn" if _maskrcnn_weights_available() else None,
            weights_path=path,
            supports_restorations=True,
            source="Mask R-CNN ResNet-50+FPN (local checkpoint)",
        )

    # "auto": prefer yolov8, then maskrcnn.
    if yolo_backend_available():
        return SegmentationBackendInfo(
            name="yolov8",
            weights_path=yolo_path,
            supports_restorations=False,
            source="YOLOv8n-seg 32-tooth (promitbhattacharjee, AGPL-3.0)",
        )

    if _maskrcnn_weights_available():
        from modules.segmentation.model import DEFAULT_CHECKPOINT_PATH
        import os as _os
        path = _os.getenv("DENTAL_SEG_WEIGHTS", str(DEFAULT_CHECKPOINT_PATH))
        return SegmentationBackendInfo(
            name="maskrcnn",
            weights_path=path,
            supports_restorations=True,
            source="Mask R-CNN ResNet-50+FPN (local checkpoint)",
        )

    return SegmentationBackendInfo(None, None, False, "No segmentation backend available.")


def run_segmentation(
    image_rgb: np.ndarray,
    backend: Optional[str] = None,
    conf_threshold: float = 0.25,
) -> "tuple[Optional[SegmentationInferenceResult], Optional[str]]":
    """
    Run tooth segmentation using the resolved backend.

    Returns (result, backend_name). If no backend is available, returns (None, None).
    Raises RuntimeError if the chosen backend fails to load or run.
    """
    info = resolve_segmentation_backend(backend)
    if info.name is None:
        return None, None

    if info.name == "yolov8":
        model = load_yolo_model()
        result = run_yolo_inference(model, image_rgb, conf_threshold=conf_threshold)
    else:
        model, device = load_segmentation_model()
        result = run_segmentation_inference(
            model=model,
            image_rgb=image_rgb,
            device=device,
            conf_threshold=conf_threshold,
        )

    return result, info.name