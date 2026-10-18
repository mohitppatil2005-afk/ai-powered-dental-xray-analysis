# Model Setup

## Overview

Two model families are used:

| Component | Model | Weights source | License | Status |
|-----------|-------|----------------|---------|--------|
| Tooth segmentation (default) | YOLOv8n-seg, 32-tooth | Hugging Face `promitbhattacharjee/dental-panoramic-xray-segmentation` | AGPL-3.0 | One-click download in the app UI |
| Tooth segmentation + restorations (optional) | 36-class Mask R-CNN ResNet-50+FPN | Self-trained local checkpoint | — | Not included; train yourself |
| Pathology screening | YOLO11n detector | Hugging Face `liodon-ai/dental-panoramic-detector` | CC-BY-NC-4.0 | One-click download in the app UI |
| Periodontal bone loss | — | — | — | **Unavailable** — no public weights exist |

Backend selection honors `DENTAL_SEG_BACKEND` (`auto` default → YOLOv8 if weights present, else Mask R-CNN) and the UI radio button.

## Segmentation Backends

### YOLOv8 32-tooth backend (default)

- **Architecture**: YOLOv8-nano instance segmentation (32 classes, `Tooth_1`..`Tooth_32` universal numbering)
- **Adapter**: `modules/segmentation/yolo_adapter.py` converts outputs to the pipeline's `RawDetection` format with FDI labels; the anatomical FDI resolver, duplicate resolution, and visualization are shared unchanged
- **Weights location**: `models/weights/yolov8n_teeth32/best.pt` (~7 MB), override via `DENTAL_SEG_YOLO_WEIGHTS`
- **Download**: the app UI button, or `python3 -c "from modules.segmentation.yolo_adapter import download_yolo_weights; download_yolo_weights()"`
- **Verified behavior**: all 32 teeth mapped anatomically correct via universal→FDI conversion (quadrant boundary mapping covered by unit tests)
- **Limitation**: detects teeth ONLY — restorations (bridge/filling-crown/implant) are not in this model's class set
- **Upstream-reported (unverified here)**: mask mAP50 94.4%, precision 89.8%, recall 91.0%

### Mask R-CNN 36-class backend (optional)

The application's original backend adds 3 restoration classes on top of 32 teeth:

| Class ID | Class Name | Type |
|----------|-----------|------|
| 0 | `__background__` | Background |
| 1-32 | FDI tooth codes | Teeth |
| 33 | `Bridge` | Restoration |
| 34 | `Filling-Crown` | Restoration |
| 35 | `Implant` | Restoration |

Default checkpoint location: `outputs/results/maskrcnn_torch/best.pth` (override via `DENTAL_SEG_WEIGHTS`).

**No checkpoint is included in the repository.** You must train a compatible 36-class Mask R-CNN checkpoint yourself. The app handles its absence gracefully — select the `yolov8` backend instead.

## Confidence Thresholds

Configured in `modules/segmentation/config.py`:

| Threshold | Value | Meaning |
|-----------|-------|---------|
| `MIN_MODEL_CONFIDENCE` | 0.30 | Detections below this are filtered out |
| `REVIEW_THRESHOLD` | 0.50 | Confidence below this → UNCERTAIN status |
| `CONFIRMED_THRESHOLD` | 0.70 | Confidence ≥ this + anatomy valid → CONFIRMED status |

These are engineering thresholds for a research prototype. They are **not clinically validated**.

## Pathology Models

| Condition | Status | Reason |
|-----------|--------|--------|
| Dental caries | Screening detector active | YOLO11n public weights; upstream-reported metrics only |
| Periodontal bone loss | **Unavailable** | No public panoramic bone-loss detector exists; the registry reports NOT_CONFIGURED and never fabricates findings |
| Periapical lesions | Screening detector active | YOLO11n public weights; requires differential-diagnosis validation vs. mental foramen/incisive canal/sinus |
| Impacted teeth | Screening detector active | YOLO11n public weights; angulation/depth validation still required |

- **Weights location**: `models/weights/yolo11n_pathology/best.pt` (~5.5 MB), override via `DENTAL_PATHO_YOLO_WEIGHTS`
- **Download**: the app UI button, or `python3 -c "from modules.pathology.yolo_detector import download_pathology_weights; download_pathology_weights()"`
- **Integration**: `register_yolo_pathology_detectors()` registers detectors into the pluggable registry; any evaluation error degrades to an honest UNVALIDATED status, never to a fabricated NEGATIVE
- **To add another condition**: implement a detector matching `Callable[[np.ndarray], ConditionEvaluationResult]` and register it via `modules.pathology.registry.register_pathology_detector()`

## CPU Execution

The application runs on CPU by default (~1 s per panoramic image with the nano models). CUDA is used automatically if available and functional. No GPU is required.