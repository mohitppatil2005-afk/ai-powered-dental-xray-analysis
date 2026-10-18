# Model Setup

## Segmentation Checkpoint (Mask R-CNN)

The application uses a 36-class Mask R-CNN ResNet-50+FPN model for tooth instance segmentation. The model expects:

- **Architecture**: Mask R-CNN with ResNet-50 FPN backbone
- **Classes**: 36 (background + 32 FDI teeth + 3 restoration types)
- **Input**: RGB panoramic radiograph
- **Output**: Instance masks, bounding boxes, class labels, confidence scores

### Class Mapping

| Class ID | Class Name | Type |
|----------|-----------|------|
| 0 | `__background__` | Background |
| 1-8 | `11`–`18` | FDI teeth (Upper Right) |
| 9-16 | `21`–`28` | FDI teeth (Upper Left) |
| 17-24 | `31`–`38` | FDI teeth (Lower Left) |
| 25-32 | `41`–`48` | FDI teeth (Lower Right) |
| 33 | `Bridge` | Restoration |
| 34 | `Filling-Crown` | Restoration |
| 35 | `Implant` | Restoration |

### Checkpoint Location

Default: `outputs/results/maskrcnn_torch/best.pth` (relative to project root)

Override via environment variable:
```bash
export DENTAL_SEG_WEIGHTS=/path/to/checkpoint.pth
```

### Checkpoint Status

**No checkpoint is included in the repository.** The default path does not exist on a fresh clone. The application handles this gracefully:

- `load_segmentation_model()` raises `FileNotFoundError` with a clear message
- The Streamlit UI shows "Tooth segmentation failed" and continues with segmentation marked as unavailable
- All other features (validation, enhancement, pathology status, reporting) work without the checkpoint

### Obtaining Weights

You must obtain or train a compatible 36-class Mask R-CNN checkpoint yourself. The project does not provide or host model weights.

## Confidence Thresholds

Configured in `modules/segmentation/config.py`:

| Threshold | Value | Meaning |
|-----------|-------|---------|
| `MIN_MODEL_CONFIDENCE` | 0.30 | Detections below this are filtered out |
| `REVIEW_THRESHOLD` | 0.50 | Confidence below this → UNCERTAIN status |
| `CONFIRMED_THRESHOLD` | 0.70 | Confidence ≥ this + anatomy valid → CONFIRMED status |

These are engineering thresholds for a research prototype. They are **not clinically validated**.

## Pathology Models

All four pathology conditions are currently **not configured**:

| Condition | Status | Reason |
|-----------|--------|--------|
| Dental caries | Not configured | No trained weights available (CariSeg research exists but no usable checkpoint) |
| Periodontal bone loss | Not configured | No trained weights available (DeNTNet research exists but no usable checkpoint) |
| Periapical lesions | Not configured | No compatible trained model found |
| Impacted teeth | Not configured | No compatible trained model found |

To add a pathology model, implement a detector function matching `Callable[[np.ndarray], ConditionEvaluationResult]` and register it via `modules.pathology.registry.register_pathology_detector()`.

## CPU Execution

The application runs on CPU by default. CUDA is used automatically if available and functional. No GPU is required.