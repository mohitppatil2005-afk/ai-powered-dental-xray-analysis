# AI-Powered Dental Panoramic X-Ray Analysis

A research and educational prototype for automated analysis of dental panoramic radiographs (OPG). Built with Streamlit, PyTorch, and OpenCV.

> ⚠️ **Educational and Research Use Only**: This application does not provide a definitive medical or dental diagnosis. All outputs, contours, tooth identifications, and condition statuses require independent review by a qualified dental professional.

## Project Structure

```
ai-powered-dental-xray-analysis/
├── app.py                          # Streamlit application entry point
├── requirements.txt                # Python dependencies
├── pytest.ini                      # Pytest configuration
├── README.md                       # This file
├── modules/                        # Core analysis modules
│   ├── __init__.py
│   ├── analysis_status.py          # Module status reporting
│   ├── segmentation/
│   │   ├── __init__.py
│   │   ├── config.py               # Class mappings, confidence thresholds, checkpoint path
│   │   ├── model.py                # Mask R-CNN inference wrapper
│   │   ├── fdi_resolver.py         # Anatomical FDI post-processing
│   │   └── visualization.py        # Segmentation overlays
│   └── pathology/
│       ├── __init__.py
│       ├── types.py                # Pathology data models
│       ├── registry.py             # Pathology condition registry
│       └── ...
├── preprocessing/                  # Image validation & enhancement
│   ├── validation.py               # Image loading and validation
│   └── enhancement.py              # CLAHE contrast enhancement
├── utils/                          # Reporting utilities
│   ├── report.py                   # TXT/JSON report generation
│   └── pdf_report.py               # PDF report generation (ReportLab)
├── tests/                          # pytest test suite
│   ├── test_validation.py
│   ├── test_enhancement.py
│   ├── test_segmentation_wrapper.py
│   ├── test_fdi_resolver.py
│   ├── test_pathology.py
│   ├── test_pdf_report.py
│   └── test_integration.py
├── docs/
│   ├── DATASETS.md                 # Dataset sources and access restrictions
│   └── MODEL_SETUP.md              # Model checkpoint configuration
└── .venv/                          # Virtual environment (not committed)
```

## Architecture

The application follows a 5-section workflow:

1. **Upload & Validation** — Load and validate panoramic radiographs (PNG, JPG, JPEG, TIF, TIFF)
2. **Enhancement** — CLAHE contrast enhancement with configurable parameters
3. **Segmentation & FDI Numbering** — Backend-selectable tooth instance segmentation (public pretrained YOLOv8 32-tooth model by default; local Mask R-CNN checkpoint optional) with anatomical FDI post-processing and duplicate resolution
4. **Pathology Detection** — Screening flags from a public pretrained YOLO11 detector (caries, periapical lesions, impacted teeth); periodontal bone loss is explicitly unavailable (no public weights exist)
5. **Reporting** — PDF, TXT, and JSON report generation with model provenance

## Setup

### Prerequisites

- Python 3.12+
- 8 GB RAM recommended (CPU inference)
- Linux (tested on Linux Lite)

### Installation

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Application

```bash
source .venv/bin/activate
streamlit run app.py
```

## Segmentation Backends

Two segmentation backends are supported (selected in the UI or via `DENTAL_SEG_BACKEND`):

| Backend | Model | Weights | Restorations |
|---------|-------|---------|--------------|
| `yolov8` (default when weights present) | YOLOv8n-seg, 32-tooth instance segmentation ([promitbhattacharjee/dental-panoramic-xray-segmentation](https://huggingface.co/promitbhattacharjee/dental-panoramic-xray-segmentation), AGPL-3.0, ~7 MB) | One-click download from the app UI (or set `DENTAL_SEG_YOLO_WEIGHTS`) | ❌ teeth only |
| `maskrcnn` | 36-class Mask R-CNN ResNet-50+FPN (local checkpoint) | Train yourself — no public weights | ✅ bridge/crown/implant |

Environment variables:

```bash
export DENTAL_SEG_BACKEND=yolov8          # auto (default) | yolov8 | maskrcnn | none
export DENTAL_SEG_YOLO_WEIGHTS=/path/to/best.pt   # optional YOLO weights override
export DENTAL_SEG_WEIGHTS=/path/to/checkpoint.pth # optional Mask R-CNN checkpoint override
```

See [`docs/MODEL_SETUP.md`](docs/MODEL_SETUP.md) for details.

## Pathology Models

One public pretrained pathology detector is integrated; one condition has **no public weights and stays explicitly unavailable**:

| Condition | Status | Notes |
|-----------|--------|-------|
| Dental caries | Screening detector (YOLO11n, [liodon-ai/dental-panoramic-detector](https://huggingface.co/liodon-ai/dental-panoramic-detector), CC-BY-NC-4.0, ~5.5 MB) | Upstream-reported metrics only; outputs are screening flags, not diagnoses |
| Periodontal bone loss | **Unavailable** | No public panoramic bone-loss detector exists — never fabricated |
| Periapical lesions | Screening detector (same YOLO11n model) | Differential-diagnosis validation still required |
| Impacted teeth | Screening detector (same YOLO11n model) | Angulation/depth validation still required |

All pathology outputs are explicitly **not clinically validated**; negative findings never imply absence of disease.

See [`docs/MODEL_SETUP.md`](docs/MODEL_SETUP.md) and [`docs/DATASETS.md`](docs/DATASETS.md) for research details.

## Testing

```bash
source .venv/bin/activate
python3 -m pytest tests/ -v
```

Baseline: 67 passed, 1 skipped (integration test skips when the test radiograph is not present on disk).

## Known Limitations

- The Mask R-CNN restoration-detection backend requires a self-trained checkpoint (not included); the default YOLOv8 backend segments teeth only and does NOT detect bridges/crowns/implants
- Periodontal bone loss detection is unavailable — no public panoramic weights exist, and the app reports this honestly rather than fabricating results
- All pathology outputs are screening flags from unvalidated public models; a NEGATIVE finding never implies absence of disease
- Validation metrics are upstream-reported only and not independently verified; do not treat as clinically validated
- YOLO11 pathology weights are CC-BY-NC-4.0 (non-commercial); YOLOv8 segmentation weights are AGPL-3.0

## License

Research and educational use only. Not for clinical diagnosis.