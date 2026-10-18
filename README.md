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
3. **Segmentation & FDI Numbering** — Mask R-CNN tooth instance segmentation with anatomical FDI post-processing and duplicate resolution
4. **Pathology Detection** — Honest status reporting for four conditions (all currently unconfigured — no trained weights available)
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

## Segmentation Checkpoint

The Mask R-CNN model requires a trained checkpoint. By default, the app looks for:

```
outputs/results/maskrcnn_torch/best.pth
```

Or you can set the `DENTAL_SEG_WEIGHTS` environment variable:

```bash
export DENTAL_SEG_WEIGHTS=/path/to/your/checkpoint.pth
```

See [`docs/MODEL_SETUP.md`](docs/MODEL_SETUP.md) for details.

## Research Models

Four pathology conditions are defined but **none have trained weights configured**:

| Condition | Status | Notes |
|-----------|--------|-------|
| Dental caries | Not configured | CariSeg research exists but no usable pretrained weights |
| Periodontal bone loss | Not configured | DeNTNet research exists but no usable pretrained weights |
| Periapical lesions | Not configured | No compatible trained model found |
| Impacted teeth | Not configured | No compatible trained model found |

See [`docs/MODEL_SETUP.md`](docs/MODEL_SETUP.md) and [`docs/DATASETS.md`](docs/DATASETS.md) for research details.

## Testing

```bash
source .venv/bin/activate
python3 -m pytest tests/ -v
```

Baseline: 44 passed, 1 skipped (integration test skips when the test radiograph is not present on disk).

## Known Limitations

- No trained segmentation checkpoint is included; the model will not load without one
- All pathology modules report "Not Configured — Unvalidated" (no detections, no false negatives)
- Validation metrics are not independently verified; do not treat as clinically validated
- CPU-only inference; may be slow on high-resolution radiographs

## License

Research and educational use only. Not for clinical diagnosis.