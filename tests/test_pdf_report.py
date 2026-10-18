import numpy as np
import pytest

from utils.pdf_report import generate_dental_pdf_report
from modules.pathology.registry import evaluate_all_pathologies
from modules.segmentation.fdi_resolver import ResolvedDentalArch, ResolvedTooth


def test_pdf_generation_full():
    H, W = 300, 600
    orig = np.random.randint(50, 200, (H, W, 3), dtype=np.uint8)
    enh = np.random.randint(50, 200, (H, W, 3), dtype=np.uint8)
    overlay = np.random.randint(50, 200, (H, W, 3), dtype=np.uint8)

    # Mock tooth
    mask = np.zeros((H, W), dtype=bool)
    mask[100:150, 200:250] = True
    tooth = ResolvedTooth(
        instance_id=1,
        fdi_number='11',
        anatomical_name='Central Incisor',
        quadrant='UR',
        is_restoration=False,
        restoration_type=None,
        confidence=0.95,
        mask=mask,
        bbox=(200, 100, 250, 150),
        centroid=(225, 125),
        area=2500,
        status='CONFIRMED',
        status_note='Verified',
    )
    arch = ResolvedDentalArch(
        teeth=[tooth],
        restorations=[],
        unassigned=[],
        missing_teeth=['18', '28'],
        total_detected_teeth=1,
        total_restorations=0,
        duplicate_conflicts_resolved=0,
        quadrant_counts={'UR': 1, 'UL': 0, 'LL': 0, 'LR': 0},
    )

    pathology = evaluate_all_pathologies(orig)

    pdf_bytes = generate_dental_pdf_report(
        original_rgb=orig,
        enhanced_rgb=enh,
        segmentation_overlay_rgb=overlay,
        dental_arch=arch,
        pathology_results=pathology,
        filename="test_scan.jpg",
        analysis_id="DENT-TEST-001",
    )

    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 1000
    # Standard PDF header magic bytes
    assert pdf_bytes.startswith(b"%PDF-")


def test_pdf_generation_minimal():
    # Test generation when segmentation and pathology are unavailable
    orig = np.zeros((200, 400, 3), dtype=np.uint8)

    pdf_bytes = generate_dental_pdf_report(
        original_rgb=orig,
        enhanced_rgb=None,
        segmentation_overlay_rgb=None,
        dental_arch=None,
        pathology_results=None,
        filename="minimal.jpg",
    )

    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes.startswith(b"%PDF-")
