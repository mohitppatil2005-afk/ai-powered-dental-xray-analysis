import numpy as np
import pytest

from modules.segmentation.fdi_resolver import (
    ALL_PERMANENT_FDI,
    compute_mask_iou,
    resolve_dental_arch,
    ResolvedDentalArch,
)
from modules.segmentation.model import RawDetection


def create_dummy_mask(h, w, box):
    x1, y1, x2, y2 = box
    mask = np.zeros((h, w), dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask


def test_empty_detections():
    result = resolve_dental_arch([], (1000, 2000))
    assert isinstance(result, ResolvedDentalArch)
    assert len(result.teeth) == 0
    assert len(result.restorations) == 0
    assert len(result.missing_teeth) == 32
    assert result.total_detected_teeth == 0


def test_mask_iou():
    m1 = np.zeros((50, 50), dtype=bool)
    m2 = np.zeros((50, 50), dtype=bool)
    m1[10:30, 10:30] = True
    m2[10:30, 10:30] = True

    assert compute_mask_iou(m1, m2) == 1.0

    m3 = np.zeros((50, 50), dtype=bool)
    m3[30:50, 30:50] = True
    assert compute_mask_iou(m1, m3) == 0.0


def test_duplicate_fdi_resolution():
    H, W = 1000, 2000
    # Two detections both claiming FDI 11 (Upper Right)
    # Detection 1: higher confidence
    mask1 = create_dummy_mask(H, W, (800, 300, 880, 420))
    det1 = RawDetection(
        mask=mask1,
        bbox=(800, 300, 880, 420),
        centroid=(840, 360),
        area=int(mask1.sum()),
        score=0.92,
        class_id=1,
        class_name='11',
        is_restoration=False,
    )
    # Detection 2: lower confidence
    mask2 = create_dummy_mask(H, W, (750, 310, 820, 430))
    det2 = RawDetection(
        mask=mask2,
        bbox=(750, 310, 820, 430),
        centroid=(785, 370),
        area=int(mask2.sum()),
        score=0.61,
        class_id=1,
        class_name='11',
        is_restoration=False,
    )

    result = resolve_dental_arch([det1, det2], (H, W))

    assert result.duplicate_conflicts_resolved == 1
    # Exactly one tooth gets FDI 11
    assigned_11 = [t for t in result.teeth if t.fdi_number == '11']
    assert len(assigned_11) == 1
    assert assigned_11[0].confidence == 0.92
    assert assigned_11[0].status == 'CONFIRMED'

    # Conflicting candidate is moved to unassigned with status UNCERTAIN
    assert len(result.unassigned) == 1
    assert result.unassigned[0].fdi_number is None
    assert result.unassigned[0].status == 'UNCERTAIN'
    assert "duplicate assignment conflict" in result.unassigned[0].status_note.lower()


def test_restoration_separation():
    H, W = 1000, 2000
    # Tooth 16
    m_tooth = create_dummy_mask(H, W, (400, 300, 500, 450))
    det_tooth = RawDetection(
        mask=m_tooth,
        bbox=(400, 300, 500, 450),
        centroid=(450, 375),
        area=int(m_tooth.sum()),
        score=0.95,
        class_id=6,
        class_name='16',
        is_restoration=False,
    )
    # Filling on tooth 16
    m_fill = create_dummy_mask(H, W, (420, 320, 480, 380))
    det_fill = RawDetection(
        mask=m_fill,
        bbox=(420, 320, 480, 380),
        centroid=(450, 350),
        area=int(m_fill.sum()),
        score=0.88,
        class_id=34,
        class_name='Filling-Crown',
        is_restoration=True,
    )

    result = resolve_dental_arch([det_tooth, det_fill], (H, W))

    assert len(result.teeth) == 1
    assert result.teeth[0].fdi_number == '16'
    assert len(result.restorations) == 1
    assert result.restorations[0].is_restoration is True
    assert result.restorations[0].restoration_type == 'Filling-Crown'


def test_missing_teeth_audit():
    H, W = 1000, 2000
    # Detect only 2 teeth: 11 and 21
    m1 = create_dummy_mask(H, W, (900, 300, 980, 400))
    m2 = create_dummy_mask(H, W, (1020, 300, 1100, 400))
    det1 = RawDetection(
        mask=m1, bbox=(900, 300, 980, 400), centroid=(940, 350),
        area=int(m1.sum()), score=0.90, class_id=1, class_name='11', is_restoration=False,
    )
    det2 = RawDetection(
        mask=m2, bbox=(1020, 300, 1100, 400), centroid=(1060, 350),
        area=int(m2.sum()), score=0.90, class_id=9, class_name='21', is_restoration=False,
    )

    result = resolve_dental_arch([det1, det2], (H, W))
    assert result.total_detected_teeth == 2
    # 32 - 2 = 30 missing teeth
    assert len(result.missing_teeth) == 30
    assert '11' not in result.missing_teeth
    assert '21' not in result.missing_teeth
    assert '16' in result.missing_teeth
    assert '36' in result.missing_teeth
