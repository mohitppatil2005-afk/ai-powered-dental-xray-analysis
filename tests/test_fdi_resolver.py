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


def test_low_confidence_tooth_is_uncertain_not_confirmed():
    """Regression: low-confidence assignments (e.g. 40.2%) must be UNCERTAIN,
    not CONFIRMED. Assigning an FDI number does not imply confirmation."""
    H, W = 1000, 2000
    mask = create_dummy_mask(H, W, (1200, 600, 1280, 720))
    det = RawDetection(
        mask=mask,
        bbox=(1200, 600, 1280, 720),
        centroid=(1240, 660),
        area=int(mask.sum()),
        score=0.402,  # reported bug: 40.2% was marked CONFIRMED
        class_id=20,
        class_name='34',
        is_restoration=False,
    )
    result = resolve_dental_arch([det], (H, W))

    assert len(result.teeth) == 1
    tooth = result.teeth[0]
    assert tooth.fdi_number == '34'
    assert tooth.confidence == 0.402
    assert tooth.status == 'UNCERTAIN', f"Expected UNCERTAIN, got {tooth.status}"
    assert tooth.status != 'CONFIRMED'
    assert 'below confirmed' in tooth.status_note.lower()


def test_reported_low_confidence_examples():
    """Regression: all three reported low-confidence cases (34: 40.2%, 31: 48.8%,
    41: 59.8%) must be UNCERTAIN, never CONFIRMED."""
    H, W = 1000, 2000
    cases = [
        ('34', 0.402, (1200, 600, 1280, 720), (1240, 660), 20),   # LL
        ('31', 0.488, (1200, 300, 1280, 420), (1240, 360), 17),   # LL
        ('41', 0.598, (800, 600, 880, 720), (840, 660), 25),      # LR
    ]
    for fdi, score, box, centroid, class_id in cases:
        mask = create_dummy_mask(H, W, box)
        det = RawDetection(
            mask=mask,
            bbox=box,
            centroid=centroid,
            area=int(mask.sum()),
            score=score,
            class_id=class_id,
            class_name=fdi,
            is_restoration=False,
        )
        result = resolve_dental_arch([det], (H, W))
        assert len(result.teeth) == 1, f"Expected 1 tooth for {fdi} at {score}"
        tooth = result.teeth[0]
        assert tooth.fdi_number == fdi, f"{fdi}: expected FDI assigned"
        assert tooth.confidence == score, f"{fdi}: confidence mismatch"
        assert tooth.status == 'UNCERTAIN', (
            f"{fdi} at {score:.1%} should be UNCERTAIN, got {tooth.status}"
        )
        assert tooth.status != 'CONFIRMED', f"{fdi} at {score:.1%} must not be CONFIRMED"


def test_confidence_threshold_boundary_confirmed():
    """Regression: confidence at or above CONFIRMED_THRESHOLD with valid anatomy
    yields CONFIRMED status."""
    H, W = 1000, 2000
    mask = create_dummy_mask(H, W, (1200, 600, 1280, 720))
    det = RawDetection(
        mask=mask,
        bbox=(1200, 600, 1280, 720),
        centroid=(1240, 660),
        area=int(mask.sum()),
        score=0.70,  # exactly at CONFIRMED_THRESHOLD
        class_id=20,
        class_name='34',
        is_restoration=False,
    )
    result = resolve_dental_arch([det], (H, W))
    tooth = result.teeth[0]
    assert tooth.status == 'CONFIRMED', (
        f"Expected CONFIRMED at 0.70 confidence, got {tooth.status}"
    )


def test_confidence_threshold_boundary_uncertain():
    """Regression: confidence just below CONFIRMED_THRESHOLD yields UNCERTAIN,
    not CONFIRMED."""
    H, W = 1000, 2000
    mask = create_dummy_mask(H, W, (1200, 600, 1280, 720))
    det = RawDetection(
        mask=mask,
        bbox=(1200, 600, 1280, 720),
        centroid=(1240, 660),
        area=int(mask.sum()),
        score=0.69,  # just below CONFIRMED_THRESHOLD
        class_id=20,
        class_name='34',
        is_restoration=False,
    )
    result = resolve_dental_arch([det], (H, W))
    tooth = result.teeth[0]
    assert tooth.status == 'UNCERTAIN', (
        f"Expected UNCERTAIN at 0.69 confidence, got {tooth.status}"
    )


def test_confidence_review_threshold_boundary():
    """Regression: confidence at REVIEW_THRESHOLD (0.50) yields UNCERTAIN."""
    H, W = 1000, 2000
    mask = create_dummy_mask(H, W, (1200, 600, 1280, 720))
    det = RawDetection(
        mask=mask,
        bbox=(1200, 600, 1280, 720),
        centroid=(1240, 660),
        area=int(mask.sum()),
        score=0.50,  # exactly at REVIEW_THRESHOLD
        class_id=20,
        class_name='34',
        is_restoration=False,
    )
    result = resolve_dental_arch([det], (H, W))
    tooth = result.teeth[0]
    assert tooth.status == 'UNCERTAIN', (
        f"Expected UNCERTAIN at 0.50 confidence, got {tooth.status}"
    )


def test_quadrant_mismatch_high_confidence_is_uncertain():
    """Regression: high model confidence but anatomically implausible quadrant
    yields UNCERTAIN, not CONFIRMED. Model confidence is distinct from
    anatomical/resolver confidence."""
    H, W = 1000, 2000
    # Establish midline on the left side with two UR teeth
    m1 = create_dummy_mask(H, W, (400, 200, 480, 320))   # tooth 11, UR-plausible (left side)
    m2 = create_dummy_mask(H, W, (500, 200, 580, 320))   # tooth 12, UR-plausible (left side)
    det1 = RawDetection(
        mask=m1, bbox=(400, 200, 480, 320), centroid=(440, 260),
        area=int(m1.sum()), score=0.90, class_id=1, class_name='11', is_restoration=False,
    )
    det2 = RawDetection(
        mask=m2, bbox=(500, 200, 580, 320), centroid=(540, 260),
        area=int(m2.sum()), score=0.90, class_id=2, class_name='12', is_restoration=False,
    )
    # Tooth 34 (LL) placed far left (wrong quadrant: LL needs X > midline_x)
    m3 = create_dummy_mask(H, W, (100, 700, 180, 820))   # far left, lower
    det3 = RawDetection(
        mask=m3, bbox=(100, 700, 180, 820), centroid=(140, 760),
        area=int(m3.sum()), score=0.95, class_id=20, class_name='34', is_restoration=False,
    )

    result = resolve_dental_arch([det1, det2, det3], (H, W))
    # Quadrant-mismatched teeth land in unassigned with fdi_number=None
    mismatch_teeth = [t for t in result.unassigned if t.confidence == 0.95]
    assert len(mismatch_teeth) == 1, f"Expected 1 unassigned tooth with confidence 0.95, got {len(mismatch_teeth)}"
    tooth_34 = mismatch_teeth[0]
    # High confidence (0.95) but quadrant mismatch → UNCERTAIN, not CONFIRMED
    assert tooth_34.status == 'UNCERTAIN', (
        f"Expected UNCERTAIN for quadrant mismatch, got {tooth_34.status}"
    )
    assert tooth_34.confidence == 0.95  # model confidence preserved separately
    assert tooth_34.fdi_number is None  # FDI not assigned due to quadrant mismatch
    assert tooth_34.quadrant == 'UNASSIGNED'


def test_duplicate_low_confidence_winner_is_uncertain():
    """Regression: duplicate FDI proposals where the winner has low confidence
    must be UNCERTAIN, not CONFIRMED."""
    H, W = 1000, 2000
    # Two detections both claiming FDI 11 (UR), non-overlapping masks (IoU=0)
    # so both survive the IoU suppression step and form a duplicate group
    mask1 = create_dummy_mask(H, W, (200, 200, 280, 320))
    det1 = RawDetection(
        mask=mask1, bbox=(200, 200, 280, 320), centroid=(240, 260),
        area=int(mask1.sum()), score=0.45, class_id=1, class_name='11', is_restoration=False,
    )
    mask2 = create_dummy_mask(H, W, (500, 200, 580, 320))
    det2 = RawDetection(
        mask=mask2, bbox=(500, 200, 580, 320), centroid=(540, 260),
        area=int(mask2.sum()), score=0.40, class_id=1, class_name='11', is_restoration=False,
    )

    result = resolve_dental_arch([det1, det2], (H, W))
    assert result.duplicate_conflicts_resolved == 1
    assigned_11 = [t for t in result.teeth if t.fdi_number == '11']
    assert len(assigned_11) == 1
    assert assigned_11[0].status == 'UNCERTAIN', (
        f"Low-confidence duplicate winner should be UNCERTAIN, got {assigned_11[0].status}"
    )
    assert assigned_11[0].confidence == 0.45
    # Conflicting candidate is unassigned
    assert len(result.unassigned) == 1
    assert result.unassigned[0].status == 'UNCERTAIN'
    assert result.unassigned[0].fdi_number is None
