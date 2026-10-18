"""
modules/segmentation/fdi_resolver.py
====================================
Anatomical FDI Post-Processing Engine for Dental Panoramic X-Rays.

Design principles & assumptions:
1. Does NOT assume a complete 32-tooth arch: accommodates missing teeth,
   extractions, edentulous spaces, and partially imaged jaws.
2. Resolves duplicate FDI assignments by retaining the highest-confidence,
   anatomically consistent proposal and marking conflicting proposals as
   "Unassigned / Uncertain".
3. Separates restoration categories (Bridge, Filling, Implant) from anatomical
   tooth identities.
4. Suppresses redundant multi-proposals on the same physical tooth instance
   via mask intersection-over-union (IoU).
5. Validates quadrant spatial plausibility against dental arch midline priors.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from modules.segmentation.config import (
    CONFIRMED_THRESHOLD,
    FDI_QUADRANT,
    MIN_MODEL_CONFIDENCE,
    MODEL_CLASSES,
    QUADRANT_FULL_NAMES,
    REVIEW_THRESHOLD,
    RESTORATION_CLASSES,
    get_anatomical_name,
)
from modules.segmentation.model import RawDetection


@dataclass
class ResolvedTooth:
    """Individual resolved tooth instance with anatomical validation metadata."""
    instance_id: int
    fdi_number: Optional[str]           # E.g. "11", "26", or None if unassigned
    anatomical_name: str
    quadrant: str                       # 'UR', 'UL', 'LL', 'LR', 'REST', 'UNASSIGNED'
    is_restoration: bool
    restoration_type: Optional[str]     # 'Bridge', 'Filling-Crown', 'Implant'
    confidence: float
    mask: np.ndarray                    # 2D boolean mask matching original image shape
    bbox: Tuple[int, int, int, int]     # (x1, y1, x2, y2)
    centroid: Tuple[int, int]           # (cx, cy)
    area: int
    status: str                         # 'CONFIRMED', 'UNCERTAIN', 'RESTORATION', 'UNASSIGNED'
    status_note: str                    # Human-readable justification of validation status


@dataclass
class ResolvedDentalArch:
    """Complete anatomically resolved dental arch."""
    teeth: List[ResolvedTooth] = field(default_factory=list)
    restorations: List[ResolvedTooth] = field(default_factory=list)
    unassigned: List[ResolvedTooth] = field(default_factory=list)
    missing_teeth: List[str] = field(default_factory=list)
    total_detected_teeth: int = 0
    total_restorations: int = 0
    duplicate_conflicts_resolved: int = 0
    quadrant_counts: Dict[str, int] = field(default_factory=lambda: {'UR': 0, 'UL': 0, 'LL': 0, 'LR': 0})


ALL_PERMANENT_FDI: List[str] = [
    # Upper Right (Q1)
    '18', '17', '16', '15', '14', '13', '12', '11',
    # Upper Left (Q2)
    '21', '22', '23', '24', '25', '26', '27', '28',
    # Lower Left (Q3)
    '31', '32', '33', '34', '35', '36', '37', '38',
    # Lower Right (Q4)
    '48', '47', '46', '45', '44', '43', '42', '41',
]


def compute_mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Calculate Intersection-over-Union between two boolean masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    if intersection == 0:
        return 0.0
    union = np.logical_or(mask1, mask2).sum()
    return float(intersection / union) if union > 0 else 0.0


def resolve_dental_arch(
    raw_detections: List[RawDetection],
    image_shape: Tuple[int, int],
    overlap_iou_threshold: float = 0.40,
    min_confidence: float = 0.30,
) -> ResolvedDentalArch:
    """
    Apply anatomical validation and conflict resolution to raw model detections.

    Parameters:
    - raw_detections: list of RawDetection outputs from model inference.
    - image_shape: (height, width) of the original image.
    - overlap_iou_threshold: IoU threshold for suppressing redundant same-tooth proposals.
    - min_confidence: Minimum confidence threshold.

    Returns:
    - ResolvedDentalArch with resolved teeth, restorations, and missing teeth audit.
    """
    H, W = image_shape
    result = ResolvedDentalArch()

    if not raw_detections:
        result.missing_teeth = list(ALL_PERMANENT_FDI)
        return result

    # 1. Filter by minimum confidence
    valid_dets = [d for d in raw_detections if d.score >= min_confidence]
    if not valid_dets:
        result.missing_teeth = list(ALL_PERMANENT_FDI)
        return result

    # 2. Separate anatomical tooth proposals from restoration proposals
    tooth_candidates: List[RawDetection] = []
    restoration_candidates: List[RawDetection] = []

    for d in valid_dets:
        if d.is_restoration:
            restoration_candidates.append(d)
        else:
            tooth_candidates.append(d)

    # 3. Suppress redundant tooth-tooth overlapping proposals
    # Sort candidates descending by confidence score
    tooth_candidates.sort(key=lambda d: d.score, reverse=True)
    kept_teeth: List[RawDetection] = []
    suppressed_teeth: List[RawDetection] = []

    for cand in tooth_candidates:
        is_redundant = False
        for kept in kept_teeth:
            iou = compute_mask_iou(cand.mask, kept.mask)
            if iou >= overlap_iou_threshold:
                is_redundant = True
                suppressed_teeth.append(cand)
                break
        if not is_redundant:
            kept_teeth.append(cand)

    # 4. Estimate dental arch midlines (vertical midline X and horizontal divider Y)
    all_cx = [d.centroid[0] for d in kept_teeth]
    all_cy = [d.centroid[1] for d in kept_teeth]
    midline_x = float(np.median(all_cx)) if all_cx else W / 2.0
    midline_y = float(np.median(all_cy)) if all_cy else H / 2.0

    # 5. Group by predicted FDI class to resolve duplicates
    fdi_groups: Dict[str, List[RawDetection]] = {}
    for tooth_det in kept_teeth:
        fdi_code = tooth_det.class_name
        if fdi_code not in fdi_groups:
            fdi_groups[fdi_code] = []
        fdi_groups[fdi_code].append(tooth_det)

    resolved_teeth: List[ResolvedTooth] = []
    unassigned_teeth: List[ResolvedTooth] = []
    instance_counter = 1

    for fdi_code, group in fdi_groups.items():
        if len(group) == 1:
            det = group[0]
            cx, cy = det.centroid
            expected_quadrant = FDI_QUADRANT.get(fdi_code, "XX")
            model_confidence = det.score

            # Check quadrant spatial plausibility (anatomical validation)
            anatomically_valid, quad_note = _verify_quadrant_plausibility(
                expected_quadrant, cx, cy, midline_x, midline_y
            )

            # Status depends on BOTH model confidence AND anatomical validity.
            # Assigning an FDI number does NOT automatically make a tooth CONFIRMED.
            if model_confidence >= CONFIRMED_THRESHOLD and anatomically_valid:
                status = "CONFIRMED"
                note = "High model confidence and anatomically consistent placement"
            elif anatomically_valid:
                status = "UNCERTAIN"
                note = (
                    f"Model confidence {model_confidence:.2f} below confirmed "
                    f"threshold ({CONFIRMED_THRESHOLD:.2f}); anatomical placement "
                    "consistent — professional review recommended"
                )
            else:
                status = "UNCERTAIN"
                note = (
                    f"Model confidence {model_confidence:.2f}; {quad_note} "
                    "— professional review recommended"
                )

            # FDI number assigned when anatomically plausible, but status
            # reflects whether confidence is sufficient for confirmation.
            resolved_teeth.append(
                ResolvedTooth(
                    instance_id=instance_counter,
                    fdi_number=fdi_code if anatomically_valid else None,
                    anatomical_name=get_anatomical_name(fdi_code),
                    quadrant=expected_quadrant if anatomically_valid else "UNASSIGNED",
                    is_restoration=False,
                    restoration_type=None,
                    confidence=det.score,
                    mask=det.mask,
                    bbox=det.bbox,
                    centroid=det.centroid,
                    area=det.area,
                    status=status,
                    status_note=note,
                )
            )
            instance_counter += 1

        else:
            # Duplicate conflict: multiple detections for the same FDI number!
            result.duplicate_conflicts_resolved += 1
            # Sort by confidence score descending
            group.sort(key=lambda d: d.score, reverse=True)
            winner = group[0]
            conflicts = group[1:]

            # Assign winner — status depends on BOTH model confidence and anatomy
            cx, cy = winner.centroid
            expected_quad = FDI_QUADRANT.get(fdi_code, "XX")
            winner_confidence = winner.score
            anatomically_valid, quad_note = _verify_quadrant_plausibility(
                expected_quad, cx, cy, midline_x, midline_y
            )

            if winner_confidence >= CONFIRMED_THRESHOLD and anatomically_valid:
                winner_status = "CONFIRMED"
                winner_note = (
                    f"Resolved duplicate conflict ({len(group)} proposals): "
                    f"retained highest confidence ({winner_confidence:.2f}); "
                    "anatomically consistent"
                )
            elif anatomically_valid:
                winner_status = "UNCERTAIN"
                winner_note = (
                    f"Resolved duplicate conflict ({len(group)} proposals): "
                    f"retained highest confidence ({winner_confidence:.2f}); "
                    f"below confirmed threshold ({CONFIRMED_THRESHOLD:.2f}) — "
                    "professional review recommended"
                )
            else:
                winner_status = "UNCERTAIN"
                winner_note = (
                    f"Resolved duplicate conflict ({len(group)} proposals): "
                    f"retained highest confidence ({winner_confidence:.2f}); "
                    f"{quad_note} — professional review recommended"
                )

            resolved_teeth.append(
                ResolvedTooth(
                    instance_id=instance_counter,
                    fdi_number=fdi_code if anatomically_valid else None,
                    anatomical_name=get_anatomical_name(fdi_code),
                    quadrant=expected_quad if anatomically_valid else "UNASSIGNED",
                    is_restoration=False,
                    restoration_type=None,
                    confidence=winner_confidence,
                    mask=winner.mask,
                    bbox=winner.bbox,
                    centroid=winner.centroid,
                    area=winner.area,
                    status=winner_status,
                    status_note=winner_note,
                )
            )
            instance_counter += 1

            # Mark conflicting duplicates as UNASSIGNED / UNCERTAIN
            for conflict in conflicts:
                unassigned_teeth.append(
                    ResolvedTooth(
                        instance_id=instance_counter,
                        fdi_number=None,
                        anatomical_name=f"Unassigned Candidate (claimed {fdi_code})",
                        quadrant="UNASSIGNED",
                        is_restoration=False,
                        restoration_type=None,
                        confidence=conflict.score,
                        mask=conflict.mask,
                        bbox=conflict.bbox,
                        centroid=conflict.centroid,
                        area=conflict.area,
                        status="UNCERTAIN",
                        status_note=(
                            f"Duplicate assignment conflict for tooth #{fdi_code} "
                            f"(score {conflict.score:.2f} < primary {winner.score:.2f}). Number unassigned."
                        ),
                    )
                )
                instance_counter += 1

    # 6. Process Restoration categories
    resolved_restorations: List[ResolvedTooth] = []
    for r_det in restoration_candidates:
        resolved_restorations.append(
            ResolvedTooth(
                instance_id=instance_counter,
                fdi_number=None,
                anatomical_name=f"Restoration: {r_det.class_name}",
                quadrant="REST",
                is_restoration=True,
                restoration_type=r_det.class_name,
                confidence=r_det.score,
                mask=r_det.mask,
                bbox=r_det.bbox,
                centroid=r_det.centroid,
                area=r_det.area,
                status="RESTORATION",
                status_note=f"Prosthetic/restorative material: {r_det.class_name}",
            )
        )
        instance_counter += 1

    # 7. Sort confirmed teeth in clinical FDI order: Q1 (18->11), Q2 (21->28), Q3 (31->38), Q4 (48->41)
    quadrant_buckets: Dict[str, List[ResolvedTooth]] = {'UR': [], 'UL': [], 'LL': [], 'LR': []}
    for t in resolved_teeth:
        if t.quadrant in quadrant_buckets:
            quadrant_buckets[t.quadrant].append(t)
        else:
            unassigned_teeth.append(t)

    # Sort each quadrant by tooth position
    quadrant_buckets['UR'].sort(key=lambda t: int(t.fdi_number or 0), reverse=True) # 18 -> 11
    quadrant_buckets['UL'].sort(key=lambda t: int(t.fdi_number or 0))                # 21 -> 28
    quadrant_buckets['LL'].sort(key=lambda t: int(t.fdi_number or 0))                # 31 -> 38
    quadrant_buckets['LR'].sort(key=lambda t: int(t.fdi_number or 0), reverse=True) # 48 -> 41

    ordered_teeth = (
        quadrant_buckets['UR'] +
        quadrant_buckets['UL'] +
        quadrant_buckets['LL'] +
        quadrant_buckets['LR']
    )

    # 8. Audit missing teeth
    assigned_fdis = {t.fdi_number for t in ordered_teeth if t.fdi_number is not None}
    missing_teeth = [fdi for fdi in ALL_PERMANENT_FDI if fdi not in assigned_fdis]

    result.teeth = ordered_teeth
    result.restorations = resolved_restorations
    result.unassigned = unassigned_teeth
    result.missing_teeth = missing_teeth
    result.total_detected_teeth = len(ordered_teeth)
    result.total_restorations = len(resolved_restorations)
    result.quadrant_counts = {q: len(quadrant_buckets[q]) for q in ['UR', 'UL', 'LL', 'LR']}

    return result


def _verify_quadrant_plausibility(
    expected_quadrant: str,
    cx: int,
    cy: int,
    midline_x: float,
    midline_y: float,
    tolerance_x_pct: float = 0.15,
    tolerance_y_pct: float = 0.20,
) -> Tuple[bool, str]:
    """
    Verify whether the instance centroid physically aligns with the expected quadrant.

    In panoramic X-ray convention:
    - UR (Q1): X < midline_x, Y < midline_y
    - UL (Q2): X > midline_x, Y < midline_y
    - LL (Q3): X > midline_x, Y > midline_y
    - LR (Q4): X < midline_x, Y > midline_y
    """
    margin_x = midline_x * tolerance_x_pct
    margin_y = midline_y * tolerance_y_pct

    if expected_quadrant == 'UR':
        if cx > midline_x + margin_x or cy > midline_y + margin_y:
            return False, f"Centroid ({cx}, {cy}) contradicts Upper Right quadrant boundaries"
    elif expected_quadrant == 'UL':
        if cx < midline_x - margin_x or cy > midline_y + margin_y:
            return False, f"Centroid ({cx}, {cy}) contradicts Upper Left quadrant boundaries"
    elif expected_quadrant == 'LL':
        if cx < midline_x - margin_x or cy < midline_y - margin_y:
            return False, f"Centroid ({cx}, {cy}) contradicts Lower Left quadrant boundaries"
    elif expected_quadrant == 'LR':
        if cx > midline_x + margin_x or cy < midline_y - margin_y:
            return False, f"Centroid ({cx}, {cy}) contradicts Lower Right quadrant boundaries"

    return True, "Spatial position consistent with quadrant"
