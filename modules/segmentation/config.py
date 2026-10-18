"""
modules/segmentation/config.py
==============================
Configuration, class mappings, and anatomical metadata for the 36-class
dental panoramic tooth instance segmentation model.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple

# Default path to the validated 36-class Mask R-CNN checkpoint
DEFAULT_CHECKPOINT_PATH = Path(
    os.getenv(
        "DENTAL_SEG_WEIGHTS",
        r"C:\Users\mohit\Downloads\dental-segmentation\outputs\results\maskrcnn_torch\best.pth"
    )
)

NUM_CLASSES = 36  # Background (0) + 32 FDI teeth (1-32) + 3 restorations (33-35)

MODEL_CLASSES: List[str] = [
    '__background__',
    '11', '12', '13', '14', '15', '16', '17', '18',
    '21', '22', '23', '24', '25', '26', '27', '28',
    '31', '32', '33', '34', '35', '36', '37', '38',
    '41', '42', '43', '44', '45', '46', '47', '48',
    'Bridge', 'Filling-Crown', 'Implant',
]

RESTORATION_CLASSES = {'Bridge', 'Filling-Crown', 'Implant'}

# FDI Quadrant mapping (Standard dental notation)
# Note: In panoramic X-ray viewing convention:
# - Patient Right (Q1 & Q4) is on the LEFT side of the radiograph
# - Patient Left (Q2 & Q3) is on the RIGHT side of the radiograph
FDI_QUADRANT: Dict[str, str] = {
    **{str(fdi): 'UR' for fdi in range(11, 19)},  # Upper Right (Maxillary Right)
    **{str(fdi): 'UL' for fdi in range(21, 29)},  # Upper Left  (Maxillary Left)
    **{str(fdi): 'LL' for fdi in range(31, 39)},  # Lower Left  (Mandibular Left)
    **{str(fdi): 'LR' for fdi in range(41, 49)},  # Lower Right (Mandibular Right)
}

QUADRANT_FULL_NAMES: Dict[str, str] = {
    'UR': 'Maxillary Right (Upper Right)',
    'UL': 'Maxillary Left (Upper Left)',
    'LL': 'Mandibular Left (Lower Left)',
    'LR': 'Mandibular Right (Lower Right)',
    'REST': 'Restoration / Prosthetic',
    'UNASSIGNED': 'Unassigned / Uncertain',
}

TOOTH_NAMES: Dict[str, str] = {
    '1': 'Central Incisor',
    '2': 'Lateral Incisor',
    '3': 'Canine (Cuspid)',
    '4': '1st Premolar (Bicuspid)',
    '5': '2nd Premolar (Bicuspid)',
    '6': '1st Molar (6-year molar)',
    '7': '2nd Molar (12-year molar)',
    '8': '3rd Molar (Wisdom tooth)',
}


def get_anatomical_name(fdi_code: str) -> str:
    """Return full descriptive anatomical name for an FDI tooth code or restoration."""
    if fdi_code in RESTORATION_CLASSES:
        return f"Restoration: {fdi_code}"
    if len(fdi_code) == 2 and fdi_code.isdigit():
        quadrant = FDI_QUADRANT.get(fdi_code, "")
        quad_name = QUADRANT_FULL_NAMES.get(quadrant, "")
        tooth_pos = fdi_code[1]
        tooth_name = TOOTH_NAMES.get(tooth_pos, "Tooth")
        return f"{tooth_name} ({quad_name})"
    return f"Tooth {fdi_code}"


# Quadrant Palette (distinct readable colors in RGB)
QUADRANT_COLORS: Dict[str, Tuple[int, int, int]] = {
    'UR': (52, 152, 219),    # Blue
    'UL': (230, 126, 34),    # Orange
    'LL': (46, 204, 113),    # Green
    'LR': (155, 89, 182),    # Purple
    'REST': (241, 196, 15),  # Yellow
    'XX': (149, 165, 166),   # Gray (Uncertain)
}
