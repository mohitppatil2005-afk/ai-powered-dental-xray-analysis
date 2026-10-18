# Dataset Sources

This document lists dataset sources referenced in the project plan and their current status.

## Dental Caries / Tooth Segmentation — CariSeg

**Paper**: "Teeth segmentation and carious lesions segmentation in panoramic X-ray images using CariSeg, a networks' ensemble"
- URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC11128823/
- DOI: 10.1016/j.heliyon.2024.e30836
- License: CC BY-NC-ND (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0)
- Published: Heliyon, 2024

**Reported datasets**:
- Tooth identification: Tufts Dental Database (1,000 panoramic radiographs) + 116 anonymized panoramic X-rays from Noor Medical Imaging Center, Qom, Iran
- Caries segmentation: ~150 images (per project plan)

**Code repository**: https://github.com/am2731/cavities-segmentation
- Contains only a Jupyter notebook (`teeth_cavities_notebook.ipynb`)
- No proper inference code, no trained weights available for download
- 0 stars, 0 forks — appears to be a minimal/research-only release

**Verdict**: Not suitable for integration without significant additional work. No pretrained weights, no production inference code, restrictive CC BY-NC-ND license.

**Storage**: N/A (weights not available)

---

## Periodontal Bone Loss — DeNTNet

**Paper**: "DeNTNet: Deep Neural Transfer Network for the detection of periodontal bone loss using panoramic dental radiographs"
- URL: https://www.nature.com/articles/s41598-019-53758-2
- DOI: 10.1038/s41598-019-53758-2
- License: CC BY 4.0 (Creative Commons Attribution 4.0)
- Published: Scientific Reports, 2019

**Dataset**: 12,179 panoramic dental radiographs annotated by experienced dental clinicians
- Training: 11,189 images
- Validation: 190 images
- Test: 800 images

**Results**: F1 score 0.75 on test set (vs. 0.69 average for dental clinicians)

**Code/weights**: No public code repository or pretrained weights found in the paper.

**Verdict**: Not suitable for integration without trained weights. No checkpoint available for download.

**Storage**: N/A (weights not available)

---

## Periapical Lesion Detection

**Dataset candidate**: Mendeley Data
- URL: https://data.mendeley.com/datasets/kx52tk2ddj/3
- DOI: 10.17632/kx52tk2ddj.3

**Status**: Not investigated in detail. A dataset alone is not a detector — a compatible trained model would also be required.

**Verdict**: Do not download without approval. Dataset alone is insufficient for lesion predictions.

---

## Impacted Tooth Detection

No specific dataset or model identified in the project plan. Further research needed.

---

## General Guidance

- **Do not download large datasets or model weights without explicit approval.**
- All pathology modules currently report "Not Configured — Unvalidated" — this is an honest state, not a negative finding.
- Do not fabricate detections, confidence scores, or accuracy metrics.
- Do not substitute random or untrained weights.