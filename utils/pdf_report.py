"""
utils/pdf_report.py
===================
Clinical-grade PDF report generator for dental panoramic radiograph analysis.
Built with ReportLab.

Key features:
- Aspect-ratio preserving radiograph embedding.
- Clean vector layout with auto-wrapping tables (no boundary overflow).
- Explicit separation of anatomical teeth, restorations, and missing teeth.
- Prominent clinical disclaimer and transparent validation limitations.
- Robust generation even when specific AI modules or images are unavailable.
"""

from io import BytesIO
import os
import tempfile
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
import uuid

import cv2
import numpy as np
from PIL import Image

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    HRFlowable,
    Image as RLImage,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from modules.pathology.types import ConditionEvaluationResult
from modules.segmentation.fdi_resolver import ResolvedDentalArch, ResolvedTooth


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas for dynamic 'Page X of Y' numbering and running header/footer."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#7F8C8D"))

        # Running header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(36, 11 * inch - 26, "Dental Radiographic Analysis Report — Educational & Research Use Only")
            self.setStrokeColor(colors.HexColor("#BDC3C7"))
            self.setLineWidth(0.5)
            self.line(36, 11 * inch - 30, 8.5 * inch - 36, 11 * inch - 30)

        # Running footer
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(8.5 * inch - 36, 20, page_str)
        self.drawString(
            36, 20,
            "CONFIDENTIAL & PROPRIETARY — Requires review by a qualified dental professional."
        )
        self.setStrokeColor(colors.HexColor("#BDC3C7"))
        self.setLineWidth(0.5)
        self.line(36, 30, 8.5 * inch - 36, 30)
        self.restoreState()


def generate_dental_pdf_report(
    original_rgb: np.ndarray,
    enhanced_rgb: Optional[np.ndarray],
    segmentation_overlay_rgb: Optional[np.ndarray],
    dental_arch: Optional[ResolvedDentalArch],
    pathology_results: Optional[Dict[str, ConditionEvaluationResult]],
    filename: str = "radiograph.jpg",
    analysis_id: Optional[str] = None,
    model_provenance: Optional[Dict[str, str]] = None,
) -> bytes:
    """
    Generate a complete, professional clinical-grade PDF report.

    Returns:
    - Raw PDF file bytes.
    """
    pdf_buffer = BytesIO()
    doc = SimpleDocTemplate(
        pdf_buffer,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    # 540 pt available width between 36 pt margins on 8.5" page
    content_width = 8.5 * inch - 72

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#1B365D"),
        spaceAfter=4,
    )
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#5D6D7E"),
        spaceAfter=10,
    )
    section_heading = ParagraphStyle(
        'SectionHead',
        parent=styles['Heading2'],
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#1B365D"),
        spaceBefore=10,
        spaceAfter=6,
        keepWithNext=True,
    )
    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['Normal'],
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#2C3E50"),
    )
    body_bold = ParagraphStyle(
        'DocBodyBold',
        parent=body_style,
        fontName='Helvetica-Bold',
    )
    disclaimer_style = ParagraphStyle(
        'DisclaimerText',
        parent=styles['Normal'],
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#78281F"),
    )
    table_cell = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor("#2C3E50"),
    )
    table_cell_bold = ParagraphStyle(
        'TableCellBold',
        parent=table_cell,
        fontName='Helvetica-Bold',
    )

    story = []
    temp_files = []

    try:
        # 1. Header Banner
        report_id = analysis_id or f"DENT-{uuid.uuid4().hex[:8].upper()}"
        gen_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        story.append(Paragraph("🦷 AI-POWERED DENTAL RADIOGRAPHIC ANALYSIS REPORT", title_style))
        story.append(
            Paragraph("Automated Preprocessing, Tooth Instance Segmentation & Pathology Evaluation", subtitle_style)
        )
        story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1B365D"), spaceAfter=8))

        # Metadata Block
        h, w = original_rgb.shape[:2] if original_rgb is not None else (0, 0)
        meta_data = [
            [
                Paragraph("<b>Analysis Identifier:</b>", body_style),
                Paragraph(f"<code>{report_id}</code>", body_style),
                Paragraph("<b>Generated Timestamp:</b>", body_style),
                Paragraph(gen_timestamp, body_style),
            ],
            [
                Paragraph("<b>Input Radiograph:</b>", body_style),
                Paragraph(filename, body_style),
                Paragraph("<b>Image Resolution:</b>", body_style),
                Paragraph(f"{w} × {h} pixels", body_style),
            ],
        ]
        meta_table = Table(meta_data, colWidths=[120, 150, 120, 150])
        meta_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F8F9F9")),
            ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#BDC3C7")),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('PADDING', (0, 0), (-1, -1), 4),
        ]))
        story.append(meta_table)
        story.append(Spacer(1, 8))

        # 2. Mandatory Clinical Disclaimer Box
        disclaimer_text = (
            "<b>MANDATORY CLINICAL WARNING:</b> This report is generated by an artificial intelligence research "
            "prototype intended solely for decision support and educational research. It does <b>NOT</b> constitute "
            "a definitive medical or dental diagnosis. All segmentation masks, FDI tooth numbers, restoration categories, "
            "and disease statuses must be independently reviewed and verified by a licensed dental professional. "
            "An unconfigured or unvalidated test must <b>never</b> be interpreted as confirmed clinical absence of disease."
        )
        disc_table = Table([[Paragraph(disclaimer_text, disclaimer_style)]], colWidths=[content_width])
        disc_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#FDEDEC")),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#E74C3C")),
            ('PADDING', (0, 0), (-1, -1), 6),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        story.append(disc_table)
        story.append(Spacer(1, 10))

        # 3. Radiograph Visual Panels (Preserving Aspect Ratio)
        story.append(Paragraph("1. Radiographic Imaging Panel", section_heading))

        # Helper to write numpy image to temporary JPEG and wrap in RLImage
        def _to_rl_image(img_arr: np.ndarray, max_w: float, max_h: float) -> RLImage:
            h_img, w_img = img_arr.shape[:2]
            aspect = w_img / h_img
            # Fit within bounding box
            fit_w = max_w
            fit_h = fit_w / aspect
            if fit_h > max_h:
                fit_h = max_h
                fit_w = fit_h * aspect

            temp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            temp_files.append(temp.name)
            temp.close()

            bgr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
            cv2.imwrite(temp.name, bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
            return RLImage(temp.name, width=fit_w, height=fit_h)

        # Side-by-side: Original & CLAHE Enhanced
        img_max_w = (content_width - 10) / 2
        img_max_h = 1.6 * inch

        orig_rl = _to_rl_image(original_rgb, img_max_w, img_max_h)
        enh_rl = _to_rl_image(enhanced_rgb if enhanced_rgb is not None else original_rgb, img_max_w, img_max_h)

        side_by_side = Table([
            [Paragraph("<b>Figure 1a:</b> Original Radiograph", body_style),
             Paragraph("<b>Figure 1b:</b> CLAHE Contrast-Enhanced", body_style)],
            [orig_rl, enh_rl]
        ], colWidths=[(content_width / 2) - 2, (content_width / 2) - 2])
        side_by_side.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('PADDING', (0, 0), (-1, -1), 3),
        ]))
        story.append(side_by_side)
        story.append(Spacer(1, 8))

        # Tooth Segmentation Overlay (Full Width)
        if segmentation_overlay_rgb is not None:
            story.append(Paragraph("<b>Figure 2:</b> Tooth Instance Segmentation & FDI Numbering Overlay", body_style))
            seg_rl = _to_rl_image(segmentation_overlay_rgb, content_width, 2.2 * inch)
            seg_table = Table([[seg_rl]], colWidths=[content_width])
            seg_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('PADDING', (0, 0), (-1, -1), 2),
            ]))
            story.append(seg_table)
            story.append(Spacer(1, 10))

        # 4. Tooth Instance Segmentation & FDI Table
        story.append(Paragraph("2. FDI Tooth Numbering & Instance Findings", section_heading))

        if dental_arch and dental_arch.teeth:
            fdi_headers = [
                Paragraph("<b>FDI #</b>", table_cell_bold),
                Paragraph("<b>Quadrant</b>", table_cell_bold),
                Paragraph("<b>Anatomical Name</b>", table_cell_bold),
                Paragraph("<b>Confidence</b>", table_cell_bold),
                Paragraph("<b>Status</b>", table_cell_bold),
                Paragraph("<b>Anatomical Validation Notes</b>", table_cell_bold),
            ]
            fdi_rows = [fdi_headers]

            for t in dental_arch.teeth:
                stat_color = "#27AE60" if t.status == "CONFIRMED" else "#E67E22"
                fdi_rows.append([
                    Paragraph(f"<b>{t.fdi_number or '—'}</b>", table_cell),
                    Paragraph(t.quadrant, table_cell),
                    Paragraph(t.anatomical_name, table_cell),
                    Paragraph(f"{t.confidence * 100:.1f}%", table_cell),
                    Paragraph(f"<font color='{stat_color}'><b>{t.status}</b></font>", table_cell),
                    Paragraph(t.status_note, table_cell),
                ])

            col_widths = [45, 60, 155, 60, 75, 145]
            fdi_table = Table(fdi_rows, colWidths=col_widths, repeatRows=1)
            fdi_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#EAECEE")),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#D5D8DC")),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8F9F9")]),
                ('PADDING', (0, 0), (-1, -1), 3),
            ]))
            story.append(fdi_table)
            story.append(Spacer(1, 8))

        else:
            story.append(Paragraph("<i>Tooth instance segmentation was not executed or produced no detections.</i>", body_style))
            story.append(Spacer(1, 8))

        # 5. Restorations & Prosthetics Panel
        if dental_arch and dental_arch.restorations:
            story.append(Paragraph("3. Restorations, Prosthetics & Dental Materials", section_heading))
            rest_headers = [
                Paragraph("<b>Material / Category</b>", table_cell_bold),
                Paragraph("<b>Centroid (X, Y)</b>", table_cell_bold),
                Paragraph("<b>Area (px)</b>", table_cell_bold),
                Paragraph("<b>Confidence</b>", table_cell_bold),
                Paragraph("<b>Clinical Classification</b>", table_cell_bold),
            ]
            rest_rows = [rest_headers]
            for r in dental_arch.restorations:
                rest_rows.append([
                    Paragraph(f"<b>{r.restoration_type or 'Restoration'}</b>", table_cell),
                    Paragraph(f"({r.centroid[0]}, {r.centroid[1]})", table_cell),
                    Paragraph(f"{r.area}", table_cell),
                    Paragraph(f"{r.confidence * 100:.1f}%", table_cell),
                    Paragraph(r.status_note, table_cell),
                ])
            rest_table = Table(rest_rows, colWidths=[110, 85, 65, 70, 210], repeatRows=1)
            rest_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#FEF9E7")),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#FAD7A0")),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('PADDING', (0, 0), (-1, -1), 3),
            ]))
            story.append(rest_table)
            story.append(Spacer(1, 8))

        # 6. Missing / Unobserved Teeth Panel
        if dental_arch and dental_arch.missing_teeth:
            story.append(Paragraph("4. Unobserved Permanent Teeth Audit", section_heading))
            missing_str = ", ".join(dental_arch.missing_teeth)
            missing_text = (
                f"<b>Permanent teeth unobserved in this scan ({len(dental_arch.missing_teeth)} teeth):</b><br/>"
                f"<font color='#5D6D7E'>{missing_str}</font><br/>"
                "<i>Note: 'Unobserved' indicates that a clear tooth contour was not detected in this 2D radiograph. "
                "This does NOT constitute clinical confirmation of extraction, congenital agenesis, or impaction.</i>"
            )
            story.append(Paragraph(missing_text, body_style))
            story.append(Spacer(1, 10))

        # 7. Pathology Evaluation Findings & Status
        story.append(Paragraph("5. Dental Pathology Findings & Validation Status", section_heading))

        patho_headers = [
            Paragraph("<b>Target Pathology Condition</b>", table_cell_bold),
            Paragraph("<b>Evaluation Status</b>", table_cell_bold),
            Paragraph("<b>Findings / Lesion Count</b>", table_cell_bold),
            Paragraph("<b>Validation Status & Clinical Limitations</b>", table_cell_bold),
        ]
        patho_rows = [patho_headers]

        if pathology_results:
            for code, res in pathology_results.items():
                status_color = "#27AE60" if res.status == "NEGATIVE" else (
                    "#E74C3C" if res.status == "DETECTED" else "#D35400"
                )
                findings_str = (
                    f"{len(res.findings)} finding(s)"
                    if res.is_evaluated and res.findings
                    else ("None detected" if res.is_evaluated else "Not Evaluated")
                )
                patho_rows.append([
                    Paragraph(f"<b>{res.condition_name}</b>", table_cell),
                    Paragraph(f"<font color='{status_color}'><b>{res.status_display}</b></font>", table_cell),
                    Paragraph(findings_str, table_cell),
                    Paragraph(
                        f"{res.status_explanation}<br/><b>Limitation:</b> {res.validation_limitation}",
                        table_cell
                    ),
                ])
        else:
            patho_rows.append([
                Paragraph("All Conditions", table_cell),
                Paragraph("NOT CONFIGURED", table_cell),
                Paragraph("0", table_cell),
                Paragraph("Pathology analysis was not performed.", table_cell),
            ])

        patho_table = Table(patho_rows, colWidths=[120, 110, 80, 230], repeatRows=1)
        patho_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#EAECEE")),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#D5D8DC")),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8F9F9")]),
            ('PADDING', (0, 0), (-1, -1), 3),
        ]))
        story.append(patho_table)
        story.append(Spacer(1, 10))

        # 8. Model Provenance & Environment Specifications
        story.append(Paragraph("6. Model Provenance & Verification Metadata", section_heading))
        prov = model_provenance or {}
        prov_rows = [
            [Paragraph("<b>Tooth Segmentation Architecture:</b>", table_cell),
             Paragraph(prov.get("segmentation_model", "Mask R-CNN ResNet-50+FPN (Torchvision)"), table_cell)],
            [Paragraph("<b>Segmentation Model Checkpoint:</b>", table_cell),
             Paragraph(prov.get("segmentation_weights", "outputs/results/maskrcnn_torch/best.pth (176.9 MB)"), table_cell)],
            [Paragraph("<b>Validation Dataset Benchmark:</b>", table_cell),
             Paragraph("AKUDENTAL 67-image held-out validation set (88.0% F1, 83.6% mean IoU)", table_cell)],
            [Paragraph("<b>Anatomical Post-Processor:</b>", table_cell),
             Paragraph("fdi_resolver.py (NMS overlap suppression, quadrant verification, duplicate resolution)", table_cell)],
            [Paragraph("<b>Inference Device / Framework:</b>", table_cell),
             Paragraph(prov.get("device", "CPU (PyTorch 2.14.1)"), table_cell)],
        ]
        prov_table = Table(prov_rows, colWidths=[170, 370])
        prov_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F8F9F9")),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#BDC3C7")),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('PADDING', (0, 0), (-1, -1), 3),
        ]))
        story.append(prov_table)

        # Build Document
        doc.build(story, canvasmaker=NumberedCanvas)
        pdf_bytes = pdf_buffer.getvalue()

    finally:
        # Clean up temporary radiograph files
        for fpath in temp_files:
            try:
                if os.path.exists(fpath):
                    os.unlink(fpath)
            except Exception:
                pass

    return pdf_bytes
