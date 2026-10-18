"""
app.py
======
AI-Powered Dental Panoramic X-Ray Analysis System
Streamlit Application Entry Point
"""

from datetime import datetime, timezone  # noqa: F401 (legacy import)
import uuid
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

from preprocessing.validation import validate_and_load_image
from preprocessing.enhancement import enhance_xray, enhance_rgb_xray
from modules.segmentation.fdi_resolver import resolve_dental_arch
from modules.segmentation.visualization import create_segmentation_overlay
from modules.segmentation.config import DEFAULT_CHECKPOINT_PATH
from modules.segmentation.backend import (
    SegmentationBackendInfo,
    resolve_segmentation_backend,
    run_segmentation,
)
from modules.segmentation.yolo_adapter import (
    download_yolo_weights,
    yolo_backend_available,
    yolo_weights_info,
)
from modules.pathology.registry import evaluate_all_pathologies
from modules.pathology.yolo_detector import (
    download_pathology_weights,
    pathology_model_available,
    pathology_weights_info,
    register_yolo_pathology_detectors,
)
from modules.analysis_status import get_module_status
from utils.report import create_report, report_as_json, report_as_text
from utils.pdf_report import generate_dental_pdf_report


st.set_page_config(
    page_title="AI-Powered Dental X-Ray Analysis",
    page_icon="🦷",
    layout="wide",
)

# Register the public pretrained YOLO pathology detectors with the registry.
# Safe to call on every rerun; the underlying model loads lazily on first
# inference. If registration fails, the app still works and conditions fall
# back to explicit NOT_CONFIGURED statuses.
try:
    _registered_conditions = register_yolo_pathology_detectors()
except Exception:
    _registered_conditions = []

st.title("🦷 AI-Powered Dental Panoramic X-Ray Analysis")

st.caption(
    "Automated X-ray enhancement, tooth instance segmentation, anatomical FDI numbering, "
    "and pathology evaluation research prototype"
)

st.warning(
    "⚠️ **Educational and Research Use Only**: This application does not provide a definitive "
    "medical or dental diagnosis. All outputs, contours, tooth identifications, and condition "
    "statuses require independent review by a qualified dental professional."
)


# ── 1. Upload & Image Validation ───────────────────────────────────────────────

st.subheader("1. Upload a dental radiograph")

uploaded_file = st.file_uploader(
    "Choose a panoramic radiograph (PNG, JPG, JPEG, TIF, TIFF)",
    type=["png", "jpg", "jpeg", "tif", "tiff"],
)

if uploaded_file is None:
    st.info("Upload a dental panoramic X-ray to begin analysis.")
    st.stop()

val_result = validate_and_load_image(uploaded_file, filename=uploaded_file.name)

if not val_result.is_valid:
    st.error(f"❌ Could not process this image: {val_result.error_message}")
    st.stop()

for warn_msg in val_result.warnings:
    st.warning(f"⚠️ {warn_msg}")

original_rgb = val_result.rgb_array
grayscale_array = val_result.grayscale_array
original_pil = val_result.original_pil


# ── 2. Image Enhancement ───────────────────────────────────────────────────────

st.subheader("2. Radiograph enhancement")

with st.expander("⚙️ Enhancement Parameters (CLAHE)", expanded=False):
    col_e1, col_e2 = st.columns(2)
    with col_e1:
        clip_limit = st.slider("Contrast Clip Limit", min_value=1.0, max_value=6.0, value=2.0, step=0.5)
    with col_e2:
        grid_size_val = st.select_slider("Tile Grid Size", options=[4, 8, 12, 16], value=8)
    tile_grid_size = (grid_size_val, grid_size_val)

try:
    enhanced_gray = enhance_xray(grayscale_array, clip_limit=clip_limit, tile_grid_size=tile_grid_size)
    enhanced_rgb = enhance_rgb_xray(original_rgb, clip_limit=clip_limit, tile_grid_size=tile_grid_size)
except Exception as exc:
    st.error(f"Enhancement error: {exc}")
    enhanced_gray = grayscale_array
    enhanced_rgb = original_rgb

col_left, col_right = st.columns(2)

with col_left:
    st.markdown("**Original X-ray**")
    st.image(original_pil, use_container_width=True)

with col_right:
    st.markdown(f"**CLAHE-Enhanced X-ray (Clip={clip_limit}, Grid={grid_size_val}×{grid_size_val})**")
    st.image(enhanced_gray, clamp=True, use_container_width=True)

st.caption(f"Image dimensions: {val_result.width} × {val_result.height} pixels | Format: {val_result.format}")


# ── 3. Tooth Instance Segmentation & FDI Numbering ─────────────────────────────

st.subheader("3. Tooth instance segmentation & FDI numbering")

# ── 3a. Backend selection & weight management ─────────────────────────────────

seg_backend_choice = st.radio(
    "Segmentation backend",
    options=["auto", "yolov8", "maskrcnn", "none"],
    index=0,
    horizontal=True,
    help=(
        "auto: prefer the public YOLOv8 32-tooth model if weights are present, "
        "else the local Mask R-CNN checkpoint. Force a specific backend, or "
        "'none' to disable segmentation entirely."
    ),
)

seg_info: SegmentationBackendInfo = resolve_segmentation_backend(seg_backend_choice)

info_col1, info_col2 = st.columns(2)

with info_col1:
    if seg_info.name == "yolov8":
        st.success(
            f"Segmentation: {seg_info.source}\n\n"
            f"Weights: `{seg_info.weights_path}`\n\n"
            "Note: this backend segments teeth only — restorations "
            "(bridge/crown/implant) are NOT detected."
        )
    elif seg_info.name == "maskrcnn":
        st.success(
            f"Segmentation: {seg_info.source}\n\n"
            f"Weights: `{seg_info.weights_path}`\n\n"
            "This backend also detects restorations (bridge/crown/implant)."
        )
    elif seg_backend_choice == "yolov8":
        st.warning(
            "YOLOv8 segmentation weights not found. Download them below or set "
            "DENTAL_SEG_YOLO_WEIGHTS to a local copy."
        )
    elif seg_backend_choice == "maskrcnn":
        st.warning(
            "Mask R-CNN checkpoint not found. Train one and place it at "
            f"`{DEFAULT_CHECKPOINT_PATH}` or set DENTAL_SEG_WEIGHTS."
        )
    elif seg_backend_choice == "none":
        st.info("Segmentation disabled. Analysis will proceed without tooth numbering.")
    else:
        st.warning(
            "No segmentation backend available. Download the YOLOv8 weights below "
            "(recommended) or configure a local Mask R-CNN checkpoint."
        )

    if st.button(
        "⬇️ Download YOLOv8 tooth-segmentation weights (~7 MB)",
        disabled=yolo_backend_available(),
        use_container_width=True,
    ):
        try:
            with st.spinner("Downloading YOLOv8 weights from Hugging Face..."):
                download_yolo_weights()
            st.success("YOLOv8 segmentation weights downloaded.")
            st.rerun()
        except Exception as exc:
            st.error(f"Failed to download YOLOv8 weights: {exc}")

with info_col2:
    patho_info = pathology_weights_info()
    if patho_info["available"]:
        st.success(
            f"Pathology detector: YOLO11n ({patho_info['repo']}, "
            f"{patho_info['license']}, ~{patho_info['approx_size_mb']} MB)\n\n"
            "Covers: dental caries, periapical lesions, impacted teeth.\n\n"
            "NOT covered: periodontal bone loss (no public weights exist — stays unavailable)."
        )
    else:
        st.warning(
            "Pathology detector weights not found. Download them below or set "
            "DENTAL_PATHO_YOLO_WEIGHTS to a local copy."
        )

    if st.button(
        "⬇️ Download YOLO11 pathology-detector weights (~5.5 MB)",
        disabled=patho_info["available"],
        use_container_width=True,
    ):
        try:
            with st.spinner("Downloading YOLO11 pathology weights from Hugging Face..."):
                download_pathology_weights()
            st.success("YOLO11 pathology weights downloaded.")
            st.rerun()
        except Exception as exc:
            st.error(f"Failed to download pathology weights: {exc}")

    if yolo_backend_available():
        seg_yolo_info = yolo_weights_info()
        st.caption(
            f"Segmentation model card: {seg_yolo_info['url']} "
            f"(license: {seg_yolo_info['license']})"
        )
    st.caption(
        f"Pathology model card: {patho_info['url']} "
        f"(license: {patho_info['license']})"
    )

with st.expander("⚙️ Segmentation & Resolver Settings", expanded=False):
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        conf_thresh = st.slider("Detection Confidence Threshold", min_value=0.15, max_value=0.80, value=0.30, step=0.05)
    with col_s2:
        overlap_thresh = st.slider("Duplicate Overlap IoU Threshold", min_value=0.20, max_value=0.70, value=0.40, step=0.05)
    with col_s3:
        show_labels = st.checkbox("Show Centroid FDI Labels", value=True)
        show_contours = st.checkbox("Show Mask Contours", value=True)

dental_arch = None
segmentation_overlay = None
seg_device = "CPU"
seg_backend_used = None

if seg_info.name is not None:
    try:
        with st.spinner(
            f"Running tooth segmentation ({seg_info.name}) and anatomical FDI resolver..."
        ):
            raw_seg_output, seg_backend_used = run_segmentation(
                image_rgb=enhanced_rgb,
                backend=seg_backend_choice,
                conf_threshold=conf_thresh,
            )
        seg_device = raw_seg_output.device_used
        st.success(
            f"✅ Segmentation backend active: **{seg_backend_used}** — {seg_info.source} "
            f"(weights: `{seg_info.weights_path}`, device: {seg_device})."
        )

        dental_arch = resolve_dental_arch(
            raw_detections=raw_seg_output.detections,
            image_shape=(val_result.height, val_result.width),
            overlap_iou_threshold=overlap_thresh,
            min_confidence=conf_thresh,
        )
        segmentation_overlay = create_segmentation_overlay(
            image_rgb=original_rgb,
            teeth=dental_arch.teeth,
            restorations=dental_arch.restorations,
            unassigned=dental_arch.unassigned,
            show_labels=show_labels,
            show_contours=show_contours,
        )

        # Summary Metrics
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.metric("Confirmed Teeth", dental_arch.total_detected_teeth)
        with m2:
            st.metric("Restorations", dental_arch.total_restorations)
        with m3:
            st.metric("Unassigned / Uncertain", len(dental_arch.unassigned))
        with m4:
            st.metric("Unobserved Permanent", len(dental_arch.missing_teeth))
        with m5:
            st.metric("Inference Time", f"{raw_seg_output.inference_time:.2f}s")

        # Display Visual Overlay
        st.image(segmentation_overlay, caption="Tooth Instance Segmentation Overlay with FDI Numbering", use_container_width=True)

        # Display Quadrant Distribution Pills
        q_counts = dental_arch.quadrant_counts
        st.markdown(
            f"**Quadrant Tooth Counts:** "
            f"`Q1 (Upper Right): {q_counts['UR']}` | "
            f"`Q2 (Upper Left): {q_counts['UL']}` | "
            f"`Q3 (Lower Left): {q_counts['LL']}` | "
            f"`Q4 (Lower Right): {q_counts['LR']}`"
        )

        # Tabs for Tooth Table, Restorations, and Unobserved Teeth
        tab_teeth, tab_rest, tab_miss, tab_uncertain = st.tabs([
            "🦷 FDI Tooth Table",
            "🛠️ Restorations & Prosthetics",
            "🔍 Unobserved Permanent Teeth",
            "❓ Uncertain Detections",
        ])

        with tab_teeth:
            if dental_arch.teeth:
                teeth_table_data = [
                    {
                        "FDI #": t.fdi_number,
                        "Quadrant": t.quadrant,
                        "Anatomical Description": t.anatomical_name,
                        "Confidence": f"{t.confidence * 100:.1f}%",
                        "Status": t.status,
                        "Validation Note": t.status_note,
                    }
                    for t in dental_arch.teeth
                ]
                st.dataframe(pd.DataFrame(teeth_table_data), use_container_width=True, hide_index=True)
            else:
                st.info("No confirmed tooth instances detected at current confidence threshold.")

        with tab_rest:
            if dental_arch.restorations:
                rest_table_data = [
                    {
                        "Restoration Type": r.restoration_type,
                        "Centroid (X, Y)": f"({r.centroid[0]}, {r.centroid[1]})",
                        "Mask Area (px)": r.area,
                        "Confidence": f"{r.confidence * 100:.1f}%",
                        "Clinical Classification": r.status_note,
                    }
                    for r in dental_arch.restorations
                ]
                st.dataframe(pd.DataFrame(rest_table_data), use_container_width=True, hide_index=True)
            else:
                st.info("No restorations (implants, bridges, fillings) detected.")

        with tab_miss:
            if dental_arch.missing_teeth:
                missing_str = ", ".join(dental_arch.missing_teeth)
                st.markdown(f"**Permanent teeth unobserved in this 2D radiograph ({len(dental_arch.missing_teeth)} teeth):**")
                st.code(missing_str)
                st.caption(
                    "Note: 'Unobserved' indicates that a clear tooth contour was not identified in this scan. "
                    "This does not confirm clinical extraction, impaction, or agenesis without independent clinical correlation."
                )
            else:
                st.success("All 32 permanent teeth were identified in this radiograph.")

        with tab_uncertain:
            if dental_arch.unassigned:
                unc_table_data = [
                    {
                        "Candidate Instance": f"#{u.instance_id}",
                        "Centroid (X, Y)": f"({u.centroid[0]}, {u.centroid[1]})",
                        "Confidence": f"{u.confidence * 100:.1f}%",
                        "Uncertainty Reason": u.status_note,
                    }
                    for u in dental_arch.unassigned
                ]
                st.dataframe(pd.DataFrame(unc_table_data), use_container_width=True, hide_index=True)
            else:
                st.success("Zero unresolved tooth conflicts. All proposals successfully assigned or cleanly filtered.")

    except FileNotFoundError as exc:
        st.error(f"Segmentation weights not found: {exc}")
        st.info("Use the download buttons above, or set the appropriate weights env var.")
    except RuntimeError as exc:
        st.error(f"Failed to load segmentation model: {exc}")
        st.info("The weights may be incompatible or corrupted. Try re-downloading them above.")
    except Exception as exc:
        st.error(f"Tooth segmentation failed: {exc}")
        st.info("Analysis will continue with segmentation marked as unavailable.")
else:
    st.info(
        "Tooth segmentation is unavailable for the selected backend. "
        "Download the YOLOv8 weights above (recommended) or configure a Mask R-CNN checkpoint."
    )


# ── 4. Dental Pathology Detection Status ───────────────────────────────────────

st.subheader("4. Dental pathology detection")

pathology_results = evaluate_all_pathologies(original_rgb)

patho_col1, patho_col2 = st.columns(2)
items = list(pathology_results.items())

for i, (code, res) in enumerate(items):
    col = patho_col1 if i % 2 == 0 else patho_col2
    with col:
        with st.container(border=True):
            st.markdown(f"**{res.condition_name}**")
            st.warning(f"Status: {res.status_display}")
            st.write(res.status_explanation)
            st.caption(f"**Limitation:** {res.validation_limitation}")


# ── 5. Download Clinical Report ────────────────────────────────────────────────

st.subheader("5. Download clinical report")

analysis_uuid = f"DENT-{uuid.uuid4().hex[:8].upper()}"

# In-memory PDF generation
try:
    pdf_bytes = generate_dental_pdf_report(
        original_rgb=original_rgb,
        enhanced_rgb=enhanced_rgb,
        segmentation_overlay_rgb=segmentation_overlay,
        dental_arch=dental_arch,
        pathology_results=pathology_results,
        filename=uploaded_file.name,
        analysis_id=analysis_uuid,
        model_provenance={
            "segmentation_model": (
                f"{seg_backend_used.upper()} — {seg_info.source}"
                if seg_backend_used and seg_info.name
                else "Unavailable (no backend/weights)"
            ),
            "segmentation_weights": (
                seg_info.weights_path if seg_info.name else "N/A"
            ),
            "device": str(seg_device),
            "pathology_model": (
                f"YOLO11n — {pathology_weights_info()['repo']} "
                f"({pathology_weights_info()['license']})"
                if pathology_model_available()
                else "Unavailable"
            ),
            "validation_dataset": (
                "DENTEX + OralXrays-9 (upstream, unverified in this repository)"
            ),
        },
    )
    pdf_ready = True
except Exception as exc:
    st.error(f"Could not generate PDF report: {exc}")
    pdf_ready = False
    pdf_bytes = b""

# Legacy text/JSON reports for backward compatibility
module_status_dict = get_module_status()
report_dict = create_report(filename=uploaded_file.name, module_status=module_status_dict)

col_d1, col_d2, col_d3 = st.columns(3)

with col_d1:
    if pdf_ready:
        st.download_button(
            label="📄 Download PDF Report (ReportLab)",
            data=pdf_bytes,
            file_name=f"dental_analysis_report_{analysis_uuid}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    else:
        st.button("📄 PDF Generation Unavailable", disabled=True, use_container_width=True)

with col_d2:
    st.download_button(
        label="Download TXT Report",
        data=report_as_text(report_dict),
        file_name=f"dental_analysis_report_{analysis_uuid}.txt",
        mime="text/plain",
        use_container_width=True,
    )

with col_d3:
    st.download_button(
        label="Download JSON Report",
        data=report_as_json(report_dict),
        file_name=f"dental_analysis_report_{analysis_uuid}.json",
        mime="application/json",
        use_container_width=True,
    )

st.divider()
st.caption(
    "Analysis ID: " + analysis_uuid + " | AI-Assisted Dental Radiograph Analysis System | "
    "Educational & Research Prototype"
)
