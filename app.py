
import streamlit as st
import numpy as np
from PIL import Image, ImageOps

from preprocessing.enhancement import enhance_xray
from modules.analysis_status import get_module_status
from utils.report import create_report, report_as_json, report_as_text


st.set_page_config(
    page_title="Dental AI Analysis",
    page_icon="🦷",
    layout="wide",
)

st.title("🦷 AI-Powered Dental X-Ray Analysis")

st.caption(
    "Image enhancement, tooth segmentation, "
    "and multi-disease analysis research prototype"
)

st.warning(
    "Educational and research use only. This application "
    "does not provide a medical diagnosis. Consult a qualified "
    "dentist for clinical interpretation."
)

st.subheader("1. Upload a dental X-ray")

uploaded_file = st.file_uploader(
    "Choose a PNG, JPG, JPEG, or TIFF image",
    type=["png", "jpg", "jpeg", "tif", "tiff"],
)

if uploaded_file is None:
    st.info("Upload a dental X-ray to begin.")
    st.stop()

try:
    pil_image = ImageOps.exif_transpose(
        Image.open(uploaded_file)
    ).convert("RGB")

    grayscale = np.array(pil_image.convert("L"))
    enhanced = enhance_xray(grayscale)

except Exception as exc:
    st.error(f"Could not process this image: {exc}")
    st.stop()

st.subheader("2. Image enhancement")

left, right = st.columns(2)

with left:
    st.markdown("**Original X-ray**")
    st.image(pil_image, use_container_width=True)

with right:
    st.markdown("**CLAHE-enhanced X-ray**")
    st.image(enhanced, clamp=True, use_container_width=True)

st.caption(
    f"Image dimensions: {pil_image.width} × {pil_image.height} pixels"
)

st.subheader("3. AI analysis module status")

module_status = get_module_status()

for name, details in module_status.items():
    with st.container(border=True):
        st.markdown(f"**{name}**")
        st.info(details["status"])
        st.caption(details["description"])

st.subheader("4. Download report")

report = create_report(
    filename=uploaded_file.name,
    module_status=module_status,
)

col1, col2 = st.columns(2)

with col1:
    st.download_button(
        "Download TXT report",
        data=report_as_text(report),
        file_name="dental_analysis_report.txt",
        mime="text/plain",
        use_container_width=True,
    )

with col2:
    st.download_button(
        "Download JSON report",
        data=report_as_json(report),
        file_name="dental_analysis_report.json",
        mime="application/json",
        use_container_width=True,
    )

st.divider()

st.caption(
    "Disease predictions remain unavailable until compatible "
    "trained models are configured."
)
