# app.py

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import streamlit as st
from PIL import Image

from pipelines.inference_pipeline import BreastCancerInferencePipeline

logging.basicConfig(level=logging.INFO, format="%(message)s")
LOGGER = logging.getLogger(__name__)


def configure_page() -> None:
    """Configure the Streamlit page metadata."""
    st.set_page_config(
        page_title="Breast Cancer AI",
        page_icon="🩺",
        layout="wide",
    )


@st.cache_resource
def load_pipeline() -> BreastCancerInferencePipeline:
    """Create and cache the inference pipeline."""
    return BreastCancerInferencePipeline()


def render_header() -> None:
    """Render the application header and description."""
    st.title("🩺 Explainable Breast Cancer Detection System")
    st.markdown(
        """
        **Research-level AI system for mammogram analysis**

        Features:
        - Tumor segmentation
        - Benign / malignant classification
        - BI-RADS prediction
        - Confidence estimation
        - Grad-CAM++ explainability
        - PDF report generation
        """
    )


def render_sidebar() -> Tuple[str, bool]:
    """Render the sidebar controls and return the selected options."""
    with st.sidebar:
        st.header("System Settings")

        patient_id = st.text_input(
            "Patient / Case ID",
            value="Demo Patient",
        )

        generate_report = st.checkbox(
            "Generate PDF Report",
            value=True,
        )

        st.warning(
            "This system is for research and educational use only. "
            "It must not be used as a medical diagnosis."
        )

    return patient_id, generate_report


def load_uploaded_image(uploaded_file) -> Optional[np.ndarray]:
    """Load an uploaded image into a NumPy array."""
    if uploaded_file is None:
        return None

    try:
        image = Image.open(uploaded_file).convert("RGB")
        return np.array(image)
    except Exception as exc:
        LOGGER.exception("Failed to read uploaded image")
        st.error(f"Unable to read the uploaded image: {exc}")
        return None


def run_inference(image_np: np.ndarray, patient_id: str, generate_report: bool) -> Optional[Dict[str, Any]]:
    """Run the prediction pipeline for the provided image."""
    with st.spinner("Running full AI pipeline..."):
        try:
            result = pipeline.predict(
                image=image_np,
                patient_id=patient_id,
                generate_report=generate_report,
            )
        except Exception as exc:
            LOGGER.exception("AI pipeline failed")
            st.error(f"AI pipeline failed: {exc}")
            return None

    if result.get("status") == "error":
        st.error(f"Inference failed: {result.get('error')}")
        return None

    return result


def render_results(result: Dict[str, Any], generate_report: bool) -> None:
    """Render the prediction results and supporting visualizations."""
    st.success("Analysis Completed")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Prediction", result["prediction"])
    with col2:
        st.metric("Cancer Confidence", f"{result['confidence']:.2f}%")
    with col3:
        st.metric("BI-RADS", result["birads"])

    col4, col5, col6 = st.columns(3)
    with col4:
        st.metric("BI-RADS Confidence", f"{result['birads_confidence']:.2f}%")
    with col5:
        st.metric("Final Confidence", f"{result['final_confidence']:.2f}%")
    with col6:
        st.metric("Reliability", result["reliability"])

    st.divider()

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "Processed Image",
            "Segmentation",
            "ROI",
            "Grad-CAM++",
            "Raw Results",
        ]
    )

    with tab1:
        st.image(
            result["processed"],
            caption="Preprocessed Mammogram",
            use_container_width=True,
        )

    with tab2:
        st.image(
            result["mask"],
            caption="Predicted Tumor Mask",
            use_container_width=True,
        )

        roi_overlay = result.get("roi_overlay")
        if roi_overlay is not None:
            st.image(
                roi_overlay,
                caption="Detected ROI Bounding Box",
                use_container_width=True,
            )
        else:
            st.info("ROI overlay unavailable.")

    with tab3:
        st.image(
            result["roi"],
            caption="Extracted ROI",
            use_container_width=True,
        )
        st.write("Bounding Box:", result["bbox"])

    with tab4:
        if result["heatmap"] is not None:
            st.image(
                result["heatmap"],
                caption="Grad-CAM++ Explainability Heatmap",
                use_container_width=True,
            )
        else:
            st.info("Grad-CAM++ explainability heatmap was unavailable.")

    with tab5:
        st.json(
            {
                "prediction": result["prediction"],
                "class_id": result["class_id"],
                "probability": result["probability"],
                "confidence": result["confidence"],
                "probabilities": result["probabilities"],
                "birads": result["birads"],
                "birads_confidence": result["birads_confidence"],
                "uncertainty": result["uncertainty"],
                "reliability": result["reliability"],
                "final_confidence": result["final_confidence"],
                "bbox": result["bbox"],
                "status": result["status"],
            }
        )

    if generate_report and result.get("report_path"):
        with open(result["report_path"], "rb") as file:
            st.download_button(
                label="Download PDF Report",
                data=file,
                file_name="breast_cancer_ai_report.pdf",
                mime="application/pdf",
            )


configure_page()
pipeline = load_pipeline()
render_header()
patient_id, generate_report = render_sidebar()

uploaded_file = st.file_uploader(
    "Upload Mammogram Image",
    type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
)

if uploaded_file is not None:
    image_np = load_uploaded_image(uploaded_file)
    if image_np is not None:
        st.subheader("Uploaded Image")
        st.image(
            image_np,
            caption="Original Mammogram",
            use_container_width=True,
        )

        if st.button("Run AI Analysis", type="primary"):
            result = run_inference(image_np, patient_id, generate_report)
            if result is not None:
                render_results(result, generate_report)
else:
    st.info("Upload a mammogram image to start analysis.")
