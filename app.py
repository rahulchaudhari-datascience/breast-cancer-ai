# app.py

from __future__ import annotations

import streamlit as st
from PIL import Image
import numpy as np

from pipelines.inference_pipeline import BreastCancerInferencePipeline


st.set_page_config(
    page_title="Breast Cancer AI",
    page_icon="🩺",
    layout="wide",
)


@st.cache_resource
def load_pipeline():
    return BreastCancerInferencePipeline()


pipeline = load_pipeline()


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


uploaded_file = st.file_uploader(
    "Upload Mammogram Image",
    type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
)


if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    st.subheader("Uploaded Image")

    st.image(
        image_np,
        caption="Original Mammogram",
        use_container_width=True,
    )

    if st.button("Run AI Analysis", type="primary"):

        with st.spinner("Running full AI pipeline..."):

            result = pipeline.predict(
                image=image_np,
                patient_id=patient_id,
                generate_report=generate_report,
            )

        st.success("Analysis Completed")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Prediction",
                result["prediction"],
            )

        with col2:
            st.metric(
                "Cancer Confidence",
                f"{result['confidence']:.2f}%",
            )

        with col3:
            st.metric(
                "BI-RADS",
                result["birads"],
            )

        col4, col5, col6 = st.columns(3)

        with col4:
            st.metric(
                "BI-RADS Confidence",
                f"{result['birads_confidence']:.2f}%",
            )

        with col5:
            st.metric(
                "Final Confidence",
                f"{result['final_confidence']:.2f}%",
            )

        with col6:
            st.metric(
                "Reliability",
                result["reliability"],
            )

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

            st.image(
                result["roi_overlay"],
                caption="Detected ROI Bounding Box",
                use_container_width=True,
            )

        with tab3:
            st.image(
                result["roi"],
                caption="Extracted ROI",
                use_container_width=True,
            )

            st.write("Bounding Box:", result["bbox"])

        with tab4:
            st.image(
                result["heatmap"],
                caption="Grad-CAM++ Explainability Heatmap",
                use_container_width=True,
            )

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

else:
    st.info("Upload a mammogram image to start analysis.")
