# app.py

from __future__ import annotations

import logging
import re
from html import escape
from time import perf_counter
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import streamlit as st
from PIL import Image, ImageOps

from config import (
    BATCH_SIZE,
    CLASSIFICATION_CHECKPOINT,
    CLASSIFICATION_MODEL_NAME,
    DATASETS_DIR,
    DEVICE,
    IMAGE_SIZE,
    NUM_CHANNELS,
    PRETRAINED,
)
from pipelines.inference_pipeline import BreastCancerInferencePipeline

logging.basicConfig(level=logging.INFO, format="%(message)s")
LOGGER = logging.getLogger(__name__)
APP_VERSION = "1.0.0"
EVALUATION_REPORT_PATH = Path(__file__).resolve().parent / "reports" / "evaluation_report.txt"


def configure_page() -> None:
    """Configure the Streamlit page metadata."""
    st.set_page_config(
        page_title="Breast Cancer AI",
        layout="wide",
    )
    st.markdown(
        """
        <style>
        :root {
            --medical-ink: #1f2933;
            --medical-muted: #667085;
            --medical-border: #d9dee5;
            --medical-surface: #ffffff;
            --medical-background: #f5f7fa;
            --medical-accent: #245b7a;
        }
        .stApp {
            background: var(--medical-background);
            color: var(--medical-ink);
            overflow-x: hidden;
        }
        [data-testid="stHeader"] {
            background: transparent;
        }
        .block-container {
            max-width: 1360px;
            padding: 2.25rem 3rem 3rem;
        }
        .app-header {
            border-bottom: 1px solid var(--medical-border);
            margin-bottom: 1.75rem;
            padding-bottom: 1.25rem;
        }
        .app-kicker {
            color: var(--medical-accent);
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        .app-subtitle {
            color: var(--medical-muted);
            font-size: 0.95rem;
            margin: 0.35rem 0 0;
        }
        .stApp button,
        .stApp [data-testid="stDownloadButton"] button {
            border: 1px solid var(--medical-border);
            border-radius: 0.3rem;
            font-weight: 600;
            min-height: 2.5rem;
        }
        .stApp button[kind="primary"] {
            background: var(--medical-accent);
            border-color: var(--medical-accent);
        }
        [data-testid="stSidebar"] {
            background: #eef1f4;
            border-right: 1px solid var(--medical-border);
        }
        .configuration-row {
            border-bottom: 1px solid var(--medical-border);
            padding: 0.35rem 0;
        }
        .configuration-row:last-child {
            border-bottom: 0;
        }
        .configuration-label {
            color: var(--medical-muted);
            font-size: 0.72rem;
            text-transform: uppercase;
        }
        .configuration-value {
            color: var(--medical-ink);
            font-size: 0.86rem;
            overflow-wrap: anywhere;
        }
        .stApp h1,
        .stApp h2,
        .stApp h3,
        .stApp [data-testid="stMarkdownContainer"] {
            max-width: 100%;
            overflow-wrap: anywhere;
        }
        .diagnostic-summary {
            background: var(--medical-surface);
            border: 1px solid var(--medical-border);
        }
        .diagnostic-row {
            display: grid;
            gap: 1.5rem;
            grid-template-columns: minmax(10rem, 24%) minmax(0, 1fr);
            padding: 0.85rem 1rem;
        }
        .diagnostic-row + .diagnostic-row {
            border-top: 1px solid var(--medical-border);
        }
        .diagnostic-label {
            color: var(--medical-muted);
            font-size: 0.84rem;
            font-weight: 600;
        }
        .diagnostic-value {
            color: var(--medical-ink);
            font-size: 0.92rem;
            line-height: 1.5;
            overflow-wrap: anywhere;
        }
        @media (max-width: 768px) {
            .diagnostic-row {
                gap: 0.35rem;
                grid-template-columns: 1fr;
            }
            .block-container {
                padding: 1.25rem 1rem 2rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_pipeline() -> BreastCancerInferencePipeline:
    """Create and cache the inference pipeline."""
    return BreastCancerInferencePipeline()


def render_header() -> None:
    """Render the application identity and concise clinical context."""
    st.markdown(
        """
        <header class="app-header">
            <div class="app-kicker">Breast imaging analysis</div>
            <h1>Explainable Breast Cancer Detection</h1>
            <p class="app-subtitle">
                Research support tool for mammogram classification, confidence
                estimation, and visual explanation.
            </p>
        </header>
        """,
        unsafe_allow_html=True,
    )


def render_section_label(label: str) -> None:
    """Render a consistent dashboard section label."""
    st.subheader(label)


def render_configuration_row(label: str, value: str) -> None:
    """Render one compact label/value row in the configuration panel."""
    st.markdown(
        f"""
        <div class="configuration-row">
            <div class="configuration-label">{label}</div>
            <div class="configuration-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def load_training_metrics() -> Dict[str, str]:
    """Read available evaluation metrics for display in the sidebar."""
    metric_names = (
        "Validation Accuracy",
        "ROC-AUC",
        "Precision",
        "Recall",
        "F1-score",
    )
    metrics = {name: "Unavailable" for name in metric_names}
    if not EVALUATION_REPORT_PATH.exists():
        return metrics

    try:
        report = EVALUATION_REPORT_PATH.read_text(encoding="utf-8")
    except OSError:
        return metrics

    for name in metric_names:
        match = re.search(rf"^{re.escape(name)}:\s*([0-9.]+)", report, re.MULTILINE)
        if match:
            metrics[name] = match.group(1)
    return metrics


def render_configuration_panel() -> None:
    """Render the read-only application configuration sidebar."""
    training_metrics = load_training_metrics()
    with st.sidebar:
        st.subheader("Configuration")

        with st.expander("Model", expanded=True):
            render_configuration_row("Backbone", CLASSIFICATION_MODEL_NAME)
            render_configuration_row("Checkpoint", CLASSIFICATION_CHECKPOINT.name)
            render_configuration_row("Pretrained weights", "Enabled" if PRETRAINED else "Disabled")

        with st.expander("Dataset"):
            render_configuration_row("Dataset directory", str(DATASETS_DIR))
            render_configuration_row("Classes", "Benign, Malignant")

        with st.expander("Input Resolution"):
            render_configuration_row("Image size", f"{IMAGE_SIZE} x {IMAGE_SIZE} pixels")
            render_configuration_row("Channels", str(NUM_CHANNELS))

        with st.expander("Device"):
            render_configuration_row("Inference device", DEVICE.upper())

        with st.expander("Training Metrics"):
            for label, value in training_metrics.items():
                render_configuration_row(label, value)
            st.caption("Loaded from the latest evaluation report.")

        with st.expander("Application Version"):
            render_configuration_row("Version", APP_VERSION)
            render_configuration_row("Batch size", str(BATCH_SIZE))


def render_upload_panel() -> Tuple[str, bool, Any]:
    """Render upload and analysis controls and return their values."""
    render_section_label("Upload Mammogram")
    uploaded_file = st.file_uploader(
        "Select an image file",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
        label_visibility="collapsed",
    )
    patient_id = st.text_input("Patient / Case ID", value="Demo Patient")
    generate_report = st.checkbox("Generate PDF report", value=True)
    st.caption("Supported formats: PNG, JPEG, BMP, and TIFF.")
    return patient_id, generate_report, uploaded_file


def render_image_panel(image_np: Optional[np.ndarray]) -> None:
    """Render the uploaded image panel or its empty state."""
    render_section_label("Uploaded Image")
    if image_np is None:
        st.info("Upload a mammogram to preview it here.")
        return

    st.image(
        prepare_analysis_image(image_np),
        caption="Original mammogram",
        use_container_width=True,
    )


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


def render_result_heading() -> None:
    """Render the heading for the post-inference results area."""
    st.subheader("Diagnostic Summary")
    st.caption("Structured model output for the current mammogram.")


def render_diagnostic_summary(result: Dict[str, Any], inference_time_ms: float) -> None:
    """Render a structured diagnostic summary for the completed inference."""
    diagnosis = str(result["prediction"])
    interpretation = (
        f"The model classified this image as {diagnosis}. "
        "This result supports research workflows and does not replace clinical assessment."
    )
    rows = (
        ("Diagnosis", diagnosis),
        ("Prediction Confidence", f"{result['confidence']:.2f}%"),
        ("Clinical Interpretation", interpretation),
        (
            "Model Confidence",
            f"{result['final_confidence']:.2f}% ({result['reliability']})",
        ),
        ("Inference Time", f"{inference_time_ms:.0f} ms"),
    )
    row_markup = "".join(
        f"""
        <div class="diagnostic-row">
            <div class="diagnostic-label">{escape(label)}</div>
            <div class="diagnostic-value">{escape(value)}</div>
        </div>
        """
        for label, value in rows
    )
    st.markdown(
        f'<div class="diagnostic-summary">{row_markup}</div>',
        unsafe_allow_html=True,
    )


def prepare_analysis_image(image_np: np.ndarray, size: int = 512) -> np.ndarray:
    """Fit an analysis image onto a shared square canvas for aligned display."""
    image = Image.fromarray(image_np).convert("RGB")
    fitted = ImageOps.contain(image, (size, size))
    canvas = Image.new("RGB", (size, size), color=(245, 247, 250))
    offset = ((size - fitted.width) // 2, (size - fitted.height) // 2)
    canvas.paste(fitted, offset)
    return np.asarray(canvas)


def render_visual_results(result: Dict[str, Any]) -> None:
    """Render aligned original and model-attention images side by side."""
    render_section_label("Image Analysis")
    image_columns = st.columns(2)
    with image_columns[0]:
        render_section_label("Original Mammogram")
        st.image(
            prepare_analysis_image(result["original"]),
            use_container_width=True,
        )
    with image_columns[1]:
        render_section_label("Model Attention (Grad-CAM)")
        heatmap = result.get("heatmap")
        if heatmap is not None:
            st.image(
                prepare_analysis_image(heatmap),
                use_container_width=True,
            )
        else:
            st.info("Model attention is unavailable for this analysis.")


def render_report_section(result: Dict[str, Any], generate_report: bool) -> None:
    """Render the report action when a report was requested and created."""
    render_section_label("PDF Report")
    if not generate_report:
        st.caption("PDF report generation was not requested.")
        return
    if not result.get("report_path"):
        st.caption("A report could not be generated for this analysis.")
        return

    with open(result["report_path"], "rb") as file:
        st.download_button(
            label="Download PDF report",
            data=file,
            file_name="breast_cancer_ai_report.pdf",
            mime="application/pdf",
        )


def render_raw_results(result: Dict[str, Any]) -> None:
    """Render the detailed inference payload for technical inspection."""
    with st.expander("View raw results"):
        st.json(
            {
                "prediction": result["prediction"],
                "class_id": result["class_id"],
                "probability": result["probability"],
                "confidence": result["confidence"],
                "probabilities": result["probabilities"],
                "uncertainty": result["uncertainty"],
                "reliability": result["reliability"],
                "final_confidence": result["final_confidence"],
                "status": result["status"],
            }
        )


configure_page()
pipeline = load_pipeline()
render_configuration_panel()
render_header()
control_column, image_column = st.columns([0.9, 1.6], gap="large")
with control_column:
    patient_id, generate_report, uploaded_file = render_upload_panel()

with image_column:
    image_np = load_uploaded_image(uploaded_file)
    render_image_panel(image_np)

if image_np is not None:
    st.divider()
    if st.button("Run AI Analysis", type="primary", use_container_width=True):
        inference_started = perf_counter()
        result = run_inference(image_np, patient_id, generate_report)
        inference_time_ms = (perf_counter() - inference_started) * 1000
        if result is not None:
            render_result_heading()
            render_diagnostic_summary(result, inference_time_ms)
            render_visual_results(result)
            report_column, details_column = st.columns([1, 1], gap="large")
            with report_column:
                render_report_section(result, generate_report)
            with details_column:
                render_raw_results(result)
else:
    st.info("Upload a mammogram image to begin analysis.")
