from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms

logging.basicConfig(level=logging.INFO, format="%(message)s")
LOGGER = logging.getLogger(__name__)

LABEL_MAP = {0: "BENIGN", 1: "MALIGNANT"}


def get_device() -> torch.device:
    """Return the best available computation device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def load_model(checkpoint_path: str) -> Tuple[nn.Module, torch.device]:
    """Load the EfficientNet-B0 model and weights from a checkpoint file."""
    device = get_device()
    model = models.efficientnet_b0(pretrained=False)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)

    checkpoint = Path(checkpoint_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        state = torch.load(str(checkpoint), map_location=device)
    except Exception as exc:
        raise RuntimeError(f"Could not load checkpoint '{checkpoint_path}': {exc}") from exc

    if isinstance(state, dict) and "model_state_dict" in state:
        state_dict = state["model_state_dict"]
    else:
        state_dict = state

    try:
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        raise RuntimeError(f"Checkpoint weights do not match model architecture: {exc}") from exc

    model.to(device)
    model.eval()
    LOGGER.info("Loaded EfficientNet-B0 model from %s on %s", checkpoint_path, device)
    return model, device


def get_transform() -> transforms.Compose:
    """Return the preprocessing transform used for inference."""
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def predict(model: nn.Module, device: torch.device, image: Image.Image) -> Tuple[int, float, np.ndarray, torch.Tensor]:
    """Run inference for a single image and return prediction details."""
    transform = get_transform()
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        pred = torch.argmax(probs).item()
        confidence = probs[pred].item()

    return pred, confidence, probs.cpu().numpy(), tensor


def render_sidebar() -> Tuple[str, bool]:
    """Render the sidebar controls and return the selected settings."""
    with st.sidebar:
        st.header("Model Settings")
        checkpoint_path = st.text_input(
            "Model checkpoint path",
            value="models/effnet_b0_epoch_15.pth",
        )
        show_gradcam = st.checkbox("Show Grad-CAM heatmap", value=False)
        st.write("Note: Grad-CAM requires a compatible model and may take longer.")
    return checkpoint_path, show_gradcam


def load_uploaded_image(uploaded_file) -> Image.Image | None:
    """Load the uploaded image into a PIL image object."""
    try:
        with Image.open(uploaded_file) as image:
            return image.convert("RGB")
    except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
        LOGGER.exception("Failed to load uploaded image")
        st.error(f"Failed to load the uploaded image: {exc}")
        return None


def main() -> None:
    """Run the Streamlit application for EfficientNet-B0 inference."""
    st.set_page_config(
        page_title="EfficientNet-B0 Breast Cancer Detector",
        page_icon="🧠",
        layout="centered",
    )

    st.title("🧠 Breast Cancer Detection AI")
    st.write("Upload a mammogram image to predict Benign vs Malignant.")

    checkpoint_path, show_gradcam = render_sidebar()
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg", "tif", "tiff"])

    if uploaded_file is None:
        st.info("Upload a mammogram image to start prediction.")
        return

    image = load_uploaded_image(uploaded_file)
    if image is None:
        return

    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Run Prediction"):
        try:
            model, device = load_model(checkpoint_path)
        except Exception as exc:
            LOGGER.exception("Failed to load model")
            st.error(f"Failed to load model: {exc}")
            return

        with st.spinner("Predicting..."):
            pred_class, confidence, probs, tensor = predict(model, device, image)

        st.subheader("Prediction Result")
        st.write(f"**Class:** {LABEL_MAP[pred_class]}")
        st.write(f"**Confidence:** {confidence:.4f}")
        st.write(f"**Scores:** Benign={probs[0]:.4f}, Malignant={probs[1]:.4f}")

        if pred_class == 1:
            st.error("⚠️ High risk detected (MALIGNANT)")
        else:
            st.success("✅ Low risk detected (BENIGN)")

        if show_gradcam:
            try:
                heatmap = generate_gradcam(model, tensor, device)
                st.subheader("Grad-CAM Heatmap")
                st.image(heatmap, caption="Grad-CAM overlay", use_column_width=True)
            except Exception as exc:
                LOGGER.exception("Grad-CAM generation failed")
                st.warning(f"Grad-CAM generation failed: {exc}")


def generate_gradcam(model: nn.Module, image_tensor: torch.Tensor, device: torch.device):
    """Generate a Grad-CAM heatmap for the provided tensor input."""
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
    except ImportError as exc:
        raise ImportError("pytorch-grad-cam is required for Grad-CAM. Install it with pip.") from exc

    target_layer = model.features[-1]
    cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=torch.cuda.is_available())
    grayscale_cam = cam(input_tensor=image_tensor)[0]

    image_np = image_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
    image_np = image_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    image_np = np.clip(image_np, 0, 1)

    heatmap = show_cam_on_image(image_np, grayscale_cam, use_rgb=True)
    return heatmap


if __name__ == "__main__":
    main()
