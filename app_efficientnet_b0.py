from __future__ import annotations

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


@st.cache_resource
def load_model(checkpoint_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.efficientnet_b0(pretrained=False)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, device


def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def predict(model, device, image: Image.Image):
    transform = get_transform()
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        pred = torch.argmax(probs).item()
        confidence = probs[pred].item()
    return pred, confidence, probs.cpu().numpy(), tensor


def main() -> None:
    st.set_page_config(
        page_title="EfficientNet-B0 Breast Cancer Detector",
        page_icon="🧠",
        layout="centered",
    )

    st.title("🧠 Breast Cancer Detection AI")
    st.write("Upload a mammogram image to predict Benign vs Malignant.")

    with st.sidebar:
        st.header("Model Settings")
        checkpoint_path = st.text_input(
            "Model checkpoint path",
            value="models/effnet_b0_epoch_15.pth",
        )
        show_gradcam = st.checkbox("Show Grad-CAM heatmap", value=False)
        st.write("Note: Grad-CAM requires a compatible model and may take longer.")

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg", "tif", "tiff"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Run Prediction"):
            try:
                model, device = load_model(checkpoint_path)
            except Exception as exc:
                st.error(f"Failed to load model: {exc}")
                return

            with st.spinner("Predicting..."):
                pred_class, confidence, probs, tensor = predict(model, device, image)

            label_map = {0: "BENIGN", 1: "MALIGNANT"}
            st.subheader("Prediction Result")
            st.write(f"**Class:** {label_map[pred_class]}")
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
                    st.warning(f"Grad-CAM generation failed: {exc}")
    else:
        st.info("Upload a mammogram image to start prediction.")


def generate_gradcam(model: nn.Module, image_tensor: torch.Tensor, device: torch.device):
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
