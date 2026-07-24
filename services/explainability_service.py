from __future__ import annotations

from pathlib import Path
from typing import Optional
import logging

import cv2
import numpy as np
import torch

from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image

from services._shared import get_service_logger, normalize_uint8_image

from config import (
    DEVICE,
    IMAGE_SIZE,
    HEATMAP_OUTPUT_DIR,
)


class ExplainabilityService:
    """Grad-CAM++ explainability service.

    Returns ``None`` when explainability cannot be produced so calling pipelines
    can continue producing inference results.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        target_layer: Optional[torch.nn.Module] = None,
    ):
        self.logger = get_service_logger(self.__class__.__name__)
        self.device = DEVICE
        self.model = model.to(self.device)
        self.model.eval()

        self.target_layer = target_layer or self._auto_find_target_layer()

        HEATMAP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        image: np.ndarray,
        class_id: Optional[int] = None,
        save_path: Optional[str] = None,
    ) -> Optional[np.ndarray]:
        if image is None:
            raise ValueError("Image must be provided for explainability generation.")

        input_tensor = self._preprocess_image(image)

        try:
            cam = GradCAMPlusPlus(model=self.model, target_layers=[self.target_layer])
        except Exception as exc:
            self.logger.warning("Failed to initialize GradCAM++: %s", exc)
            return None

        targets = None
        if class_id is not None:
            try:
                from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

                targets = [ClassifierOutputTarget(class_id)]
            except Exception:
                targets = None

        try:
            with torch.enable_grad():
                grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
        except Exception as exc:
            self.logger.warning("GradCAM++ generation failed: %s", exc)
            return None

        try:
            rgb_image = self._prepare_rgb_for_overlay(image)
            heatmap = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)
        except Exception as exc:
            self.logger.warning("Failed to render heatmap overlay: %s", exc)
            return None

        if save_path is None:
            save_path = str(HEATMAP_OUTPUT_DIR / "gradcam_output.png")

        try:
            self.save_heatmap(heatmap, save_path)
        except Exception as exc:
            self.logger.warning("Could not save heatmap: %s", exc)

        return heatmap

    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Apply the fixed explainability preprocessing path."""
        image = self._ensure_rgb(image)

        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        image = image.astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        image = (image - mean) / std

        tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
        tensor = tensor.unsqueeze(0)

        return tensor.to(self.device)

    def _prepare_rgb_for_overlay(self, image: np.ndarray) -> np.ndarray:
        """Prepare a normalized RGB image for Grad-CAM overlay rendering."""
        image = self._ensure_rgb(image)
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        image = image.astype(np.float32)
        if image.max() > 1:
            image = image / 255.0
        return image

    def _ensure_rgb(self, image: np.ndarray) -> np.ndarray:
        """Normalize grayscale or RGBA inputs to RGB."""
        if image is None:
            raise ValueError("Image is None.")

        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.ndim == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Image must be HxWx3 or HxW, got {image.shape}.")

        return image

    def _auto_find_target_layer(self):
        """Select the last convolution layer as the default Grad-CAM target."""
        last_layer = None
        for module in self.model.modules():
            if isinstance(module, torch.nn.Conv2d):
                last_layer = module

        if last_layer is None:
            raise ValueError("No Conv2d layer found for Grad-CAM++ target layer.")

        return last_layer

    def save_heatmap(self, heatmap: np.ndarray, save_path: str) -> str:
        """Persist a Grad-CAM heatmap to disk and return the written path."""
        if heatmap is None:
            raise ValueError("Heatmap is None; nothing to save.")

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        hm = normalize_uint8_image(heatmap)

        # Ensure 3 channel RGB
        if hm.ndim == 2:
            hm = cv2.cvtColor(hm, cv2.COLOR_GRAY2RGB)

        if hm.ndim == 3 and hm.shape[2] == 3:
            hm_bgr = cv2.cvtColor(hm, cv2.COLOR_RGB2BGR)
        else:
            hm_bgr = hm

        if not cv2.imwrite(str(save_path), hm_bgr):
            raise IOError(f"Failed to write heatmap to {save_path}")

        return str(save_path)
