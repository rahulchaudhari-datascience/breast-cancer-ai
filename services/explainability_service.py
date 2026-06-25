from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image

from config import (
    DEVICE,
    IMAGE_SIZE,
    HEATMAP_OUTPUT_DIR,
)


class ExplainabilityService:
    """
    Grad-CAM++ explainability service (hardened).

    Returns None when explainability cannot be produced so calling pipelines
    can continue producing inference results.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        target_layer: Optional[torch.nn.Module] = None,
    ):
        self.device = DEVICE
        self.model = model.to(self.device)
        self.model.eval()

        self.target_layer = target_layer or self._auto_find_target_layer()

        HEATMAP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def generate(
        self,
        roi: np.ndarray,
        class_id: Optional[int] = None,
        save_path: Optional[str] = None,
    ) -> Optional[np.ndarray]:
        if roi is None:
            raise ValueError("ROI must be provided for explainability generation.")

        input_tensor = self._preprocess_roi(roi)

        try:
            cam = GradCAMPlusPlus(model=self.model, target_layers=[self.target_layer])
        except Exception as exc:
            print(f"[WARNING] Failed to initialize GradCAM++: {exc}")
            return None

        targets = None
        if class_id is not None:
            try:
                from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

                targets = [ClassifierOutputTarget(class_id)]
            except Exception:
                targets = None

        try:
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
        except Exception as exc:
            print(f"[WARNING] GradCAM++ generation failed: {exc}")
            return None

        try:
            rgb_roi = self._prepare_rgb_for_overlay(roi)
            heatmap = show_cam_on_image(rgb_roi, grayscale_cam, use_rgb=True)
        except Exception as exc:
            print(f"[WARNING] Failed to render heatmap overlay: {exc}")
            return None

        if save_path is None:
            save_path = str(HEATMAP_OUTPUT_DIR / "gradcam_output.png")

        try:
            self.save_heatmap(heatmap, save_path)
        except Exception as exc:
            print(f"[WARNING] Could not save heatmap: {exc}")

        return heatmap

    def _preprocess_roi(self, roi: np.ndarray) -> torch.Tensor:
        roi = self._ensure_rgb(roi)

        roi = cv2.resize(roi, (IMAGE_SIZE, IMAGE_SIZE))
        roi = roi.astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        roi = (roi - mean) / std

        tensor = torch.from_numpy(roi.transpose(2, 0, 1)).float()
        tensor = tensor.unsqueeze(0)

        return tensor.to(self.device)

    def _prepare_rgb_for_overlay(self, roi: np.ndarray) -> np.ndarray:
        roi = self._ensure_rgb(roi)
        roi = cv2.resize(roi, (IMAGE_SIZE, IMAGE_SIZE))
        roi = roi.astype(np.float32)
        if roi.max() > 1:
            roi = roi / 255.0
        return roi

    def _ensure_rgb(self, roi: np.ndarray) -> np.ndarray:
        if roi is None:
            raise ValueError("ROI image is None.")

        if roi.ndim == 2:
            roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)
        elif roi.ndim == 3 and roi.shape[2] == 4:
            roi = cv2.cvtColor(roi, cv2.COLOR_RGBA2RGB)

        if roi.ndim != 3 or roi.shape[2] != 3:
            raise ValueError(f"ROI must be HxWx3 or HxW, got {roi.shape}.")

        return roi

    def _auto_find_target_layer(self):
        last_layer = None
        for module in self.model.modules():
            if isinstance(module, torch.nn.Conv2d):
                last_layer = module

        if last_layer is None:
            raise ValueError("No Conv2d layer found for Grad-CAM++ target layer.")

        return last_layer

    def save_heatmap(self, heatmap: np.ndarray, save_path: str) -> str:
        if heatmap is None:
            raise ValueError("Heatmap is None; nothing to save.")

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        hm = heatmap
        # Convert float [0,1] to uint8 [0,255]
        if hm.dtype != np.uint8:
            try:
                hm = np.clip(hm, 0.0, 1.0)
                hm = (hm * 255).astype(np.uint8)
            except Exception:
                hm = np.clip(hm, 0, 255).astype(np.uint8)

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
