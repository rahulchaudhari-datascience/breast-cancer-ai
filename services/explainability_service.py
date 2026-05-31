# services/explainability_service.py

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
    Grad-CAM++ explainability service.

    Purpose:
        Shows which lesion region influenced the model prediction.

    Input:
        - Trained classification model
        - ROI image

    Output:
        - Grad-CAM++ heatmap overlay
    """

    def __init__(
        self,
        model: torch.nn.Module,
        target_layer: Optional[torch.nn.Module] = None,
    ):
        self.device = DEVICE
        self.model = model.to(self.device)
        self.model.eval()

        self.target_layer = (
            target_layer
            or self._auto_find_target_layer()
        )

    def generate(
        self,
        roi: np.ndarray,
        class_id: Optional[int] = None,
        save_path: Optional[str] = None,
    ) -> np.ndarray:

        input_tensor = self._preprocess_roi(roi)

        cam = GradCAMPlusPlus(
            model=self.model,
            target_layers=[self.target_layer],
        )

        targets = None

        if class_id is not None:
            from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

            targets = [
                ClassifierOutputTarget(class_id)
            ]

        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=targets,
        )[0]

        rgb_roi = self._prepare_rgb_for_overlay(roi)

        heatmap = show_cam_on_image(
            rgb_roi,
            grayscale_cam,
            use_rgb=True,
        )

        if save_path is None:
            save_path = str(
                HEATMAP_OUTPUT_DIR / "gradcam_output.png"
            )

        self.save_heatmap(
            heatmap,
            save_path,
        )

        return heatmap

    def _preprocess_roi(
        self,
        roi: np.ndarray,
    ) -> torch.Tensor:

        if roi is None:
            raise ValueError("ROI image is None.")

        if roi.ndim == 2:
            roi = cv2.cvtColor(
                roi,
                cv2.COLOR_GRAY2RGB,
            )

        roi = cv2.resize(
            roi,
            (IMAGE_SIZE, IMAGE_SIZE),
        )

        roi = roi.astype(np.float32) / 255.0

        mean = np.array(
            [0.485, 0.456, 0.406],
            dtype=np.float32,
        )

        std = np.array(
            [0.229, 0.224, 0.225],
            dtype=np.float32,
        )

        roi = (roi - mean) / std

        tensor = torch.from_numpy(
            roi.transpose(2, 0, 1)
        ).float()

        tensor = tensor.unsqueeze(0)

        return tensor.to(self.device)

    def _prepare_rgb_for_overlay(
        self,
        roi: np.ndarray,
    ) -> np.ndarray:

        if roi.ndim == 2:
            roi = cv2.cvtColor(
                roi,
                cv2.COLOR_GRAY2RGB,
            )

        roi = cv2.resize(
            roi,
            (IMAGE_SIZE, IMAGE_SIZE),
        )

        roi = roi.astype(np.float32)

        if roi.max() > 1:
            roi = roi / 255.0

        return roi

    def _auto_find_target_layer(self):

        """
        Finds the last convolution-like layer automatically.

        Works well for ConvNeXt / ResNet / EfficientNet style models.
        """

        last_layer = None

        for module in self.model.modules():

            if isinstance(
                module,
                torch.nn.Conv2d,
            ):
                last_layer = module

        if last_layer is None:
            raise ValueError(
                "No Conv2d layer found for Grad-CAM++ target layer."
            )

        return last_layer

    def save_heatmap(
        self,
        heatmap: np.ndarray,
        save_path: str,
    ) -> str:

        save_path = Path(save_path)

        save_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        heatmap_bgr = cv2.cvtColor(
            heatmap,
            cv2.COLOR_RGB2BGR,
        )

        cv2.imwrite(
            str(save_path),
            heatmap_bgr,
        )

        return str(save_path)


