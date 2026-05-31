from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from config import (
    DEVICE,
    SEGMENTATION_THRESHOLD,
    NNUNET_CHECKPOINT
)


class SegmentationService:
    """
    Research-grade segmentation service.

    Responsibilities:
    -----------------
    1. Load segmentation model
    2. Run inference
    3. Create binary mask
    4. Post-process mask
    5. Return clean tumor mask

    Output Contract:
    ----------------
    mask : np.ndarray
        uint8 mask
        shape(H,W)
        values = 0 or 255
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None
    ):

        self.device = DEVICE

        self.checkpoint_path = (
            checkpoint_path
            or str(NNUNET_CHECKPOINT)
        )

        self.model = None

        self.load_model()

    # =====================================================
    # MODEL LOADING
    # =====================================================

    def load_model(self):

        """
        Replace this with actual nnU-Net loading
        once model training is completed.
        """

        if Path(
            self.checkpoint_path
        ).exists():

            self.model = torch.jit.load(
                self.checkpoint_path,
                map_location=self.device
            )

            self.model.eval()

        else:

            self.model = None

            print(
                "[INFO] Segmentation model not found."
            )

    # =====================================================
    # INFERENCE
    # =====================================================

    @torch.no_grad()
    def segment(
        self,
        image_tensor
    ) -> np.ndarray:

        """
        image_tensor:
        Shape:
            [C,H,W]
            or
            [1,C,H,W]
        """

        if self.model is None:

            return self.fallback_mask(
                image_tensor
            )

        if image_tensor.dim() == 3:

            image_tensor = (
                image_tensor
                .unsqueeze(0)
            )

        image_tensor = (
            image_tensor
            .to(self.device)
        )

        logits = self.model(
            image_tensor
        )

        probs = torch.sigmoid(
            logits
        )

        mask = (
            probs >
            SEGMENTATION_THRESHOLD
        ).float()

        mask = (
            mask
            .squeeze()
            .cpu()
            .numpy()
        )

        mask = (
            mask * 255
        ).astype(np.uint8)

        mask = self.postprocess(
            mask
        )

        return mask

    # =====================================================
    # POSTPROCESSING
    # =====================================================

    def postprocess(
        self,
        mask: np.ndarray
    ) -> np.ndarray:

        kernel = np.ones(
            (5, 5),
            np.uint8
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel
        )

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) == 0:

            return mask

        largest = max(
            contours,
            key=cv2.contourArea
        )

        clean_mask = np.zeros_like(
            mask
        )

        cv2.drawContours(
            clean_mask,
            [largest],
            -1,
            255,
            thickness=-1
        )

        return clean_mask

    # =====================================================
    # FALLBACK
    # =====================================================

    def fallback_mask(
        self,
        image_tensor
    ) -> np.ndarray:

        """
        Used before training.

        Creates a rough ROI mask.
        """

        image = (
            image_tensor
            .cpu()
            .numpy()
        )

        if image.ndim == 3:

            image = np.transpose(
                image,
                (1, 2, 0)
            )

        image = cv2.normalize(
            image,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        )

        image = image.astype(
            np.uint8
        )

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2GRAY
        )

        _, mask = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY +
            cv2.THRESH_OTSU
        )

        return self.postprocess(
            mask
        )

    # =====================================================
    # VISUALIZATION
    # =====================================================

    def overlay_mask(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        alpha: float = 0.4
    ) -> np.ndarray:

        colored = np.zeros_like(
            image
        )

        colored[:, :, 0] = mask

        overlay = cv2.addWeighted(
            image,
            1 - alpha,
            colored,
            alpha,
            0
        )

        return overlay

