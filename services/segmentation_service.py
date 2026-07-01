from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from services.model_builder import build_segmentation_model

from config import (
    DEVICE,
    SEGMENTATION_THRESHOLD,
    UNETPP_CHECKPOINT,
    MASK_OUTPUT_DIR,
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
        checkpoint_path: Optional[str] = None,
        encoder_name: str = "resnet34",
        in_channels: int = 3,
        classes: int = 1,
    ):

        self.device = DEVICE
        self.checkpoint_path = (
            checkpoint_path
            or str(UNETPP_CHECKPOINT)
        )
        MASK_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.encoder_name = encoder_name
        self.in_channels = in_channels
        self.classes = classes

        self.model = self._build_model()
        self.load_model()

    # =====================================================
    # MODEL BUILDING
    # =====================================================

    def _build_model(self) -> torch.nn.Module:
        return build_segmentation_model(
            encoder_name=self.encoder_name,
            in_channels=self.in_channels,
            classes=self.classes,
            activation=None,
        )

    # =====================================================
    # MODEL LOADING
    # =====================================================

    def load_model(self):

        if Path(self.checkpoint_path).exists():
            try:
                checkpoint = torch.load(
                    self.checkpoint_path,
                    map_location=self.device,
                )

                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    self.model.load_state_dict(checkpoint["model_state_dict"])
                else:
                    self.model.load_state_dict(checkpoint)

                print(f"[INFO] Loaded segmentation checkpoint: {self.checkpoint_path}")
            except Exception as exc:
                print(
                    f"[WARNING] Failed to load segmentation checkpoint: {exc}. "
                    "Using untrained U-Net++ model."
                )
        else:
            print("[INFO] Segmentation checkpoint not found. Using untrained U-Net++ model.")

        self.model.to(self.device)
        self.model.eval()

    # =====================================================
    # INFERENCE
    # =====================================================

    @torch.no_grad()
    def segment(
        self,
        image_tensor,
    ) -> np.ndarray:

        """
        image_tensor:
            Torch tensor of shape [C,H,W] or [1,C,H,W].
        """

        if image_tensor is None:
            raise ValueError("Input tensor for segmentation is None.")

        if not isinstance(image_tensor, torch.Tensor):
            image_tensor = torch.tensor(
                image_tensor,
                dtype=torch.float32,
            )

        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)

        if image_tensor.dim() != 4:
            raise ValueError(
                f"Segmentation input must be 4D tensor, got shape {image_tensor.shape}."
            )

        image_tensor = image_tensor.to(self.device)

        try:
            logits = self.model(image_tensor)
        except Exception as exc:
            print(f"[WARNING] Segmentation model inference failed: {exc}. Falling back to Otsu.")
            return self.fallback_mask(image_tensor)

        probs = torch.sigmoid(logits)

        mask = (probs > SEGMENTATION_THRESHOLD).float()
        mask = mask.squeeze().cpu().numpy()

        if mask.ndim == 3:
            mask = mask[0]

        mask = (mask * 255).astype(np.uint8)
        mask = self.postprocess(mask)
        self._save_mask(mask)
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

    def _save_mask(self, mask: np.ndarray, filename: Optional[str] = None) -> Path:
        save_path = MASK_OUTPUT_DIR / (filename or "segmentation_mask.png")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), mask)
        return save_path

    # =====================================================
    # FALLBACK
    # =====================================================

    def fallback_mask(
        self,
        image_tensor,
    ) -> np.ndarray:

        """
        Used before training. Creates a rough tumor mask from the input tensor.
        """

        image = image_tensor.cpu().numpy()

        if image.ndim == 4:
            image = image.squeeze(0)

        if image.ndim == 3 and image.shape[0] in (1, 3):
            image = np.transpose(image, (1, 2, 0))

        if image.ndim not in (2, 3):
            raise ValueError(
                f"Fallback mask expects 2D or HxWxC image, got shape {image.shape}."
            )

        image = image.astype(np.float32)
        if image.max() <= 1.0:
            image = image * 255.0

        image = cv2.normalize(
            image,
            None,
            0,
            255,
            cv2.NORM_MINMAX,
        ).astype(np.uint8)

        if image.ndim == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        _, mask = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )

        return self.postprocess(mask)

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

    def _save_mask(self, mask: np.ndarray, filename: Optional[str] = None) -> Path:
        save_path = MASK_OUTPUT_DIR / (filename or "segmentation_mask.png")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), mask)
        return save_path

