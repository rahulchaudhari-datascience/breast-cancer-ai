# services/birads_service.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import timm

from config import (
    DEVICE,
    IMAGE_SIZE,
    BIRADS_CLASSES,
    BIRADS_LABELS,
    CHECKPOINT_DIR,
)


class BIRADSService:
    """
    BI-RADS prediction service.

    BI-RADS labels used:
        0 -> BI-RADS 2
        1 -> BI-RADS 3
        2 -> BI-RADS 4
        3 -> BI-RADS 5
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        model_name: str = "convnextv2_tiny.fcmae_ft_in22k_in1k",
    ):
        self.device = DEVICE
        self.model_name = model_name
        self.checkpoint_path = (
            checkpoint_path
            or str(CHECKPOINT_DIR / "birads_convnextv2_best.pth")
        )

        self.model = self._build_model()
        self._load_checkpoint()

        self.model.to(self.device)
        self.model.eval()

    def _build_model(self) -> nn.Module:
        return timm.create_model(
            self.model_name,
            pretrained=True,
            num_classes=BIRADS_CLASSES,
        )

    def _load_checkpoint(self):
        path = Path(self.checkpoint_path)

        if not path.exists():
            print("[INFO] BI-RADS checkpoint not found. Using pretrained base model.")
            return

        checkpoint = torch.load(path, map_location=self.device)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)

        print(f"[INFO] Loaded BI-RADS checkpoint: {path}")

    def preprocess_roi(
        self,
        roi: np.ndarray,
    ) -> torch.Tensor:

        if roi is None:
            raise ValueError("ROI image is None.")

        if roi.ndim == 2:
            roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)

        roi = cv2.resize(
            roi,
            (IMAGE_SIZE, IMAGE_SIZE),
        )

        roi = roi.astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        roi = (roi - mean) / std

        tensor = torch.from_numpy(
            roi.transpose(2, 0, 1)
        ).float()

        return tensor.unsqueeze(0)

    @torch.no_grad()
    def predict(
        self,
        roi: np.ndarray,
    ) -> Dict:

        tensor = self.preprocess_roi(roi).to(self.device)

        logits = self.model(tensor)

        probs = torch.softmax(logits, dim=1)

        prob_values = probs.squeeze(0).cpu().numpy()

        birads_id = int(np.argmax(prob_values))
        confidence = float(prob_values[birads_id])

        probabilities = {
            BIRADS_LABELS[i]: float(prob_values[i])
            for i in range(BIRADS_CLASSES)
        }

        return {
            "birads_id": birads_id,
            "birads": BIRADS_LABELS[birads_id],
            "confidence": confidence * 100,
            "probabilities": probabilities,
            "logits": logits.detach().cpu(),
        }


