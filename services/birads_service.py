# services/birads_service.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from services.model_builder import build_classification_model
from services.preprocessing_service import PreprocessingService

from config import (
    DEVICE,
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

        self.preprocessing = PreprocessingService()
        self.model = self._build_model()
        self.model.to(self.device)
        self._load_checkpoint()
        self.model.eval()

    def _build_model(self) -> nn.Module:
        return build_classification_model(
            model_name=self.model_name,
            num_classes=BIRADS_CLASSES,
            pretrained=False,
        )

    def _load_checkpoint(self):
        path = Path(self.checkpoint_path)

        if not path.exists():
            print("[INFO] BI-RADS checkpoint not found. Using untrained ConvNeXt model.")
            return

        try:
            checkpoint = torch.load(path, map_location=self.device)

            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)

            print(f"[INFO] Loaded BI-RADS checkpoint: {path}")
        except Exception as exc:
            print(f"[WARNING] Failed to load BI-RADS checkpoint: {exc}. Using model weights as initialized.")

    def preprocess_roi(
        self,
        roi: np.ndarray,
    ) -> torch.Tensor:
        return self.preprocessing.preprocess_for_model(roi)

    @torch.no_grad()
    def predict(
        self,
        roi: np.ndarray,
    ) -> Dict:

        tensor = self.preprocess_roi(roi).to(self.device)

        logits = self.model(tensor)

        if logits.ndim == 1:
            logits = logits.unsqueeze(0)

        probs = torch.softmax(logits, dim=1)

        prob_values = probs.squeeze(0).cpu().numpy()

        birads_id = int(np.argmax(prob_values))
        confidence = float(prob_values[birads_id])

        probabilities = {
            BIRADS_LABELS.get(i, f"BI-RADS {i}"): float(prob_values[i])
            for i in range(len(prob_values))
        }

        return {
            "birads_id": birads_id,
            "birads": BIRADS_LABELS.get(birads_id, f"BI-RADS {birads_id}"),
            "confidence": confidence * 100,
            "probabilities": probabilities,
            "logits": logits.detach().cpu(),
        }


