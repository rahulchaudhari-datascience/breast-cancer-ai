# services/classification_service.py

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
    NUM_CLASSES,
    CLASS_NAMES,
    CONVNEXT_CHECKPOINT,
)


class ClassificationService:
    """
    ConvNeXt-V2 Tiny based classifier.

    Input:
        ROI image: numpy RGB image

    Output:
        {
            "prediction": "Benign" / "Malignant",
            "class_id": int,
            "probability": float,
            "probabilities": dict,
            "logits": tensor
        }
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        model_name: str = "convnextv2_tiny.fcmae_ft_in22k_in1k",
    ):
        self.device = DEVICE
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path or str(CONVNEXT_CHECKPOINT)

        self.preprocessing = PreprocessingService()
        self.model = self._build_model()
        self.model.to(self.device)
        self._load_checkpoint()
        self.model.eval()

    def _build_model(self) -> nn.Module:
        return build_classification_model(
            model_name=self.model_name,
            num_classes=NUM_CLASSES,
            pretrained=False,
        )

    def _load_checkpoint(self):
        path = Path(self.checkpoint_path)

        if not path.exists():
            print("[INFO] Classification checkpoint not found. Using untrained ConvNeXt model.")
            return

        try:
            checkpoint = torch.load(
                path,
                map_location=self.device,
            )

            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)

            print(f"[INFO] Loaded classification checkpoint: {path}")
        except Exception as exc:
            print(f"[WARNING] Failed to load classification checkpoint: {exc}. Using model weights as initialized.")

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

        class_id = int(np.argmax(prob_values))
        probability = float(prob_values[class_id])

        probabilities = {
            CLASS_NAMES.get(i, f"Class_{i}"): float(prob_values[i])
            for i in range(len(prob_values))
        }

        return {
            "prediction": CLASS_NAMES.get(class_id, f"Class_{class_id}"),
            "class_id": class_id,
            "probability": probability,
            "confidence": probability * 100,
            "probabilities": probabilities,
            "logits": logits.detach().cpu(),
        }

    def predict_batch(
        self,
        rois: list[np.ndarray],
    ) -> list[Dict]:

        results = []

        for roi in rois:
            results.append(self.predict(roi))

        return results


