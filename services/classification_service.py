# services/classification_service.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional
import logging

import numpy as np
import torch
import torch.nn as nn
from services.model_builder import build_classification_model
from services.preprocessing_service import PreprocessingService

from config import (
    DEVICE,
    NUM_CLASSES,
    CLASS_NAMES,
    EFFICIENTNET_CHECKPOINT,
    PRETRAINED,
    CLASSIFICATION_MODEL_NAME,
)

logger = logging.getLogger(__name__)


class ClassificationService:
    """EfficientNet-B0 based classifier wrapper.

    Provides single-image and batch prediction utilities returning softmax
    probabilities and a human-readable label.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        self.device = DEVICE
        self.model_name = model_name or CLASSIFICATION_MODEL_NAME
        self.checkpoint_path = checkpoint_path or str(EFFICIENTNET_CHECKPOINT)

        self.preprocessing = PreprocessingService()
        self.model = self._build_model()
        self.model.to(self.device)
        self._load_checkpoint()
        self.model.eval()

    def _build_model(self) -> nn.Module:
        # Use config.PRETRAINED by default; allow override in build call
        return build_classification_model(
            model_name=self.model_name,
            num_classes=NUM_CLASSES,
            pretrained=PRETRAINED,
        )

    def _load_checkpoint(self):
        path = Path(self.checkpoint_path)

        if not path.exists():
            logger.warning("Classification checkpoint not found at %s. Using model initialization (pretrained=%s).",
                           path, PRETRAINED)
            # If the model was created without ImageNet weights and a checkpoint
            # is missing, attempt to fall back to ImageNet weights for better
            # out-of-the-box performance.
            if not PRETRAINED:
                try:
                    logger.info("Rebuilding model with ImageNet pretrained weights as fallback.")
                    self.model = build_classification_model(
                        model_name=self.model_name,
                        num_classes=NUM_CLASSES,
                        pretrained=True,
                    ).to(self.device)
                except Exception:
                    logger.exception("Fallback to ImageNet pretrained model failed.")
            return

        try:
            checkpoint = torch.load(path, map_location=self.device)

            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state = checkpoint["model_state_dict"]
            elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                state = checkpoint["state_dict"]
            else:
                state = checkpoint

            self.model.load_state_dict(state)
            logger.info("Loaded classification checkpoint: %s", path)
        except Exception as exc:
            logger.exception("Failed to load classification checkpoint '%s': %s. Using initialized weights.", path, exc)

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


