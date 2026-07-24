# services/classification_service.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from services.model_builder import build_classification_model
from services._shared import get_service_logger, load_state_dict_into_model
from services.preprocessing_service import PreprocessingService

from config import (
    DEVICE,
    NUM_CLASSES,
    CLASS_NAMES,
    EFFICIENTNET_CHECKPOINT,
    PRETRAINED,
    CLASSIFICATION_MODEL_NAME,
)

logger = get_service_logger(__name__)


class ClassificationService:
    """EfficientNet-B0-based classifier wrapper.

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
        """Build the classifier using the configured backbone and class count."""
        return build_classification_model(
            model_name=self.model_name,
            num_classes=NUM_CLASSES,
            pretrained=PRETRAINED,
        )

    def _load_checkpoint(self):
        path = Path(self.checkpoint_path)

        if not path.exists():
            logger.warning(
                "Classification checkpoint not found at %s. Using model initialization (pretrained=%s).",
                path,
                PRETRAINED,
            )
            # Preserve the existing fallback behavior for environments that
            # rely on ImageNet-initialized weights when no fine-tuned checkpoint is available.
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
            load_state_dict_into_model(self.model, path, map_location=self.device)
            logger.info("Loaded classification checkpoint: %s", path)
        except Exception as exc:
            logger.exception(
                "Failed to load classification checkpoint '%s': %s. Using initialized weights.",
                path,
                exc,
            )

    def preprocess_image(
        self,
        image: np.ndarray,
    ) -> torch.Tensor:
        """Apply the shared preprocessing pipeline to a mammogram image."""
        return self.preprocessing.preprocess_for_model(image)

    @torch.no_grad()
    def predict(
        self,
        image: np.ndarray,
    ) -> Dict:

        tensor = self.preprocess_image(image).to(self.device)

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
        images: list[np.ndarray],
    ) -> list[Dict]:
        """Run prediction on a batch of mammogram images."""
        return [self.predict(image) for image in images]


