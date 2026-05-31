from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch


class ConfidenceService:
    """
    Confidence and uncertainty estimation service.

    Used for:
    - Prediction confidence
    - Entropy-based uncertainty
    - Risk reliability level

    Input:
        probabilities from classifier or BI-RADS model

    Output:
        {
            "confidence": 94.2,
            "uncertainty": 0.18,
            "reliability": "High"
        }
    """

    def __init__(
        self,
        high_threshold: float = 85.0,
        medium_threshold: float = 65.0,
    ):
        self.high_threshold = high_threshold
        self.medium_threshold = medium_threshold

    def analyze(
        self,
        probabilities,
    ) -> Dict:

        probs = self._to_numpy(probabilities)

        if probs.ndim > 1:
            probs = probs.squeeze()

        confidence = float(np.max(probs) * 100)

        entropy = self.entropy(probs)

        uncertainty = self.normalized_entropy(probs)

        reliability = self.reliability_level(confidence)

        return {
            "confidence": confidence,
            "entropy": entropy,
            "uncertainty": uncertainty,
            "reliability": reliability,
        }

    def entropy(
        self,
        probabilities,
        eps: float = 1e-8,
    ) -> float:

        probs = self._to_numpy(probabilities)

        probs = np.clip(
            probs,
            eps,
            1.0,
        )

        return float(
            -np.sum(probs * np.log(probs))
        )

    def normalized_entropy(
        self,
        probabilities,
    ) -> float:

        probs = self._to_numpy(probabilities)

        num_classes = len(probs)

        if num_classes <= 1:
            return 0.0

        ent = self.entropy(probs)

        max_entropy = np.log(num_classes)

        return float(ent / max_entropy)

    def reliability_level(
        self,
        confidence: float,
    ) -> str:

        if confidence >= self.high_threshold:
            return "High"

        if confidence >= self.medium_threshold:
            return "Medium"

        return "Low"

    def combine_confidence(
        self,
        classification_confidence: float,
        birads_confidence: Optional[float] = None,
        segmentation_quality: Optional[float] = None,
    ) -> Dict:

        values = [classification_confidence]

        if birads_confidence is not None:
            values.append(birads_confidence)

        if segmentation_quality is not None:
            values.append(segmentation_quality)

        final_confidence = float(np.mean(values))

        return {
            "final_confidence": final_confidence,
            "reliability": self.reliability_level(final_confidence),
        }

    def calibration_warning(
        self,
        confidence: float,
    ) -> str:

        if confidence >= 90:
            return "High confidence prediction."

        if confidence >= 70:
            return "Moderate confidence prediction. Clinical review recommended."

        return "Low confidence prediction. Manual expert review strongly recommended."

    def _to_numpy(
        self,
        probabilities,
    ) -> np.ndarray:

        if isinstance(probabilities, torch.Tensor):
            probabilities = probabilities.detach().cpu().numpy()

        if isinstance(probabilities, dict):
            probabilities = list(probabilities.values())

        return np.array(
            probabilities,
            dtype=np.float32,
        )

