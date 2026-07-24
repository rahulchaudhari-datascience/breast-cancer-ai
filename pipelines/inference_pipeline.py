"""End-to-end inference pipeline for mammogram analysis."""

from __future__ import annotations

from typing import Dict, Optional
import logging

import cv2
import numpy as np
import torch

from services.preprocessing_service import PreprocessingService
from services.classification_service import ClassificationService
from services.confidence_service import ConfidenceService
from services.explainability_service import ExplainabilityService
from services.report_service import ReportService


logger = logging.getLogger(__name__)


class BreastCancerInferencePipeline:
    """Coordinate preprocessing, classification, explainability, and reporting."""

    def __init__(self):
        self.preprocessing_service = PreprocessingService()
        self.classification_service = ClassificationService()
        self.confidence_service = ConfidenceService()

        self.explainability_service = ExplainabilityService(
            model=self.classification_service.model
        )

        self.report_service = ReportService()

    def predict(
        self,
        image: np.ndarray,
        patient_id: str = "Demo Patient",
        generate_report: bool = True,
    ) -> Dict:

        if image is None:
            raise ValueError("Input image is None.")

        original_image = self._ensure_rgb(image)

        if original_image.size == 0:
            raise ValueError("Input image is empty.")

        try:
            processed_tensor = self.preprocessing_service.preprocess(
                original_image,
                training=False,
            )

            processed_image = self._tensor_to_rgb_image(
                processed_tensor,
            )

            classification_result = self.classification_service.predict(
                processed_image,
            )

            confidence_result = self.confidence_service.analyze(
                classification_result["probabilities"],
            )

            final_confidence = self.confidence_service.combine_confidence(
                classification_confidence=classification_result["confidence"],
            )
        except Exception as exc:
            logger.exception("Inference pipeline failed: %s", exc)
            return {
                "status": "error",
                "error": str(exc),
            }

        try:
            heatmap = self.explainability_service.generate(
                image=processed_image,
                class_id=classification_result["class_id"],
            )
        except Exception as exc:
            heatmap = None
            logger.warning("Explainability generation failed: %s", exc)

        report_path: Optional[str] = None

        if generate_report:
            try:
                report_path = self.report_service.generate(
                    prediction=classification_result,
                    confidence=final_confidence,
                    heatmap=heatmap,
                    patient_id=patient_id,
                )
            except Exception as exc:
                report_path = None
                logger.warning("Report generation failed: %s", exc)

        return {
            "original": original_image,
            "processed": processed_image,
            "prediction": classification_result["prediction"],
            "class_id": classification_result["class_id"],
            "probability": classification_result["probability"],
            "confidence": classification_result["confidence"],
            "probabilities": classification_result["probabilities"],
            "uncertainty": confidence_result["uncertainty"],
            "reliability": final_confidence["reliability"],
            "final_confidence": final_confidence["final_confidence"],
            "heatmap": heatmap,
            "report_path": report_path,
            "status": "success",
        }

    def predict_without_report(
        self,
        image: np.ndarray,
    ) -> Dict:

        return self.predict(
            image=image,
            generate_report=False,
        )

    def _ensure_rgb(
        self,
        image: np.ndarray,
    ) -> np.ndarray:

        if image is None:
            raise ValueError("Input image is None.")

        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        if image.ndim == 3 and image.shape[-1] == 4:
            return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"Unsupported image shape: {image.shape}. Expected HxWx3 or HxW.")

        return image

    def _tensor_to_rgb_image(
        self,
        tensor: torch.Tensor,
    ) -> np.ndarray:

        image = tensor.detach().cpu()

        if image.ndim == 4:
            image = image.squeeze(0)

        image = image.permute(1, 2, 0).numpy()

        image = image - image.min()
        image = image / (image.max() + 1e-8)

        image = (image * 255).astype(np.uint8)

        return image



