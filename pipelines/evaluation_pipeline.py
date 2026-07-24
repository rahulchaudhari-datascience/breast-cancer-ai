"""Evaluation pipeline for the trained classification model."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional
import logging

import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from services.model_builder import build_classification_model

from config import (
    DEVICE,
    BATCH_SIZE,
    NUM_WORKERS,
    NUM_CLASSES,
    CLASS_NAMES,
    CLASSIFICATION_CHECKPOINT,
    METRICS_OUTPUT_DIR,
    PREDICTIONS_OUTPUT_DIR,
)

from services.preprocessing_service import PreprocessingService
from utils.metrics_utils import MetricsUtils
from utils.visualization_utils import VisualizationUtils


logger = logging.getLogger(__name__)


class MammogramEvaluationDataset(Dataset):
    """
    CSV format:

    image_path,label

    label:
        0 = Benign
        1 = Malignant
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
    ):
        self.dataframe = dataframe.reset_index(drop=True)
        self.preprocessing = PreprocessingService()

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(
        self,
        index: int,
    ):
        row = self.dataframe.iloc[index]

        image_path = row["image_path"]
        label = int(row["label"])

        image = cv2.imread(image_path)

        if image is None:
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        image_tensor = self.preprocessing.preprocess(
            image,
            training=False,
        )

        return (
            image_tensor,
            torch.tensor(label, dtype=torch.long),
            image_path,
        )


class EvaluationPipeline:
    """
    Evaluation pipeline for trained classification models.

    Outputs:
        - Metrics dictionary
        - Predictions CSV
        - Metrics CSV
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        model_name: str = "convnextv2_tiny.fcmae_ft_in22k_in1k",
    ):
        self.device = DEVICE
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path or str(CLASSIFICATION_CHECKPOINT)

        self.model = self._build_model()
        self._load_checkpoint()

        self.model.to(self.device)
        self.model.eval()

        self.pin_memory = torch.cuda.is_available()

        METRICS_OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        PREDICTIONS_OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

    def _build_model(self):
        return build_classification_model(
            model_name=self.model_name,
            num_classes=NUM_CLASSES,
            pretrained=False,
        )

    def _load_checkpoint(self):
        path = Path(self.checkpoint_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {path}"
            )

        checkpoint = torch.load(
            path,
            map_location=self.device,
        )

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.model.load_state_dict(
                checkpoint["model_state_dict"]
            )
        else:
            self.model.load_state_dict(
                checkpoint
            )

        logger.info("Loaded checkpoint: %s", path)

    def create_loader(
        self,
        test_csv: str,
    ) -> DataLoader:

        csv_path = Path(test_csv)

        if not csv_path.exists():
            raise FileNotFoundError(f"Test CSV not found: {csv_path}")

        test_df = pd.read_csv(csv_path)

        if "image_path" not in test_df.columns or "label" not in test_df.columns:
            raise ValueError("Test CSV must contain 'image_path' and 'label' columns.")

        if len(test_df) == 0:
            raise ValueError("Test dataset is empty. Check test CSV and image paths.")

        dataset = MammogramEvaluationDataset(test_df)

        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=self.pin_memory,
        )

        return loader

    @torch.no_grad()
    def evaluate(
        self,
        test_csv: str,
        save_prefix: str = "convnextv2_test",
    ) -> Dict:

        loader = self.create_loader(
            test_csv
        )

        y_true = []
        y_pred = []
        y_prob = []
        image_paths = []

        for images, labels, paths in tqdm(
            loader,
            desc="Evaluating",
        ):

            images = images.to(self.device)

            logits = self.model(images)

            probs = torch.softmax(
                logits,
                dim=1,
            )

            preds = torch.argmax(
                probs,
                dim=1,
            )

            y_true.extend(
                labels.cpu().numpy().tolist()
            )

            y_pred.extend(
                preds.cpu().numpy().tolist()
            )

            y_prob.extend(
                probs[:, 1].cpu().numpy().tolist()
            )

            image_paths.extend(
                list(paths)
            )

        metrics = MetricsUtils.classification_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
        )

        predictions_df = pd.DataFrame(
            {
                "image_path": image_paths,
                "true_label": y_true,
                "predicted_label": y_pred,
                "true_class": [
                    CLASS_NAMES[i] for i in y_true
                ],
                "predicted_class": [
                    CLASS_NAMES[i] for i in y_pred
                ],
                "malignant_probability": y_prob,
            }
        )

        predictions_path = (
            PREDICTIONS_OUTPUT_DIR /
            f"{save_prefix}_predictions.csv"
        )

        metrics_path = (
            METRICS_OUTPUT_DIR /
            f"{save_prefix}_metrics.csv"
        )

        predictions_df.to_csv(
            predictions_path,
            index=False,
        )

        pd.DataFrame(
            [metrics]
        ).to_csv(
            metrics_path,
            index=False,
        )

        cm_path = METRICS_OUTPUT_DIR / f"{save_prefix}_confusion_matrix.png"
        roc_path = METRICS_OUTPUT_DIR / f"{save_prefix}_roc_curve.png"

        try:
            VisualizationUtils.plot_confusion_matrix(y_true, y_pred, str(cm_path))
            logger.info("Confusion matrix saved to: %s", cm_path)
        except Exception as exc:
            logger.warning("Could not save confusion matrix: %s", exc)

        try:
            VisualizationUtils.plot_roc_curve(y_true, y_prob, str(roc_path))
            logger.info("ROC curve saved to: %s", roc_path)
        except Exception as exc:
            logger.warning("Could not save ROC curve: %s", exc)

        logger.info("Evaluation Metrics")
        logger.info("%s", metrics)

        logger.info("Predictions saved to: %s", predictions_path)
        logger.info("Metrics saved to: %s", metrics_path)

        return {
            "metrics": metrics,
            "predictions_path": str(predictions_path),
            "metrics_path": str(metrics_path),
            "confusion_matrix_path": str(cm_path),
            "roc_curve_path": str(roc_path),
        }


if __name__ == "__main__":

    pipeline = EvaluationPipeline()

    results = pipeline.evaluate(
        test_csv="datasets/annotations/test.csv",
    )

    print(results)



