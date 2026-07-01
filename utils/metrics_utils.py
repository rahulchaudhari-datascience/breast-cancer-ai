
from typing import Dict, Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


class MetricsUtils:
    """Collection of classification and segmentation metric helpers."""

    @staticmethod
    def classification_metrics(
        y_true, y_pred, y_prob=None
    ) -> Dict[str, Any]:
        """Compute common binary classification metrics.

        Returns a dict with accuracy, precision, recall, f1, specificity,
        sensitivity, roc_auc (if probabilities provided), and confusion matrix.
        """
        if len(y_true) == 0 or len(y_pred) == 0:
            raise ValueError("Empty y_true or y_pred passed to classification_metrics.")

        metrics: Dict[str, Any] = {}
        y_true_arr = np.asarray(y_true)
        y_pred_arr = np.asarray(y_pred)

        metrics["accuracy"] = float(accuracy_score(y_true_arr, y_pred_arr))
        metrics["precision"] = float(precision_score(y_true_arr, y_pred_arr, zero_division=0))
        metrics["recall"] = float(recall_score(y_true_arr, y_pred_arr, zero_division=0))
        metrics["sensitivity"] = metrics["recall"]
        metrics["f1_score"] = float(f1_score(y_true_arr, y_pred_arr, zero_division=0))

        cm = confusion_matrix(y_true_arr, y_pred_arr)
        metrics["confusion_matrix"] = cm.tolist()

        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics["specificity"] = float((tn / (tn + fp)) if (tn + fp) > 0 else 0.0)
            metrics["sensitivity"] = float((tp / (tp + fn)) if (tp + fn) > 0 else 0.0)
        else:
            metrics["specificity"] = 0.0
            metrics["sensitivity"] = 0.0

        if y_prob is not None:
            try:
                metrics["roc_auc"] = float(roc_auc_score(y_true_arr, y_prob))
            except Exception:
                metrics["roc_auc"] = 0.0

        try:
            metrics["classification_report"] = classification_report(y_true_arr, y_pred_arr, output_dict=True)
        except Exception:
            metrics["classification_report"] = {}

        return metrics

    @staticmethod
    def dice_score(prediction, target, smooth: float = 1e-6) -> float:
        """Compute Dice coefficient for binary masks.

        Both inputs should be binary arrays (0/1) or probabilities thresholded prior to use.
        """
        pred = np.asarray(prediction).astype(np.float32).flatten()
        tgt = np.asarray(target).astype(np.float32).flatten()

        intersection = np.sum(pred * tgt)
        union = np.sum(pred) + np.sum(tgt)

        if union == 0:
            return 1.0 if np.sum(tgt) == 0 else 0.0

        return float((2.0 * intersection + smooth) / (union + smooth))

    @staticmethod
    def iou_score(prediction, target, smooth: float = 1e-6) -> float:
        """Compute Intersection over Union (IoU) for binary masks."""
        pred = np.asarray(prediction).astype(np.float32).flatten()
        tgt = np.asarray(target).astype(np.float32).flatten()

        intersection = np.sum(pred * tgt)
        union = np.sum(pred) + np.sum(tgt) - intersection

        if union == 0:
            return 1.0 if np.sum(tgt) == 0 else 0.0

        return float((intersection + smooth) / (union + smooth))



