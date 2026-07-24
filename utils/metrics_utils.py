
"""Metric helpers used by training and evaluation pipelines."""

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
    """Collection of classification metric helpers."""

    @staticmethod
    def _as_numpy_array(values):
        """Convert sequence-like inputs into a NumPy array."""
        return np.asarray(values)

    @staticmethod
    def _safe_divide(numerator: float, denominator: float) -> float:
        """Return a guarded division result for metric calculations."""
        return float(numerator / denominator) if denominator > 0 else 0.0

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
        y_true_arr = MetricsUtils._as_numpy_array(y_true)
        y_pred_arr = MetricsUtils._as_numpy_array(y_pred)

        metrics["accuracy"] = float(accuracy_score(y_true_arr, y_pred_arr))
        metrics["precision"] = float(precision_score(y_true_arr, y_pred_arr, zero_division=0))
        metrics["recall"] = float(recall_score(y_true_arr, y_pred_arr, zero_division=0))
        metrics["sensitivity"] = metrics["recall"]
        metrics["f1_score"] = float(f1_score(y_true_arr, y_pred_arr, zero_division=0))

        cm = confusion_matrix(y_true_arr, y_pred_arr)
        metrics["confusion_matrix"] = cm.tolist()

        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics["specificity"] = MetricsUtils._safe_divide(tn, tn + fp)
            metrics["sensitivity"] = MetricsUtils._safe_divide(tp, tp + fn)
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




