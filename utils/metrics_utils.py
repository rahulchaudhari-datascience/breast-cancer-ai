
from typing import Dict

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


class MetricsUtils:

    @staticmethod
    def classification_metrics(
        y_true,
        y_pred,
        y_prob=None
    ) -> Dict:

        if len(y_true) == 0 or len(y_pred) == 0:
            raise ValueError("Empty y_true or y_pred passed to classification_metrics.")

        metrics = {}

        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
        metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
        metrics["f1_score"] = f1_score(y_true, y_pred, zero_division=0)

        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics["specificity"] = (tn / (tn + fp)) if (tn + fp) > 0 else 0.0
            metrics["sensitivity"] = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        else:
            metrics["specificity"] = 0.0
            metrics["sensitivity"] = 0.0

        if y_prob is not None:
            try:
                metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
            except ValueError:
                metrics["roc_auc"] = 0.0

        return metrics

    @staticmethod
    def dice_score(
        prediction,
        target,
        smooth: float = 1e-6
    ) -> float:

        prediction = prediction.flatten()
        target = target.flatten()

        intersection = np.sum(prediction * target)
        union = np.sum(prediction) + np.sum(target)

        if union == 0:
            return 1.0 if np.sum(target) == 0 else 0.0

        return float((2.0 * intersection + smooth) / (union + smooth))



