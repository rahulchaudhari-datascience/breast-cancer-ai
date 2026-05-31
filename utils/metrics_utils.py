
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

        metrics = {}

        metrics["accuracy"] = accuracy_score(
            y_true,
            y_pred
        )

        metrics["precision"] = precision_score(
            y_true,
            y_pred,
            zero_division=0
        )

        metrics["recall"] = recall_score(
            y_true,
            y_pred,
            zero_division=0
        )

        metrics["f1_score"] = f1_score(
            y_true,
            y_pred,
            zero_division=0
        )

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            y_pred
        ).ravel()

        metrics["specificity"] = (
            tn / (tn + fp)
        ) if (tn + fp) > 0 else 0

        metrics["sensitivity"] = (
            tp / (tp + fn)
        ) if (tp + fn) > 0 else 0

        if y_prob is not None:

            metrics["roc_auc"] = roc_auc_score(
                y_true,
                y_prob
            )

        return metrics

    @staticmethod
    def dice_score(
        prediction,
        target,
        smooth=1e-6
    ):

        prediction = prediction.flatten()



