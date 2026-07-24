
"""Visualization helpers for images, metrics, and training artifacts."""

import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay, auc, confusion_matrix, roc_curve

from utils._shared import ensure_rgb


class VisualizationUtils:
    """Convenience helpers for rendering and saving visual outputs."""

    @staticmethod
    def show_image(
        image: np.ndarray,
        title: str = "Image",
    ) -> None:
        """Display an image using matplotlib."""
        if image is None:
            raise ValueError("Cannot display a None image.")

        plt.figure(figsize=(8, 8))
        plt.imshow(image)
        plt.title(title)
        plt.axis("off")
        plt.show()

    @staticmethod
    def plot_confusion_matrix(
        y_true,
        y_pred,
        output_path: str,
    ) -> None:
        """Save a confusion matrix figure to disk."""
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()

    @staticmethod
    def plot_roc_curve(
        y_true,
        y_prob,
        output_path: str,
    ) -> None:
        """Save a ROC curve figure to disk."""
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.4f}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()

    @staticmethod
    def save_heatmap(
        heatmap: np.ndarray,
        output_path: str,
    ) -> None:
        """Save a heatmap figure to disk."""
        if heatmap is None:
            raise ValueError("Heatmap must not be None.")

        plt.figure(figsize=(8, 8))
        plt.imshow(heatmap, cmap="jet")
        plt.axis("off")
        plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
        plt.close()

    @staticmethod
    def plot_training_curves(
        train_losses,
        val_losses,
    ) -> None:
        """Plot training and validation loss curves."""
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label="Train")
        plt.plot(val_losses, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.show()

