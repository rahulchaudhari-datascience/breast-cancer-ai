
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay, auc, confusion_matrix, roc_curve


class VisualizationUtils:

    @staticmethod
    def show_image(
        image: np.ndarray,
        title: str = "Image",
    ) -> None:
        if image is None:
            raise ValueError("Cannot display a None image.")

        plt.figure(figsize=(8, 8))
        plt.imshow(image)
        plt.title(title)
        plt.axis("off")
        plt.show()

    @staticmethod
    def overlay_mask(
        image: np.ndarray,
        mask: np.ndarray,
        alpha: float = 0.4,
    ) -> np.ndarray:
        if image is None or mask is None:
            raise ValueError("Image and mask must not be None.")

        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        mask_arr = mask.astype(np.float32)
        if mask_arr.max() <= 1.0:
            mask_arr *= 255.0

        mask_arr = np.clip(mask_arr, 0, 255).astype(np.uint8)

        colored_mask = np.zeros_like(image)
        colored_mask[:, :, 0] = mask_arr

        return cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)

    @staticmethod
    def plot_confusion_matrix(
        y_true,
        y_pred,
        output_path: str,
    ) -> None:
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
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label="Train")
        plt.plot(val_losses, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.show()

