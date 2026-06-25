
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


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
    ) -> None:
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.show()

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

