
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay
)


class VisualizationUtils:

    @staticmethod
    def show_image(
        image,
        title="Image"
    ):

        plt.figure(
            figsize=(8, 8)
        )

        plt.imshow(image)

        plt.title(title)

        plt.axis("off")

        plt.show()

    @staticmethod
    def overlay_mask(
        image,
        mask,
        alpha=0.4
    ):

        if len(mask.shape) == 2:

            colored_mask = np.zeros_like(
                image
            )

            colored_mask[:, :, 0] = (
                mask * 255
            )

        else:

            colored_mask = mask

        return cv2.addWeighted(
            image,
            1 - alpha,
            colored_mask,
            alpha,
            0
        )

    @staticmethod
    def plot_confusion_matrix(
        y_true,
        y_pred
    ):

        cm = confusion_matrix(
            y_true,
            y_pred
        )

        disp = (
            ConfusionMatrixDisplay(
                confusion_matrix=cm
            )
        )

        disp.plot()

        plt.show()

    @staticmethod
    def save_heatmap(
        heatmap,
        output_path
    ):

        plt.figure(
            figsize=(8, 8)
        )

        plt.imshow(
            heatmap,
            cmap="jet"
        )

        plt.axis("off")

        plt.savefig(
            output_path,
            bbox_inches="tight",
            pad_inches=0
        )

        plt.close()

    @staticmethod
    def plot_training_curves(
        train_losses,
        val_losses
    ):

        plt.figure(
            figsize=(10, 5)
        )

        plt.plot(
            train_losses,
            label="Train"
        )

        plt.plot(
            val_losses,
            label="Validation"
        )

        plt.xlabel("Epoch")

        plt.ylabel("Loss")

        plt.legend()

        plt.grid(True)

        plt.show()

