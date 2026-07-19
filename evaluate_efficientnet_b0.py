"""Evaluate an EfficientNet-B0 classifier and write the same reports and plots.

The script preserves the exact evaluation logic, predictions, metrics, and output
file names while improving organization, comments, logging, and readability.
"""

import argparse
import logging
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

ROOT_DIR = Path(__file__).resolve().parent

logging.basicConfig(level=logging.INFO, format="%(message)s")
LOGGER = logging.getLogger(__name__)


class CBISDataset(Dataset):
    """Dataset wrapper for evaluation images and labels from a CSV manifest."""

    def __init__(self, csv_file: str, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.label_map = {
            "BENIGN": 0,
            "MALIGNANT": 1,
            "BENIGN_WITHOUT_CALLBACK": 0,
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = row["image_path"]
        label = self._normalize_label(row["label"])

        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB")
        except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
            raise ValueError(f"Failed to load image '{image_path}'") from exc

        if self.transform is not None:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long), image_path

    def _normalize_label(self, label):
        if isinstance(label, str):
            normalized = label.strip().upper()
            return self.label_map.get(normalized, int(normalized))
        return int(label)


def get_transforms():
    """Return the evaluation image transformations used by the script."""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    val_tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])
    return val_tfms


def build_loader(csv_file: str, batch_size: int = 16, num_workers: int = 4):
    """Create a DataLoader for evaluation data without changing the evaluation behavior."""
    val_tfms = get_transforms()
    dataset = CBISDataset(csv_file, transform=val_tfms)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return loader


def build_model(checkpoint_path: str, device: torch.device) -> nn.Module:
    """Load the EfficientNet-B0 checkpoint and place it on the requested device."""
    model = models.efficientnet_b0(pretrained=False)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model = model.to(device)

    if not Path(checkpoint_path).is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model


def evaluate_confusion_matrix(model, loader, device, output_dir: Path):
    """Run inference, save the confusion matrix figure, and return evaluation arrays."""
    y_true = []
    y_pred = []
    y_score = []
    image_paths = []

    model.eval()
    with torch.no_grad():
        for images, labels, paths in loader:
            images = images.to(device, non_blocking=torch.cuda.is_available())
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)
            y_true.extend(labels.detach().cpu().numpy())
            y_pred.extend(preds.detach().cpu().numpy())
            y_score.extend(probs.detach().cpu().numpy())
            image_paths.extend([str(path) for path in paths])

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    report = classification_report(y_true, y_pred, target_names=["Benign", "Malignant"], digits=4)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Benign", "Malignant"])
    ax.set_yticklabels(["Benign", "Malignant"])
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            ax.text(col, row, int(cm[row, col]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    confusion_path = output_dir / "confusion_matrix.png"
    fig.savefig(confusion_path, bbox_inches="tight")
    plt.close(fig)

    LOGGER.info("Confusion Matrix:\n%s", report)
    LOGGER.info("Saved confusion matrix to: %s", confusion_path)
    return y_true, y_pred, y_score, image_paths


def plot_roc_curve(y_true, y_score, output_dir: Path):
    """Save the ROC curve figure and return the computed ROC-AUC value."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.grid(True)
    roc_path = output_dir / "roc_curve.png"
    fig.savefig(roc_path, bbox_inches="tight")
    plt.close(fig)

    LOGGER.info("Saved ROC curve to: %s", roc_path)
    return roc_auc


def save_prediction_outputs(image_paths, y_true, y_pred, y_score, output_dir: Path):
    """Write prediction, false-positive, and false-negative CSV outputs."""
    predictions_df = pd.DataFrame({
        "image_path": image_paths,
        "true_label": y_true,
        "predicted_label": y_pred,
        "probability_benign": 1 - np.asarray(y_score),
        "probability_malignant": np.asarray(y_score),
    })
    predictions_path = output_dir / "predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)

    false_positives_df = predictions_df[(predictions_df["true_label"] == 0) & (predictions_df["predicted_label"] == 1)]
    false_negatives_df = predictions_df[(predictions_df["true_label"] == 1) & (predictions_df["predicted_label"] == 0)]

    false_positives_path = output_dir / "false_positives.csv"
    false_negatives_path = output_dir / "false_negatives.csv"
    false_positives_df.to_csv(false_positives_path, index=False)
    false_negatives_df.to_csv(false_negatives_path, index=False)

    LOGGER.info("Saved predictions to: %s", predictions_path)
    LOGGER.info("Saved false positives to: %s", false_positives_path)
    LOGGER.info("Saved false negatives to: %s", false_negatives_path)
    return predictions_df, false_positives_df, false_negatives_df


def _register_hooks(target_layer, features, gradients):
    """Register forward and backward hooks for Grad-CAM feature capture."""

    def forward_hook(module, input, output):
        features.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    handle_fwd = target_layer.register_forward_hook(forward_hook)
    try:
        handle_bwd = target_layer.register_full_backward_hook(backward_hook)
    except AttributeError:
        handle_bwd = target_layer.register_backward_hook(backward_hook)

    return handle_fwd, handle_bwd


def generate_gradcam(model, image_tensor, device):
    """Generate a Grad-CAM heatmap using the existing model architecture."""
    model.eval()
    features = []
    gradients = []
    target_layer = model.features[-1]
    handle_fwd, handle_bwd = _register_hooks(target_layer, features, gradients)

    image_tensor = image_tensor.unsqueeze(0).to(device)
    output = model(image_tensor)
    pred_class = output.argmax(dim=1).item()

    model.zero_grad()
    output[0, pred_class].backward(retain_graph=True)

    if not gradients or not features:
        handle_fwd.remove()
        handle_bwd.remove()
        raise RuntimeError("Grad-CAM hooks did not capture features or gradients.")

    pooled_grad = torch.mean(gradients[0], dim=[0, 2, 3])
    feature_map = features[0][0]

    for i in range(pooled_grad.shape[0]):
        feature_map[i] *= pooled_grad[i]

    heatmap = torch.mean(feature_map, dim=0).cpu().detach().numpy()
    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() > 0:
        heatmap /= heatmap.max()

    handle_fwd.remove()
    handle_bwd.remove()
    return heatmap, pred_class


def save_gradcam(heatmap, image_path: Path, output_path: Path):
    """Save a Grad-CAM overlay image to disk."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Cannot read image for Grad-CAM overlay: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(heatmap_color, 0.4, cv2.cvtColor(image, cv2.COLOR_RGB2BGR), 0.6, 0)
    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(6, 6))
    plt.imshow(overlay)
    plt.axis("off")
    plt.title("Grad-CAM")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close()


def write_evaluation_report(output_dir: Path, accuracy: float, roc_auc: float, precision: float,
                            recall: float, f1: float, y_true, y_pred) -> Path:
    """Write the evaluation report text file using the same content structure."""
    metrics_path = output_dir / "evaluation_report.txt"
    metrics_path.write_text(
        f"Validation Accuracy: {accuracy:.4f}\n"
        f"ROC-AUC: {roc_auc:.4f}\n"
        f"Precision: {precision:.4f}\n"
        f"Recall: {recall:.4f}\n"
        f"F1-score: {f1:.4f}\n"
        f"Confusion matrix: {output_dir / 'confusion_matrix.png'}\n"
        f"ROC curve: {output_dir / 'roc_curve.png'}\n"
        f"Classification report:\n{classification_report(y_true, y_pred, target_names=['Benign', 'Malignant'], digits=4)}\n",
        encoding="utf-8",
    )
    return metrics_path


def main():
    """Run evaluation, save outputs, and write the report file."""
    parser = argparse.ArgumentParser(description="Evaluate EfficientNet-B0 medical model with ROC, confusion matrix, and Grad-CAM.")
    parser.add_argument("--checkpoint", type=str, default=str(ROOT_DIR / "models" / "effnet_best.pth"), help="Path to trained EfficientNet-B0 checkpoint.")
    parser.add_argument("--val-csv", type=str, default=str(ROOT_DIR / "datasets" / "annotations" / "val.csv"))
    parser.add_argument("--output-dir", type=str, default=str(ROOT_DIR / "reports"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--gradcam-samples", type=int, default=0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.checkpoint, device)
    loader = build_loader(args.val_csv, batch_size=args.batch_size, num_workers=args.num_workers)

    y_true, y_pred, y_score, image_paths = evaluate_confusion_matrix(model, loader, device, output_dir)
    roc_auc = plot_roc_curve(y_true, y_score, output_dir)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    LOGGER.info("Validation Accuracy: %.4f", accuracy)
    LOGGER.info("ROC-AUC: %.4f", roc_auc)
    LOGGER.info("Precision: %.4f", precision)
    LOGGER.info("Recall: %.4f", recall)
    LOGGER.info("F1-score: %.4f", f1)
    LOGGER.info("Classification Report:\n%s", classification_report(y_true, y_pred, target_names=["Benign", "Malignant"], digits=4))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    LOGGER.info("True Positives: %s", tp)
    LOGGER.info("True Negatives: %s", tn)
    LOGGER.info("False Positives: %s", fp)
    LOGGER.info("False Negatives: %s", fn)

    save_prediction_outputs(image_paths, y_true, y_pred, y_score, output_dir)

    if args.gradcam_samples > 0:
        LOGGER.info("Generating Grad-CAM for %s validation samples...", args.gradcam_samples)
        count = 0
        for images, labels, paths in loader:
            for idx in range(images.size(0)):
                if count >= args.gradcam_samples:
                    break
                try:
                    heatmap, pred_class = generate_gradcam(model, images[idx], device)
                    output_path = output_dir / f"gradcam_{count+1}_{Path(paths[idx]).stem}.png"
                    save_gradcam(heatmap, Path(paths[idx]), output_path)
                    LOGGER.info("Saved Grad-CAM %s -> %s", count + 1, output_path)
                except Exception as exc:
                    LOGGER.info("Grad-CAM failed for %s: %s", paths[idx], exc)
                count += 1
            if count >= args.gradcam_samples:
                break

    metrics_path = write_evaluation_report(output_dir, accuracy, roc_auc, precision, recall, f1, y_true, y_pred)
    LOGGER.info("Saved evaluation report to: %s", metrics_path)


if __name__ == "__main__":
    main()
