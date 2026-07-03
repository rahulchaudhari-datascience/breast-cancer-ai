import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import auc, classification_report, confusion_matrix, roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


class CBISDataset(Dataset):
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
        label = row["label"]

        if isinstance(label, str):
            normalized = label.strip().upper()
            label = self.label_map.get(normalized, int(normalized))
        else:
            label = int(label)

        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long), image_path


def get_transforms():
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
    model = models.efficientnet_b0(pretrained=False)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model = model.to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model


def evaluate_confusion_matrix(model, loader, device, output_dir: Path):
    y_true = []
    y_pred = []

    model.eval()
    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=["Benign", "Malignant"], digits=4)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    confusion_path = output_dir / "confusion_matrix.png"
    fig.savefig(confusion_path, bbox_inches="tight")
    plt.close(fig)

    print("Confusion Matrix:\n", report)
    print(f"Saved confusion matrix to: {confusion_path}")
    return y_true, y_pred


def plot_roc_curve(y_true, y_score, output_dir: Path):
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

    print(f"Saved ROC curve to: {roc_path}")
    return roc_auc


def _register_hooks(target_layer, features, gradients):
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


def main():
    parser = argparse.ArgumentParser(description="Evaluate EfficientNet-B0 medical model with ROC, confusion matrix, and Grad-CAM.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained EfficientNet-B0 checkpoint.")
    parser.add_argument("--val-csv", type=str, default="datasets/annotations/val.csv")
    parser.add_argument("--output-dir", type=str, default="outputs/evaluation")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--gradcam-samples", type=int, default=5)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.checkpoint, device)
    loader = build_loader(args.val_csv, batch_size=args.batch_size, num_workers=args.num_workers)

    y_true, y_pred = evaluate_confusion_matrix(model, loader, device, output_dir)

    all_probs = []
    all_labels = []
    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())

    roc_auc = plot_roc_curve(all_labels, all_probs, output_dir)
    print(f"Validation ROC AUC: {roc_auc:.4f}")

    if args.gradcam_samples > 0:
        print(f"Generating Grad-CAM for {args.gradcam_samples} validation samples...")
        count = 0
        for images, labels, paths in loader:
            for idx in range(images.size(0)):
                if count >= args.gradcam_samples:
                    break
                try:
                    heatmap, pred_class = generate_gradcam(model, images[idx], device)
                    output_path = output_dir / f"gradcam_{count+1}_{Path(paths[idx]).stem}.png"
                    save_gradcam(heatmap, Path(paths[idx]), output_path)
                    print(f"Saved Grad-CAM {count+1} -> {output_path}")
                except Exception as exc:
                    print(f"Grad-CAM failed for {paths[idx]}: {exc}")
                count += 1
            if count >= args.gradcam_samples:
                break

    metrics_path = output_dir / "evaluation_report.txt"
    metrics_path.write_text(
        f"Validation ROC AUC: {roc_auc:.4f}\n"
        f"Confusion matrix: {output_dir / 'confusion_matrix.png'}\n"
        f"ROC curve: {output_dir / 'roc_curve.png'}\n"
        f"Classification report:\n{classification_report(y_true, y_pred, target_names=['Benign', 'Malignant'], digits=4)}\n",
        encoding="utf-8",
    )
    print(f"Saved evaluation report to: {metrics_path}")


if __name__ == "__main__":
    main()
