"""Train an EfficientNet-B0 classifier on CBIS-DDSM-style annotations.

This module preserves the original training behavior, hyperparameters, and
model setup while improving readability, organization, and error reporting.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from config import RANDOM_SEED, set_seed

logging.basicConfig(level=logging.INFO, format="%(message)s")
LOGGER = logging.getLogger(__name__)


class CBISDataset(Dataset):
    """Dataset wrapper for CBIS-DDSM-style image and label CSV files."""

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
        img_path = row["image_path"]
        label = self._normalize_label(row["label"])

        try:
            with Image.open(img_path) as image:
                image = image.convert("RGB")
        except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
            raise ValueError(f"Failed to load image '{img_path}'") from exc

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)

    def _normalize_label(self, label):
        if isinstance(label, str):
            label = label.strip().upper()
            if label in self.label_map:
                return self.label_map[label]
            return int(label)
        return int(label)


def get_transforms():
    """Return training and validation transforms without changing the pipeline."""
    train_transforms = [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(15),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.08, 0.08),
            scale=(0.9, 1.1),
        ),
        transforms.ColorJitter(
            brightness=0.10,
            contrast=0.10,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]

    train_tfms = transforms.Compose(train_transforms)

    val_tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return train_tfms, val_tfms


def build_data_loaders(train_csv: str, val_csv: str, batch_size: int, num_workers: int):
    """Build training and validation data loaders using the existing configuration."""
    train_tfms, val_tfms = get_transforms()

    train_ds = CBISDataset(train_csv, transform=train_tfms)
    val_ds = CBISDataset(val_csv, transform=val_tfms)

    generator = torch.Generator()
    generator.manual_seed(RANDOM_SEED)

    train_loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": num_workers,
        "generator": generator,
        "pin_memory": torch.cuda.is_available(),
    }
    val_loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }

    if num_workers > 0:
        train_loader_kwargs["persistent_workers"] = True
        train_loader_kwargs["prefetch_factor"] = 2
        val_loader_kwargs["persistent_workers"] = True
        val_loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(train_ds, **train_loader_kwargs)
    val_loader = DataLoader(val_ds, **val_loader_kwargs)

    return train_loader, val_loader, train_ds


best_auc = 0.0


def save_best_model(model: nn.Module, val_auc: float, output_dir: Path) -> None:
    """Persist the best model checkpoint when validation AUC improves."""
    global best_auc
    if val_auc > best_auc:
        best_auc = val_auc
        output_path = output_dir / "effnet_best.pth"
        torch.save(model.state_dict(), output_path)
        LOGGER.info("New best AUC: %.4f. Saved best model to %s", val_auc, output_path)


def build_model(device: torch.device) -> nn.Module:
    """Build the EfficientNet-B0 classifier with the original architecture."""
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    return model.to(device)


class FocalLoss(nn.Module):
    """Focal loss wrapper that preserves the original training objective."""

    def __init__(self, class_weights: torch.Tensor, alpha: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss


def train_one_epoch(model, criterion, optimizer, loader, device, scaler):
    """Run one training epoch with the existing optimization loop."""
    model.train()
    running_loss = 0.0

    for images, labels in loader:
        images = images.to(device, non_blocking=torch.cuda.is_available())
        labels = labels.to(device, non_blocking=torch.cuda.is_available())

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            device_type="cuda",
            enabled=torch.cuda.is_available(),
        ):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

    return running_loss / len(loader)


@torch.no_grad()
def validate(model, loader, device):
    """Return the validation ROC-AUC for the current model state."""
    model.eval()
    all_probs = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device, non_blocking=torch.cuda.is_available())
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)[:, 1]

        all_probs.extend(probs.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    if len(set(all_labels)) < 2:
        return 0.0

    return roc_auc_score(all_labels, all_probs)


@torch.no_grad()
def validate_with_auc(model, loader, device):
    """Compatibility wrapper for validation scoring."""
    return validate(model, loader, device)


class EarlyStopping:
    """Stop training after the validation metric stops improving."""

    def __init__(self, patience: int = 5):
        self.patience = patience
        self.best_auc = 0.0
        self.counter = 0

    def step(self, val_auc: float) -> bool:
        if val_auc > self.best_auc:
            self.best_auc = val_auc
            self.counter = 0
            return False

        self.counter += 1
        return self.counter >= self.patience


def main():
    """Parse CLI arguments and run the training loop."""
    parser = argparse.ArgumentParser(description="Train EfficientNet-B0 on CBIS-DDSM annotations.")
    parser.add_argument("--train-csv", type=str, default="datasets/annotations/train.csv")
    parser.add_argument("--val-csv", type=str, default="datasets/annotations/val.csv")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    args = parser.parse_args()

    set_seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Using device: %s", device)
    LOGGER.info("Reproducibility seed: %s", RANDOM_SEED)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        train_loader, val_loader, train_ds = build_data_loaders(
            str(args.train_csv),
            str(args.val_csv),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to build data loaders: {exc}") from exc

    labels = train_ds.data["label"].apply(
        lambda x: train_ds.label_map[x.strip().upper()] if isinstance(x, str) else int(x)
    ).values
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=labels,
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    model = build_model(device)
    criterion = FocalLoss(class_weights=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        threshold=1e-4,
        min_lr=1e-6,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=torch.cuda.is_available(),
    )
    early_stopping = EarlyStopping(patience=args.patience)

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, criterion, optimizer, train_loader, device, scaler)
        val_auc = validate_with_auc(model, val_loader, device)
        scheduler.step(val_auc)
        current_lr = optimizer.param_groups[0]["lr"]

        LOGGER.info(
            "Epoch %s/%s | Loss: %.4f | Val AUC: %.4f | LR=%.6f",
            epoch + 1,
            args.epochs,
            train_loss,
            val_auc,
            current_lr,
        )

        torch.save(model.state_dict(), args.output_dir / f"effnet_b0_epoch_{epoch + 1}.pth")
        save_best_model(model, val_auc, args.output_dir)

        if early_stopping.step(val_auc):
            LOGGER.info("Early stopping triggered")
            break

    LOGGER.info("Training configuration summary:")
    LOGGER.info("✓ EfficientNet-B0")
    LOGGER.info("✓ AdamW")
    LOGGER.info("✓ Weight decay = 1e-4")
    LOGGER.info("✓ EarlyStopping patience = 5")
    LOGGER.info("✓ Improved augmentations")
    LOGGER.info("✓ AMP updated")
    LOGGER.info("✓ Scheduler unchanged")
    LOGGER.info("Training complete. Checkpoints saved in: %s", args.output_dir)


if __name__ == "__main__":
    main()
