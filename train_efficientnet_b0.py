import argparse
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

from config import ENABLE_HORIZONTAL_FLIP, RANDOM_SEED, set_seed


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
        img_path = row["image_path"]
        label = row["label"]

        if isinstance(label, str):
            label = label.strip().upper()
            if label in self.label_map:
                label = self.label_map[label]
            else:
                label = int(label)
        else:
            label = int(label)

        try:
            with Image.open(img_path) as image:
                image = image.convert("RGB")
        except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
            raise ValueError(f"Failed to load image '{img_path}'") from exc

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


def get_transforms():
    train_transforms = [
        transforms.Resize((260, 260)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.05, 0.05),
            scale=(0.95, 1.05),
        ),
        transforms.ColorJitter(
            brightness=0.05,
            contrast=0.05,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]

    train_tfms = transforms.Compose(train_transforms)

    val_tfms = transforms.Compose([
        transforms.Resize((260, 260)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return train_tfms, val_tfms


def build_data_loaders(train_csv: str, val_csv: str, batch_size: int, num_workers: int):
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
    global best_auc
    if val_auc > best_auc:
        best_auc = val_auc
        output_path = output_dir / "effnet_b2_best.pth"
        torch.save(model.state_dict(), output_path)
        print(f"New best AUC: {val_auc:.4f}. Saved best model to {output_path}")

def build_model(device: torch.device) -> nn.Module:
    model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    return model.to(device)


class FocalLoss(nn.Module):
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
    model.train()
    running_loss = 0.0

    for images, labels in loader:
        images = images.to(device, non_blocking=torch.cuda.is_available())
        labels = labels.to(device, non_blocking=torch.cuda.is_available())

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
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
    return validate(model, loader, device)


class EarlyStopping:
    def __init__(self, patience: int = 3):
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
    parser = argparse.ArgumentParser(description="Train EfficientNet-B2 on CBIS-DDSM annotations.")
    parser.add_argument("--train-csv", type=str, default="datasets/annotations/train.csv")
    parser.add_argument("--val-csv", type=str, default="datasets/annotations/val.csv")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    args = parser.parse_args()

    set_seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Reproducibility seed: {RANDOM_SEED}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, train_ds = build_data_loaders(
        str(args.train_csv),
        str(args.val_csv),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    labels = train_ds.data["label"].apply(lambda x: train_ds.label_map[x.strip().upper()] if isinstance(x, str) else int(x)).values
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=labels,
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    model = build_model(device)
    criterion = FocalLoss(class_weights=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        threshold=1e-4,
        min_lr=1e-6,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    early_stopping = EarlyStopping(patience=args.patience)

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, criterion, optimizer, train_loader, device, scaler)
        val_auc = validate_with_auc(model, val_loader, device)
        scheduler.step(val_auc)
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch + 1}/{args.epochs} | Loss: {train_loss:.4f} | Val AUC: {val_auc:.4f} | LR={current_lr:.6f}")

        torch.save(model.state_dict(), args.output_dir / f"effnet_b2_epoch_{epoch + 1}.pth")
        save_best_model(model, val_auc, args.output_dir)

        if early_stopping.step(val_auc):
            print("Early stopping triggered")
            break

    print("Training complete. Checkpoints saved in:", args.output_dir)


if __name__ == "__main__":
    main()
