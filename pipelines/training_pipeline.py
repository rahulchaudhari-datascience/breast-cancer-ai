
# pipelines/training_pipeline.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import timm

from config import (
    DEVICE,
    IMAGE_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    NUM_CLASSES,
    CLASS_NAMES,
    CHECKPOINT_DIR,
    CONVNEXT_CHECKPOINT,
)

from services.preprocessing_service import PreprocessingService
from utils.metrics_utils import MetricsUtils


class MammogramClassificationDataset(Dataset):
    """
    Dataset CSV must contain:

    image_path,label

    label:
        0 = Benign
        1 = Malignant
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        training: bool = True,
    ):
        self.dataframe = dataframe.reset_index(drop=True)
        self.training = training
        self.preprocessing = PreprocessingService()

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index: int):
        row = self.dataframe.iloc[index]

        image_path = row["image_path"]
        label = int(row["label"])

        image = cv2.imread(image_path)

        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_tensor = self.preprocessing.preprocess(
            image,
            training=self.training,
        )

        label_tensor = torch.tensor(
            label,
            dtype=torch.long,
        )

        return image_tensor, label_tensor


class FocalLoss(nn.Module):
    """
    Focal Loss for imbalanced medical classification.
    """

    def __init__(
        self,
        alpha: float = 0.75,
        gamma: float = 2.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction="none")

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)

        pt = torch.exp(-ce_loss)

        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * ce_loss

        return focal_loss.mean()


class TrainingPipeline:
    """
    ConvNeXt-V2 classification training pipeline.

    Trains:
        Benign vs Malignant classifier

    Saves:
        models/checkpoints/convnextv2_best.pth
    """

    def __init__(
        self,
        model_name: str = "convnextv2_tiny.fcmae_ft_in22k_in1k",
    ):
        self.device = DEVICE
        self.model_name = model_name

        self.model = self._build_model()
        self.model.to(self.device)

        self.criterion = FocalLoss()

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )

        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=EPOCHS,
        )

        CHECKPOINT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

    def _build_model(self) -> nn.Module:
        model = timm.create_model(
            self.model_name,
            pretrained=True,
            num_classes=NUM_CLASSES,
        )

        return model

    def create_loaders(
        self,
        train_csv: str,
        val_csv: str,
    ) -> Tuple[DataLoader, DataLoader]:

        train_df = pd.read_csv(train_csv)
        val_df = pd.read_csv(val_csv)

        train_dataset = MammogramClassificationDataset(
            train_df,
            training=True,
        )

        val_dataset = MammogramClassificationDataset(
            val_df,
            training=False,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

        return train_loader, val_loader

    def train(
        self,
        train_csv: str,
        val_csv: str,
    ) -> Dict:

        train_loader, val_loader = self.create_loaders(
            train_csv,
            val_csv,
        )

        best_auc = 0.0
        best_metrics = {}

        history = {
            "train_loss": [],
            "val_loss": [],
            "val_auc": [],
        }

        for epoch in range(EPOCHS):

            train_loss = self._train_one_epoch(
                train_loader,
                epoch,
            )

            val_loss, metrics = self._validate(
                val_loader,
                epoch,
            )

            self.scheduler.step()

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_auc"].append(metrics.get("roc_auc", 0.0))

            print(
                f"\nEpoch [{epoch + 1}/{EPOCHS}] "
                f"Train Loss: {train_loss:.4f} "
                f"Val Loss: {val_loss:.4f} "
                f"AUC: {metrics.get('roc_auc', 0.0):.4f} "
                f"F1: {metrics.get('f1_score', 0.0):.4f}"
            )

            current_auc = metrics.get("roc_auc", 0.0)

            if current_auc > best_auc:
                best_auc = current_auc
                best_metrics = metrics

                self.save_checkpoint(
                    epoch=epoch,
                    metrics=metrics,
                    path=CONVNEXT_CHECKPOINT,
                )

                print("[INFO] Best model saved.")

        return {
            "best_auc": best_auc,
            "best_metrics": best_metrics,
            "history": history,
        }

    def _train_one_epoch(
        self,
        loader: DataLoader,
        epoch: int,
    ) -> float:

        self.model.train()

        running_loss = 0.0

        progress = tqdm(
            loader,
            desc=f"Training Epoch {epoch + 1}",
        )

        for images, labels in progress:

            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            logits = self.model(images)

            loss = self.criterion(
                logits,
                labels,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=1.0,
            )

            self.optimizer.step()

            running_loss += loss.item()

            progress.set_postfix(
                loss=loss.item()
            )

        return running_loss / len(loader)

    @torch.no_grad()
    def _validate(
        self,
        loader: DataLoader,
        epoch: int,
    ) -> Tuple[float, Dict]:

        self.model.eval()

        running_loss = 0.0

        y_true = []
        y_pred = []
        y_prob = []

        progress = tqdm(
            loader,
            desc=f"Validation Epoch {epoch + 1}",
        )

        for images, labels in progress:

            images = images.to(self.device)
            labels = labels.to(self.device)

            logits = self.model(images)

            loss = self.criterion(
                logits,
                labels,
            )

            probs = torch.softmax(
                logits,
                dim=1,
            )

            preds = torch.argmax(
                probs,
                dim=1,
            )

            running_loss += loss.item()

            y_true.extend(
                labels.cpu().numpy().tolist()
            )

            y_pred.extend(
                preds.cpu().numpy().tolist()
            )

            y_prob.extend(
                probs[:, 1].cpu().numpy().tolist()
            )

        metrics = MetricsUtils.classification_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
        )

        return running_loss / len(loader), metrics

    def save_checkpoint(
        self,
        epoch: int,
        metrics: Dict,
        path,
    ):

        checkpoint = {
            "epoch": epoch,
            "model_name": self.model_name,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "class_names": CLASS_NAMES,
            "image_size": IMAGE_SIZE,
        }

        torch.save(
            checkpoint,
            path,
        )


if __name__ == "__main__":
    pipeline = TrainingPipeline()

    results = pipeline.train(
        train_csv="datasets/annotations/train.csv",
        val_csv="datasets/annotations/val.csv",
    )

    print(results)



