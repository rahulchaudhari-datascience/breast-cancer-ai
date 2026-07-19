
"""Training pipeline for the EfficientNet-B0 classifier."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple
import logging

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from services.model_builder import build_classification_model
from services.preprocessing_service import PreprocessingService
from utils.metrics_utils import MetricsUtils

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
    EFFICIENTNET_CHECKPOINT,
    PRETRAINED,
    EARLY_STOPPING_PATIENCE,
    LOGS_DIR,
    TENSORBOARD_DIR,
    RANDOM_SEED,
    set_seed,
)



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
    """EfficientNet-B0 classification training pipeline with mixed precision.

    Trains Benign vs Malignant classifier and saves the best checkpoint to
    `EFFICIENTNET_CHECKPOINT`.
    """

    def __init__(
        self,
        model_name: str = "efficientnet_b0",
    ):
        self.device = DEVICE
        self.model_name = model_name

        set_seed(RANDOM_SEED)

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

        self.pin_memory = torch.cuda.is_available()

        # AMP scaler for mixed-precision
        self.scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        TENSORBOARD_DIR.mkdir(parents=True, exist_ok=True)
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(LOGS_DIR / "training.log", encoding="utf-8"),
            ],
        )

        self.writer = SummaryWriter(log_dir=TENSORBOARD_DIR)
        self.logger = logging.getLogger(self.__class__.__name__)

    def _build_model(self) -> nn.Module:
        return build_classification_model(
            model_name=self.model_name,
            num_classes=NUM_CLASSES,
            pretrained=PRETRAINED,
        )

    def create_loaders(
        self,
        train_csv: str,
        val_csv: str,
    ) -> Tuple[DataLoader, DataLoader]:

        train_csv = Path(train_csv)
        val_csv = Path(val_csv)

        if not train_csv.exists():
            raise FileNotFoundError(f"Training CSV not found: {train_csv}")
        if not val_csv.exists():
            raise FileNotFoundError(f"Validation CSV not found: {val_csv}")

        train_df = pd.read_csv(train_csv)
        val_df = pd.read_csv(val_csv)

        if "image_path" not in train_df.columns or "label" not in train_df.columns:
            raise ValueError("Train CSV must contain 'image_path' and 'label' columns.")
        if "image_path" not in val_df.columns or "label" not in val_df.columns:
            raise ValueError("Val CSV must contain 'image_path' and 'label' columns.")

        train_dataset = MammogramClassificationDataset(
            train_df,
            training=True,
        )

        val_dataset = MammogramClassificationDataset(
            val_df,
            training=False,
        )

        if len(train_dataset) == 0:
            raise ValueError("Training dataset is empty. Check train CSV and file paths.")
        if len(val_dataset) == 0:
            raise ValueError("Validation dataset is empty. Check validation CSV and file paths.")

        generator = torch.Generator()
        generator.manual_seed(RANDOM_SEED)

        train_loader_kwargs = {
            "batch_size": BATCH_SIZE,
            "shuffle": True,
            "num_workers": NUM_WORKERS,
            "pin_memory": self.pin_memory,
            "generator": generator,
        }
        val_loader_kwargs = {
            "batch_size": BATCH_SIZE,
            "shuffle": False,
            "num_workers": NUM_WORKERS,
            "pin_memory": self.pin_memory,
        }

        if NUM_WORKERS > 0:
            train_loader_kwargs["persistent_workers"] = True
            train_loader_kwargs["prefetch_factor"] = 2
            val_loader_kwargs["persistent_workers"] = True
            val_loader_kwargs["prefetch_factor"] = 2

        train_loader = DataLoader(
            train_dataset,
            **train_loader_kwargs,
        )

        val_loader = DataLoader(
            val_dataset,
            **val_loader_kwargs,
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
        early_stop_counter = 0

        history = {"train_loss": [], "val_loss": [], "val_auc": []}

        try:
            for epoch in range(EPOCHS):

                train_loss = self._train_one_epoch(train_loader, epoch)

                val_loss, metrics = self._validate(val_loader, epoch)

                self.scheduler.step()

                history["train_loss"].append(train_loss)
                history["val_loss"].append(val_loss)
                history["val_auc"].append(metrics.get("roc_auc", 0.0))

                self.writer.add_scalar("Loss/Train", train_loss, epoch)
                self.writer.add_scalar("Loss/Val", val_loss, epoch)
                self.writer.add_scalar("Metrics/AUC", metrics.get("roc_auc", 0.0), epoch)
                self.writer.add_scalar("Metrics/F1", metrics.get("f1_score", 0.0), epoch)
                self.writer.add_scalar("Metrics/Precision", metrics.get("precision", 0.0), epoch)
                self.writer.add_scalar("Metrics/Recall", metrics.get("recall", 0.0), epoch)
                self.writer.add_scalar("Metrics/Accuracy", metrics.get("accuracy", 0.0), epoch)

                self.logger.info(
                    "Epoch [%s/%s] Train Loss: %.4f Val Loss: %.4f AUC: %.4f F1: %.4f",
                    epoch + 1,
                    EPOCHS,
                    train_loss,
                    val_loss,
                    metrics.get("roc_auc", 0.0),
                    metrics.get("f1_score", 0.0),
                )

                current_auc = metrics.get("roc_auc", 0.0)

                if current_auc > best_auc:
                    best_auc = current_auc
                    best_metrics = metrics
                    early_stop_counter = 0

                    self.save_checkpoint(epoch=epoch, metrics=metrics, path=EFFICIENTNET_CHECKPOINT)
                    self.logger.info("Best model saved.")
                else:
                    early_stop_counter += 1

                if early_stop_counter >= EARLY_STOPPING_PATIENCE:
                    self.logger.info(
                        "Early stopping triggered (no improvement for %s epochs).",
                        EARLY_STOPPING_PATIENCE,
                    )
                    break

        finally:
            try:
                self.writer.close()
            except Exception:
                pass

        return {"best_auc": best_auc, "best_metrics": best_metrics, "history": history}

    def _train_one_epoch(
        self,
        loader: DataLoader,
        epoch: int,
    ) -> float:
        self.model.train()

        running_loss = 0.0

        progress = tqdm(loader, desc=f"Training Epoch {epoch + 1}")

        for images, labels in progress:

            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = self.model(images)
                loss = self.criterion(logits, labels)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item()

            progress.set_postfix(loss=loss.item())

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

            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)

                    probs = torch.softmax(logits, dim=1)
                    preds = torch.argmax(probs, dim=1)

            running_loss += loss.item()

            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
            y_prob.extend(probs[:, 1].cpu().numpy().tolist())

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
    ) -> None:

        """Persist the current training state to disk."""

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


def run_kaggle_training(
    train_csv: str = "datasets/annotations/train.csv",
    val_csv: str = "datasets/annotations/val.csv",
    model_name: str = "efficientnet_b0",
):
    """Convenience entry point for Kaggle / Colab training runs."""
    pipeline = TrainingPipeline(model_name=model_name)
    return pipeline.train(train_csv=train_csv, val_csv=val_csv)


if __name__ == "__main__":
    results = run_kaggle_training()
    print(results)



