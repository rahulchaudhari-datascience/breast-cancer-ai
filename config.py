"""Central configuration for the Breast Cancer AI project.

This module preserves the existing public configuration names for backward
compatibility while improving readability through grouping, comments, and
descriptive constants.
"""

import os
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set deterministic seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =====================================================
# PROJECT ROOT
# =====================================================

ROOT_DIR = Path(__file__).resolve().parent

# Kaggle / Colab friendly overrides.
DATASET_ROOT = os.getenv("DATASET_ROOT") or os.getenv("KAGGLE_DATASET_PATH") or ""
OUTPUT_ROOT = os.getenv("OUTPUT_ROOT") or os.getenv("KAGGLE_WORKING_DIR") or ""


# =====================================================
# DATASET PATHS
# =====================================================

DATASETS_DIR = Path(DATASET_ROOT).expanduser().resolve() if DATASET_ROOT else (ROOT_DIR / "datasets")
RAW_DATA_DIR = DATASETS_DIR / "raw"
PROCESSED_DATA_DIR = DATASETS_DIR / "processed"
MASKS_DIR = DATASETS_DIR / "masks"
ANNOTATIONS_DIR = DATASETS_DIR / "annotations"


# =====================================================
# MODEL PATHS
# =====================================================

MODELS_DIR = ROOT_DIR / "models"
NNUNET_DIR = MODELS_DIR / "nnunet"
CONVNEXT_DIR = MODELS_DIR / "convnextv2"
RESNET_DIR = MODELS_DIR / "resnet50"
EFFICIENTNET_DIR = MODELS_DIR / "efficientnet"
CHECKPOINT_DIR = MODELS_DIR / "checkpoints"


# =====================================================
# OUTPUT PATHS
# =====================================================

OUTPUTS_DIR = Path(OUTPUT_ROOT).expanduser().resolve() / "outputs" if OUTPUT_ROOT else (ROOT_DIR / "outputs")
MASK_OUTPUT_DIR = OUTPUTS_DIR / "masks"
HEATMAP_OUTPUT_DIR = OUTPUTS_DIR / "heatmaps"
REPORT_OUTPUT_DIR = OUTPUTS_DIR / "reports"
METRICS_OUTPUT_DIR = OUTPUTS_DIR / "metrics"
PREDICTIONS_OUTPUT_DIR = OUTPUTS_DIR / "predictions"
LOGS_DIR = OUTPUTS_DIR / "logs"
TENSORBOARD_DIR = OUTPUTS_DIR / "tensorboard"


# =====================================================
# DEVICE CONFIGURATION
# =====================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =====================================================
# IMAGE CONFIGURATION
# =====================================================

DEFAULT_IMAGE_SIZE = 224
IMAGE_SIZE = DEFAULT_IMAGE_SIZE

DEFAULT_NUM_CHANNELS = 3
NUM_CHANNELS = DEFAULT_NUM_CHANNELS


# =====================================================
# TRAINING CONFIGURATION
# =====================================================

DEFAULT_BATCH_SIZE = 16
BATCH_SIZE = DEFAULT_BATCH_SIZE

DEFAULT_NUM_WORKERS = 4
NUM_WORKERS = DEFAULT_NUM_WORKERS

DEFAULT_EPOCHS = 30
EPOCHS = DEFAULT_EPOCHS

DEFAULT_LEARNING_RATE = 1e-4
LEARNING_RATE = DEFAULT_LEARNING_RATE

DEFAULT_WEIGHT_DECAY = 1e-4
WEIGHT_DECAY = DEFAULT_WEIGHT_DECAY

DEFAULT_EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_PATIENCE = DEFAULT_EARLY_STOPPING_PATIENCE

# Use pretrained weights for both classification and encoder backbones.
PRETRAINED = True

# Default classification backbone for Kaggle / local training.
CLASSIFICATION_MODEL_NAME = "convnext_tiny"

# Encoder weights identifier for segmentation models (e.g. 'imagenet' or None).
ENCODER_WEIGHTS = "imagenet"

DEFAULT_RANDOM_SEED = 42
RANDOM_SEED = DEFAULT_RANDOM_SEED

ENABLE_HORIZONTAL_FLIP = True


# =====================================================
# MODEL CONFIGURATION
# =====================================================

DEFAULT_NUM_CLASSES = 2
NUM_CLASSES = DEFAULT_NUM_CLASSES

DEFAULT_BIRADS_CLASS_COUNT = 4
BIRADS_CLASSES = DEFAULT_BIRADS_CLASS_COUNT

DEFAULT_DROPOUT_RATE = 0.2
DROPOUT_RATE = DEFAULT_DROPOUT_RATE


# =====================================================
# LABELS
# =====================================================

CLASS_NAMES = {
    0: "Benign",
    1: "Malignant",
}

BIRADS_LABELS = {
    0: "BI-RADS 2",
    1: "BI-RADS 3",
    2: "BI-RADS 4",
    3: "BI-RADS 5",
}


# =====================================================
# THRESHOLDS
# =====================================================

DEFAULT_SEGMENTATION_THRESHOLD = 0.5
SEGMENTATION_THRESHOLD = DEFAULT_SEGMENTATION_THRESHOLD

DEFAULT_CLASSIFICATION_THRESHOLD = 0.5
CLASSIFICATION_THRESHOLD = DEFAULT_CLASSIFICATION_THRESHOLD


# =====================================================
# CHECKPOINTS
# =====================================================

NNUNET_CHECKPOINT = CHECKPOINT_DIR / "nnunet_best.pth"
UNETPP_CHECKPOINT = CHECKPOINT_DIR / "unetpp_best.pth"
CONVNEXT_CHECKPOINT = CHECKPOINT_DIR / "convnextv2_best.pth"
CLASSIFICATION_CHECKPOINT = CHECKPOINT_DIR / "classification_best.pth"

# Backward-compatible alias for the classifier checkpoint.
EFFICIENTNET_CHECKPOINT = CLASSIFICATION_CHECKPOINT


# =====================================================
# DIRECTORY CREATION
# =====================================================

ALL_DIRS = [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MASKS_DIR,
    ANNOTATIONS_DIR,
    CHECKPOINT_DIR,
    MASK_OUTPUT_DIR,
    HEATMAP_OUTPUT_DIR,
    REPORT_OUTPUT_DIR,
    METRICS_OUTPUT_DIR,
    PREDICTIONS_OUTPUT_DIR,
    LOGS_DIR,
    TENSORBOARD_DIR,
]

for directory in ALL_DIRS:
    directory.mkdir(parents=True, exist_ok=True)


