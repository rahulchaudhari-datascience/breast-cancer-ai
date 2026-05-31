from pathlib import Path
import torch

# =====================================================
# PROJECT ROOT
# =====================================================

ROOT_DIR = Path(__file__).resolve().parent

# =====================================================
# DATASET PATHS
# =====================================================

DATASETS_DIR = ROOT_DIR / "datasets"

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

OUTPUTS_DIR = ROOT_DIR / "outputs"

MASK_OUTPUT_DIR = OUTPUTS_DIR / "masks"
HEATMAP_OUTPUT_DIR = OUTPUTS_DIR / "heatmaps"
REPORT_OUTPUT_DIR = OUTPUTS_DIR / "reports"
METRICS_OUTPUT_DIR = OUTPUTS_DIR / "metrics"
PREDICTIONS_OUTPUT_DIR = OUTPUTS_DIR / "predictions"

# =====================================================
# DEVICE CONFIGURATION
# =====================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================================
# IMAGE CONFIGURATION
# =====================================================

IMAGE_SIZE = 512

NUM_CHANNELS = 3

# =====================================================
# TRAINING CONFIGURATION
# =====================================================

BATCH_SIZE = 16

NUM_WORKERS = 4

EPOCHS = 100

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

RANDOM_SEED = 42

# =====================================================
# MODEL CONFIGURATION
# =====================================================

NUM_CLASSES = 2

BIRADS_CLASSES = 4

DROPOUT_RATE = 0.2

# =====================================================
# LABELS
# =====================================================

CLASS_NAMES = {
    0: "Benign",
    1: "Malignant"
}

BIRADS_LABELS = {
    0: "BI-RADS 2",
    1: "BI-RADS 3",
    2: "BI-RADS 4",
    3: "BI-RADS 5"
}

# =====================================================
# THRESHOLDS
# =====================================================

SEGMENTATION_THRESHOLD = 0.5

CLASSIFICATION_THRESHOLD = 0.5

# =====================================================
# CHECKPOINTS
# =====================================================

NNUNET_CHECKPOINT = (
    CHECKPOINT_DIR /
    "nnunet_best.pth"
)

CONVNEXT_CHECKPOINT = (
    CHECKPOINT_DIR /
    "convnextv2_best.pth"
)

# =====================================================
# CREATE DIRECTORIES
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
    PREDICTIONS_OUTPUT_DIR
]

for directory in ALL_DIRS:
    directory.mkdir(
        parents=True,
        exist_ok=True
    )


