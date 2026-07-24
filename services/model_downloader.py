"""Model discovery helper for artifacts stored in the local `models/` folder.

Usage:
    python -c "from services.model_downloader import download_model; download_model('classification')"
"""

from pathlib import Path
import logging
import torch


logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


def download_model(
    model_type: str,
    force_download: bool = False,
) -> Path:
    """Return the expected local path for a trained model artifact.

    The function preserves the existing backward-compatible file naming
    behavior used by older training scripts.
    """
    model_path = MODELS_DIR / f"{model_type}_model.pth"
    
    if model_path.exists() and not force_download:
        logger.info("Using cached model: %s", model_path)
        return model_path
    
    logger.warning(
        "Model not found at %s.\nSteps to download:\n"
        "  1. Train model in Google Colab (see notebooks/colab_training.ipynb)\n"
        "  2. Save to Google Drive during training\n"
        "  3. Download to /models/ folder locally\n"
        "  4. Run inference again",
        model_path,
    )
    
    raise FileNotFoundError(
        f"Model file {model_path} not found. "
        f"Train model in Colab and download to {model_path}."
    )


def load_checkpoint(model_path: Path) -> dict:
    """Load a PyTorch checkpoint from disk."""
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")
    
    return torch.load(model_path, map_location="cpu")
