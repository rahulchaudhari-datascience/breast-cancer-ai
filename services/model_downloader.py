"""
Model downloader and loader utility for models trained in Google Colab.

Usage:
    python -c "from services.model_downloader import download_model; download_model('classification')"
"""

from pathlib import Path
from typing import Optional
import torch


MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


def download_model(
    model_type: str,  # "classification", "segmentation"
    force_download: bool = False,
) -> Path:
    """
    Download or check for a trained model.
    
    Args:
        model_type: 'classification' or 'segmentation'
        force_download: re-download even if exists
    
    Returns:
        Path to model file
    
    Note: 
        Models should be uploaded to Google Drive during Colab training.
        Manual download via gdown or drag-drop into /models/ is required.
    """
    model_path = MODELS_DIR / f"{model_type}_model.pth"
    
    if model_path.exists() and not force_download:
        print(f"[INFO] Using cached model: {model_path}")
        return model_path
    
    print(
        f"[WARNING] Model not found at {model_path}.\n"
        f"Steps to download:\n"
        f"  1. Train model in Google Colab (see notebooks/colab_training.ipynb)\n"
        f"  2. Save to Google Drive during training\n"
        f"  3. Download to /models/ folder locally\n"
        f"  4. Run inference again"
    )
    
    raise FileNotFoundError(
        f"Model file {model_path} not found. "
        f"Train model in Colab and download to {model_path}."
    )


def load_checkpoint(model_path: Path) -> dict:
    """Load a PyTorch checkpoint."""
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")
    
    return torch.load(model_path, map_location="cpu")
