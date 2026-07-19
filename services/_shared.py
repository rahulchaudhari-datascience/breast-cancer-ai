"""Shared helpers for service-layer components.

These utilities centralize repeated non-business logic such as checkpoint
loading and common image conversions so the service classes can stay focused
on their core responsibilities.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import logging

import numpy as np
import torch


def get_service_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger for a service class or helper module."""
    return logging.getLogger(name)


def load_checkpoint_payload(checkpoint_path: str | Path, map_location: Any) -> Any:
    """Load a PyTorch checkpoint payload from disk.

    The caller remains responsible for interpreting the checkpoint structure,
    which keeps backward compatibility with existing checkpoint formats.
    """
    path = Path(checkpoint_path)
    return torch.load(path, map_location=map_location)


def extract_state_dict(checkpoint: Any) -> Any:
    """Extract a state dict from a checkpoint payload when a wrapper dict is used."""
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


def load_state_dict_into_model(model, checkpoint_path: str | Path, map_location: Any) -> Any:
    """Load a checkpoint into a model and return the raw checkpoint payload.

    This helper keeps the calling service in control of its own logging and any
    fallback behavior while removing repeated checkpoint parsing logic.
    """
    checkpoint = load_checkpoint_payload(checkpoint_path, map_location=map_location)
    model.load_state_dict(extract_state_dict(checkpoint))
    return checkpoint


def normalize_uint8_image(image: np.ndarray) -> np.ndarray:
    """Convert a numeric image array to uint8 while preserving the existing scaling behavior."""
    if image is None:
        raise ValueError("Image is None.")

    try:
        if getattr(image, "dtype", None) is not None and image.dtype == np.uint8:
            return image
        clipped = np.clip(image, 0.0, 1.0)
        return (clipped * 255).astype(np.uint8)
    except Exception:
        return np.clip(image, 0, 255).astype(np.uint8)
