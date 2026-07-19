"""Shared helpers for utility modules.

These helpers keep repeated image conversion and mask normalization logic in
one place while preserving the existing public utility APIs.
"""

from __future__ import annotations

import cv2
import numpy as np


def ensure_rgb(image: np.ndarray) -> np.ndarray:
    """Normalize grayscale or RGBA images to RGB."""
    if image is None:
        raise ValueError("Image must not be None.")

    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Unsupported image shape: {image.shape}. Expected HxWx3 or HxW.")

    return image


def normalize_mask_to_uint8(mask: np.ndarray) -> np.ndarray:
    """Normalize a mask to uint8 while preserving the existing scaling behavior."""
    if mask is None:
        raise ValueError("Mask must not be None.")

    mask_arr = mask.astype(np.float32)
    if mask_arr.max() <= 1.0:
        mask_arr *= 255.0

    return np.clip(mask_arr, 0, 255).astype(np.uint8)