"""Shared helpers for utility modules.

These helpers keep repeated image conversion logic in one place while
preserving the existing public utility APIs.
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