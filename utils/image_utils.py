"""General-purpose image helpers used across the project."""

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

from utils._shared import ensure_rgb


def read_image(path: str) -> np.ndarray:
    """Read an image from disk and return it in RGB format."""
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    image = cv2.imread(str(path_obj))
    if image is None:
        raise ValueError(f"Could not read image file: {path}")

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def resize_image(
    image: np.ndarray,
    size: Tuple[int, int] = (512, 512),
) -> np.ndarray:
    """Resize an image to the requested spatial size."""
    if image is None:
        raise ValueError("Cannot resize a None image.")

    return cv2.resize(image, size)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize an image to floating point values in the [0, 1] range."""
    if image is None:
        raise ValueError("Cannot normalize a None image.")

    image = image.astype(np.float32)
    image = np.clip(image, 0, 255) / 255.0
    return image


def apply_clahe(image: np.ndarray) -> np.ndarray:
    """Apply CLAHE to an image and return a 3-channel RGB result."""
    if image is None:
        raise ValueError("Cannot apply CLAHE to a None image.")

    image = ensure_rgb(image)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)



