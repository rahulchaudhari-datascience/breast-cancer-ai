from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


def read_image(path: str) -> np.ndarray:
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
    if image is None:
        raise ValueError("Cannot resize a None image.")

    return cv2.resize(image, size)


def normalize_image(image: np.ndarray) -> np.ndarray:
    if image is None:
        raise ValueError("Cannot normalize a None image.")

    image = image.astype(np.float32)
    image = np.clip(image, 0, 255) / 255.0
    return image


def apply_clahe(image: np.ndarray) -> np.ndarray:
    if image is None:
        raise ValueError("Cannot apply CLAHE to a None image.")

    if image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    if image.ndim == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)



