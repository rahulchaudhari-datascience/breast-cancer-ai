from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import torch
import albumentations as A

from albumentations.pytorch import ToTensorV2

from config import IMAGE_SIZE


class PreprocessingService:
    """
    Enterprise-grade mammogram preprocessing service.

    Pipeline:
    1. Artifact Removal
    2. Breast Region Extraction
    3. CLAHE Enhancement
    4. Intensity Normalization
    5. Resize
    6. Augmentation (training only)
    7. Tensor Conversion
    """

    def __init__(
        self,
        image_size: int = IMAGE_SIZE
    ):

        self.image_size = image_size

        self.train_transform = A.Compose([
            A.HorizontalFlip(p=0.5),

            A.Rotate(
                limit=15,
                border_mode=cv2.BORDER_CONSTANT,
                p=0.5
            ),

            A.RandomBrightnessContrast(
                brightness_limit=0.15,
                contrast_limit=0.15,
                p=0.5
            ),

            A.GaussNoise(
                p=0.2
            ),

            A.Resize(
                image_size,
                image_size
            ),

            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),

            ToTensorV2()
        ])

        self.val_transform = A.Compose([

            A.Resize(
                image_size,
                image_size
            ),

            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),

            ToTensorV2()
        ])

    # -------------------------------------------------
    # PUBLIC API
    # -------------------------------------------------

    def preprocess(
        self,
        image: np.ndarray,
        training: bool = False
    ):
        """
        Main preprocessing entry point.
        """

        image = self.ensure_rgb(image)

        image = self.remove_artifacts(image)

        image = self.extract_breast_region(image)

        image = self.apply_clahe(image)

        image = self.zscore_normalization(image)

        if training:

            image = self.train_transform(
                image=image
            )["image"]

        else:

            image = self.val_transform(
                image=image
            )["image"]

        return image

    def preprocess_for_model(
        self,
        image: np.ndarray,
        image_size: Optional[int] = None,
    ) -> torch.Tensor:
        image = self.ensure_rgb(image)

        size = image_size or self.image_size
        image = cv2.resize(image, (size, size))
        image = image.astype(np.float32) / 255.0

        mean = np.array((0.485, 0.456, 0.406), dtype=np.float32)
        std = np.array((0.229, 0.224, 0.225), dtype=np.float32)

        image = (image - mean) / std

        tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
        return tensor.unsqueeze(0)

    def ensure_rgb(
        self,
        image: np.ndarray
    ) -> np.ndarray:

        if image is None:
            raise ValueError("Input image is None.")

        if image.ndim == 2:
            return cv2.cvtColor(
                image,
                cv2.COLOR_GRAY2RGB
            )

        if image.ndim == 3 and image.shape[2] == 4:
            return cv2.cvtColor(
                image,
                cv2.COLOR_RGBA2RGB
            )

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"Unsupported image shape: {image.shape}. Expected HxWx3 or HxW."
            )

        return image

    # -------------------------------------------------
    # ARTIFACT REMOVAL
    # -------------------------------------------------

    def remove_artifacts(
        self,
        image: np.ndarray
    ) -> np.ndarray:

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2GRAY
        )

        _, thresh = cv2.threshold(
            gray,
            5,
            255,
            cv2.THRESH_BINARY
        )

        kernel = np.ones(
            (5, 5),
            np.uint8
        )

        thresh = cv2.morphologyEx(
            thresh,
            cv2.MORPH_OPEN,
            kernel
        )

        result = cv2.bitwise_and(
            image,
            image,
            mask=thresh
        )

        return result

    # -------------------------------------------------
    # BREAST EXTRACTION
    # -------------------------------------------------

    def extract_breast_region(
        self,
        image: np.ndarray
    ) -> np.ndarray:

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2GRAY
        )

        _, thresh = cv2.threshold(
            gray,
            10,
            255,
            cv2.THRESH_BINARY
        )

        contours, _ = cv2.findContours(
            thresh,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:

            return image

        largest = max(
            contours,
            key=cv2.contourArea
        )

        x, y, w, h = cv2.boundingRect(
            largest
        )

        breast = image[
            y:y+h,
            x:x+w
        ]

        return breast

    # -------------------------------------------------
    # CLAHE
    # -------------------------------------------------

    def apply_clahe(
        self,
        image: np.ndarray
    ) -> np.ndarray:

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2GRAY
        )

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        enhanced = clahe.apply(
            gray
        )

        enhanced = cv2.cvtColor(
            enhanced,
            cv2.COLOR_GRAY2RGB
        )

        return enhanced

    # -------------------------------------------------
    # NORMALIZATION
    # -------------------------------------------------

    def zscore_normalization(
        self,
        image: np.ndarray
    ) -> np.ndarray:

        image = image.astype(
            np.float32
        )

        mean = np.mean(image)

        std = np.std(image)

        std = max(
            std,
            1e-8
        )

        image = (
            image - mean
        ) / std

        image = cv2.normalize(
            image,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        )

        image = image.astype(
            np.uint8
        )

        return image

