from __future__ import annotations

from typing import Dict, Tuple, Optional

import cv2
import numpy as np

from config import IMAGE_SIZE


class ROIService:
    """
    ROI Extraction Service for mammogram lesion regions.

    Input:
        image: RGB image as numpy array
        mask : binary tumor mask as numpy array

    Output:
        ROI crop, bounding box, overlay, and metadata
    """

    def __init__(
        self,
        output_size: int = IMAGE_SIZE,
        padding_ratio: float = 0.15,
        min_area: int = 50
    ):
        self.output_size = output_size
        self.padding_ratio = padding_ratio
        self.min_area = min_area

    def extract(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> Dict:
        image = self._ensure_rgb(image)
        mask = self._prepare_mask(mask)

        bbox = self.get_largest_bbox(mask)

        if bbox is None:
            roi = cv2.resize(
                image,
                (self.output_size, self.output_size)
            )

            return {
                "roi": roi,
                "bbox": None,
                "mask_crop": None,
                "overlay": image,
                "status": "no_valid_lesion_found"
            }

        x, y, w, h = self.add_padding(
            bbox,
            image.shape
        )

        roi = image[y:y + h, x:x + w]
        mask_crop = mask[y:y + h, x:x + w]

        roi = cv2.resize(
            roi,
            (self.output_size, self.output_size)
        )

        mask_crop = cv2.resize(
            mask_crop,
            (self.output_size, self.output_size),
            interpolation=cv2.INTER_NEAREST
        )

        overlay = self.draw_bbox(
            image.copy(),
            (x, y, w, h)
        )

        return {
            "roi": roi,
            "bbox": [int(x), int(y), int(w), int(h)],
            "mask_crop": mask_crop,
            "overlay": overlay,
            "status": "success"
        }

    def get_largest_bbox(
        self,
        mask: np.ndarray
    ) -> Optional[Tuple[int, int, int, int]]:

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        valid_contours = [
            c for c in contours
            if cv2.contourArea(c) >= self.min_area
        ]

        if not valid_contours:
            return None

        largest = max(
            valid_contours,
            key=cv2.contourArea
        )

        return cv2.boundingRect(largest)

    def add_padding(
        self,
        bbox: Tuple[int, int, int, int],
        image_shape
    ) -> Tuple[int, int, int, int]:

        x, y, w, h = bbox
        img_h, img_w = image_shape[:2]

        pad_w = int(w * self.padding_ratio)
        pad_h = int(h * self.padding_ratio)

        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)

        x2 = min(img_w, x + w + pad_w)
        y2 = min(img_h, y + h + pad_h)

        return (
            x1,
            y1,
            x2 - x1,
            y2 - y1
        )

    def draw_bbox(
        self,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int]
    ) -> np.ndarray:

        x, y, w, h = bbox

        cv2.rectangle(
            image,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            2
        )

        return image

    def _prepare_mask(
        self,
        mask: np.ndarray
    ) -> np.ndarray:

        if mask.ndim == 3:
            mask = cv2.cvtColor(
                mask,
                cv2.COLOR_RGB2GRAY
            )

        mask = cv2.resize(
            mask,
            (self.output_size, self.output_size)
        )

        _, mask = cv2.threshold(
            mask,
            127,
            255,
            cv2.THRESH_BINARY
        )

        return mask.astype(np.uint8)

    def _ensure_rgb(
        self,
        image: np.ndarray
    ) -> np.ndarray:

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
                f"Unsupported image shape for ROI extraction: {image.shape}."
            )

        return image


