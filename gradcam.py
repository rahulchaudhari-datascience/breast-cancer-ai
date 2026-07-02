from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from services.classification_service import ClassificationService
from services.explainability_service import ExplainabilityService
from services.preprocessing_service import PreprocessingService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate GradCAM heatmaps for selected images.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional path to a trained classification checkpoint.",
    )
    parser.add_argument(
        "--images",
        type=Path,
        nargs="+",
        required=True,
        help="One or more image files or a directory containing images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "heatmaps",
        help="Output directory for GradCAM heatmaps.",
    )
    return parser.parse_args()


def load_images(input_paths: List[Path]) -> List[Path]:
    image_files: List[Path] = []
    for path in input_paths:
        if path.is_dir():
            image_files.extend(sorted(path.glob("**/*.*")))
        elif path.is_file():
            image_files.append(path)
        else:
            raise FileNotFoundError(f"Input path not found: {path}")
    return [p for p in image_files if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}]


def main() -> None:
    args = parse_args()
    image_paths = load_images(args.images)

    if not image_paths:
        raise RuntimeError("No valid image files found for GradCAM generation.")

    classification_service = ClassificationService(checkpoint_path=str(args.checkpoint) if args.checkpoint else None)
    explainability_service = ExplainabilityService(model=classification_service.model)
    preprocessing_service = PreprocessingService()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Skipping invalid image: {image_path}")
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        processed = preprocessing_service.preprocess(image, training=False)
        roi = processed.permute(1, 2, 0).cpu().numpy()
        roi = (roi - roi.min()) / (roi.max() - roi.min() + 1e-8)
        roi = (roi * 255).astype("uint8")

        predicted = classification_service.predict(roi)
        heatmap = explainability_service.generate(
            roi=roi,
            class_id=predicted["class_id"],
            save_path=str(args.output_dir / f"{image_path.stem}_gradcam.png"),
        )

        if heatmap is not None:
            print(f"Saved GradCAM for {image_path.name} -> {args.output_dir / f'{image_path.stem}_gradcam.png'}")
        else:
            print(f"GradCAM generation failed for {image_path.name}")


if __name__ == "__main__":
    main()
