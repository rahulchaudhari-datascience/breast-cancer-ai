from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from services.classification_service import ClassificationService
from services.preprocessing_service import PreprocessingService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on breast cancer images.")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input image file or directory of images.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint path to load the trained model.",
    )
    return parser.parse_args()


def load_image_paths(input_path: Path) -> List[Path]:
    if input_path.is_dir():
        return sorted(
            [
                path
                for path in input_path.rglob("**/*.*")
                if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
            ]
        )
    if input_path.is_file():
        return [input_path]
    raise FileNotFoundError(f"Input path not found: {input_path}")


def main() -> None:
    args = parse_args()
    image_paths = load_image_paths(args.input)
    if not image_paths:
        raise RuntimeError("No valid input images were found.")

    classifier = ClassificationService(checkpoint_path=str(args.checkpoint) if args.checkpoint else None)
    preprocessing = PreprocessingService()

    results: List[Dict] = []

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Skipping invalid image: {image_path}")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        try:
            prediction = classifier.predict(image)
        except Exception as exc:
            print(f"Failed to process {image_path}: {exc}")
            continue

        output = {
            "image_path": str(image_path),
            "prediction": prediction["prediction"],
            "class_id": prediction["class_id"],
            "probability": prediction["probability"],
            "confidence": prediction["confidence"],
            "probabilities": prediction["probabilities"],
        }
        results.append(output)

        print(f"{image_path.name}: {output['prediction']} ({output['probability']:.4f})")

    if results:
        print("Inference completed for %d images." % len(results))


if __name__ == "__main__":
    main()
