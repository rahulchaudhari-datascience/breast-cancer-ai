from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List

import cv2

from services.classification_service import ClassificationService
from services.preprocessing_service import PreprocessingService

logging.basicConfig(level=logging.INFO, format="%(message)s")
LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for batch inference."""
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
    """Return a sorted list of supported image paths from a file or directory."""
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


def build_services(checkpoint_path: Path | None):
    """Create the classifier and preprocessing services for inference."""
    classifier = ClassificationService(checkpoint_path=str(checkpoint_path) if checkpoint_path else None)
    preprocessing = PreprocessingService()
    return classifier, preprocessing


def run_inference(image_paths: List[Path], classifier, preprocessing) -> List[Dict]:
    """Process each image and return a list of prediction dictionaries."""
    results: List[Dict] = []

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            LOGGER.info("Skipping invalid image: %s", image_path)
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        try:
            prediction = classifier.predict(image)
        except Exception as exc:
            LOGGER.info("Failed to process %s: %s", image_path, exc)
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

        LOGGER.info("%s: %s (%.4f)", image_path.name, output["prediction"], output["probability"])

    return results


def main() -> None:
    """CLI entry point for running inference over one image or a directory."""
    args = parse_args()
    image_paths = load_image_paths(args.input)
    if not image_paths:
        raise RuntimeError("No valid input images were found.")

    classifier, preprocessing = build_services(args.checkpoint)
    results = run_inference(image_paths, classifier, preprocessing)

    if results:
        LOGGER.info("Inference completed for %d images.", len(results))


if __name__ == "__main__":
    main()
