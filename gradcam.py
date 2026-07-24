from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

import cv2
import numpy as np

from services.classification_service import ClassificationService
from services.explainability_service import ExplainabilityService
from services.preprocessing_service import PreprocessingService

logging.basicConfig(level=logging.INFO, format="%(message)s")
LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for Grad-CAM generation."""
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
    """Collect valid image files from one or more input paths."""
    image_files: List[Path] = []
    for path in input_paths:
        if path.is_dir():
            image_files.extend(sorted(path.glob("**/*.*")))
        elif path.is_file():
            image_files.append(path)
        else:
            raise FileNotFoundError(f"Input path not found: {path}")

    return [path for path in image_files if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}]


def prepare_image_for_gradcam(image: np.ndarray) -> np.ndarray:
    """Convert an image to the format expected by the explainability pipeline."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    processed = PreprocessingService().preprocess(image, training=False)
    processed_image = processed.permute(1, 2, 0).cpu().numpy()
    processed_image = (processed_image - processed_image.min()) / (processed_image.max() - processed_image.min() + 1e-8)
    return (processed_image * 255).astype("uint8")


def generate_heatmap_for_image(image_path: Path, output_dir: Path, classification_service, explainability_service) -> None:
    """Generate a Grad-CAM heatmap for a single image and save it to disk."""
    image = cv2.imread(str(image_path))
    if image is None:
        LOGGER.info("Skipping invalid image: %s", image_path)
        return

    processed_image = prepare_image_for_gradcam(image)
    predicted = classification_service.predict(processed_image)
    output_path = output_dir / f"{image_path.stem}_gradcam.png"
    heatmap = explainability_service.generate(
        image=processed_image,
        class_id=predicted["class_id"],
        save_path=str(output_path),
    )

    if heatmap is not None:
        LOGGER.info("Saved GradCAM for %s -> %s", image_path.name, output_path)
    else:
        LOGGER.info("GradCAM generation failed for %s", image_path.name)


def main() -> None:
    """CLI entry point for generating Grad-CAM heatmaps."""
    args = parse_args()
    image_paths = load_images(args.images)

    if not image_paths:
        raise RuntimeError("No valid image files found for GradCAM generation.")

    classification_service = ClassificationService(checkpoint_path=str(args.checkpoint) if args.checkpoint else None)
    explainability_service = ExplainabilityService(model=classification_service.model)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for image_path in image_paths:
        generate_heatmap_for_image(image_path, args.output_dir, classification_service, explainability_service)


if __name__ == "__main__":
    main()
