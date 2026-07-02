from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

from pipelines.evaluation_pipeline import EvaluationPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained breast cancer classification model.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint path to load the classification model from.",
    )
    parser.add_argument(
        "--test-csv",
        type=Path,
        required=True,
        help="CSV file containing image_path,label for evaluation.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Optional TIMM model name to build architecture.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pipeline = EvaluationPipeline(
        checkpoint_path=str(args.checkpoint) if args.checkpoint else None,
        model_name=args.model_name,
    )

    results = pipeline.evaluate(
        test_csv=str(args.test_csv),
        save_prefix=args.test_csv.stem,
    )

    print("Evaluation complete.")
    print(results)


if __name__ == "__main__":
    main()
