import json
import os
import sys
from pathlib import Path

# Make the repo importable when running from Kaggle
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Allow Kaggle environment variables to override dataset and output paths.
# Example:
#   DATASET_ROOT=/kaggle/input/your-dataset-name
#   OUTPUT_ROOT=/kaggle/working
os.environ.setdefault("DATASET_ROOT", os.getenv("DATASET_ROOT", ""))
os.environ.setdefault("OUTPUT_ROOT", os.getenv("OUTPUT_ROOT", ""))

from pipelines.training_pipeline import run_kaggle_training


def main() -> None:
    dataset_root = os.getenv("DATASET_ROOT", "").strip()
    output_root = os.getenv("OUTPUT_ROOT", "").strip()

    if not dataset_root:
        raise RuntimeError(
            "Please set DATASET_ROOT to your Kaggle dataset folder, e.g. /kaggle/input/your-dataset"
        )

    if not output_root:
        output_root = "/kaggle/working"
        os.environ["OUTPUT_ROOT"] = output_root

    train_csv = str(Path(dataset_root) / "annotations" / "train.csv")
    val_csv = str(Path(dataset_root) / "annotations" / "val.csv")

    if not Path(train_csv).exists():
        raise FileNotFoundError(f"Training CSV not found: {train_csv}")
    if not Path(val_csv).exists():
        raise FileNotFoundError(f"Validation CSV not found: {val_csv}")

    results = run_kaggle_training(
        train_csv=train_csv,
        val_csv=val_csv,
        model_name="efficientnet_b0",
    )

    output_dir = Path(output_root) / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "training_summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("Training complete.")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
