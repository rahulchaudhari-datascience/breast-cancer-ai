"""Dataset-oriented helper utilities."""

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


class DataUtils:
    """Utility collection for dataset loading, splitting, and validation."""

    SUPPORTED_IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
    IMAGE_EXTENSIONS = SUPPORTED_IMAGE_EXTENSIONS

    @staticmethod
    def load_annotations(csv_path: str) -> pd.DataFrame:
        """Load a CBIS-DDSM-style annotation CSV and validate required columns."""
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"Annotation CSV not found: {csv_path}")

        df = pd.read_csv(path)
        required = {"image_path", "label"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Annotation CSV missing columns: {', '.join(sorted(missing))}")

        return df

    @staticmethod
    def create_splits(
        dataframe: pd.DataFrame,
        test_size: float = 0.15,
        val_size: float = 0.15,
        random_state: int = 42,
        label_column: str = "label",
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create stratified train/validation/test splits."""
        if label_column not in dataframe.columns:
            raise ValueError(f"Label column '{label_column}' not found in dataframe.")

        if dataframe[label_column].nunique() < 2:
            raise ValueError("Dataframe must contain at least two label classes for stratified split.")

        train_df, test_df = train_test_split(
            dataframe,
            test_size=test_size,
            stratify=dataframe[label_column],
            random_state=random_state,
        )

        train_df, val_df = train_test_split(
            train_df,
            test_size=(val_size / (1 - test_size)),
            stratify=train_df[label_column],
            random_state=random_state,
        )

        return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)

    @classmethod
    def verify_dataset(
        cls,
        image_dir: str,
    ) -> Dict[str, int]:
        """Count supported image artifacts present in the image directory."""
        image_dir = Path(image_dir)

        if not image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")

        images = [
            path for path in image_dir.rglob("*")
            if path.suffix.lower() in cls.IMAGE_EXTENSIONS
        ]

        return {"images_found": len(images)}

    @staticmethod
    def class_distribution(
        dataframe: pd.DataFrame,
        label_column: str = "label",
    ) -> Dict:
        """Return class counts for the configured label column."""
        if label_column not in dataframe.columns:
            raise ValueError(f"Label column '{label_column}' not found in dataframe.")

        return dataframe[label_column].value_counts().to_dict()

