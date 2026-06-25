from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


class DataUtils:

    IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]

    @staticmethod
    def load_annotations(csv_path: str) -> pd.DataFrame:
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
        mask_dir: Optional[str] = None,
    ) -> Dict[str, int]:
        image_dir = Path(image_dir)

        if not image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")

        images = [
            path for path in image_dir.rglob("*")
            if path.suffix.lower() in cls.IMAGE_EXTENSIONS
        ]

        report = {"images_found": len(images)}

        if mask_dir is not None:
            mask_dir = Path(mask_dir)
            if not mask_dir.exists():
                raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

            masks = [
                path for path in mask_dir.rglob("*")
                if path.suffix.lower() in cls.IMAGE_EXTENSIONS
            ]
            report["masks_found"] = len(masks)

        return report

    @staticmethod
    def class_distribution(
        dataframe: pd.DataFrame,
        label_column: str = "label",
    ) -> Dict:
        if label_column not in dataframe.columns:
            raise ValueError(f"Label column '{label_column}' not found in dataframe.")

        return dataframe[label_column].value_counts().to_dict()

