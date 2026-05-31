from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


class DataUtils:

    @staticmethod
    def load_annotations(
        csv_path
    ):

        return pd.read_csv(
            csv_path
        )

    @staticmethod
    def create_splits(
        dataframe,
        test_size=0.15,
        val_size=0.15,
        random_state=42
    ):

        train_df, test_df = train_test_split(
            dataframe,
            test_size=test_size,
            stratify=dataframe["label"],
            random_state=random_state
        )

        train_df, val_df = train_test_split(
            train_df,
            test_size=(
                val_size /
                (1 - test_size)
            ),
            stratify=train_df["label"],
            random_state=random_state
        )

        return (
            train_df,
            val_df,
            test_df
        )

    @staticmethod
    def verify_dataset(
        image_dir,
        mask_dir=None
    ):

        image_dir = Path(image_dir)

        images = list(
            image_dir.rglob("*")
        )

        report = {
            "images_found": len(images)
        }

        if mask_dir:

            mask_dir = Path(mask_dir)

            masks = list(
                mask_dir.rglob("*")
            )

            report["masks_found"] = len(
                masks
            )

        return report

    @staticmethod
    def class_distribution(
        dataframe,
        label_column="label"
    ):

        return (
            dataframe[label_column]
            .value_counts()
            .to_dict()
        )

