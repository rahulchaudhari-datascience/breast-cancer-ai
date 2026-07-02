from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import cv2
import pandas as pd
from sklearn.model_selection import train_test_split

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
TRAIN_CSV_FILENAMES = [
    "mass_case_description_train_set.csv",
    "calc_case_description_train_set.csv",
]
TEST_CSV_FILENAMES = [
    "mass_case_description_test_set.csv",
    "calc_case_description_test_set.csv",
]
LABEL_INCLUDE_PATTERNS = {
    "benign": 0,
    "malignant": 1,
}
IGNORED_LABEL_PATTERNS = {"benign_without_callback", "other", "unknown"}
IMAGE_COLUMNS = [
    "image file path",
    "image_path",
    "image path",
    "image file",
    "image_file",
    "imagefilename",
    "image filename",
    "file_name",
    "file name",
    "filename",
    "image",
]
LABEL_COLUMNS = [
    "pathology",
    "pathology_status",
    "assessment",
    "assessment code",
    "pathology description",
    "pathology_type",
    "biopsy result",
    "biopsy_result",
]
DICOM_COLUMNS = [
    "image file path",
    "image_path",
    "dicom_path",
    "filepath",
    "file_path",
    "file",
    "imageid",
    "image_id",
    "fileid",
    "file_id",
]


def configure_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(str(log_path), encoding="utf-8"),
        ],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare CBIS-DDSM dataset annotations for breast cancer training."
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path.cwd(),
        help="Path to the CBIS-DDSM dataset root containing csv/ and jpeg/ directories.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd() / "datasets" / "annotations",
        help="Output directory for generated train/val annotation CSV files.",
    )

    parser.add_argument(
        "--val-size",
        type=float,
        default=0.15,
        help="Fraction of training data to reserve for validation.",
    )

    return parser.parse_args()


def normalize_text(value: Optional[str]) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).strip().lower())


def extract_image_path_candidates(value: str) -> List[str]:
    value = value.strip()
    if not value:
        return []

    candidates = [value]
    if "/" in value or "\\" in value:
        candidates.append(Path(value).name)
        candidates.append(Path(value).stem + Path(value).suffix.lower())
    if not any(value.lower().endswith(ext) for ext in SUPPORTED_IMAGE_EXTENSIONS):
        for ext in [".jpg", ".jpeg"]:
            candidates.append(value + ext)
    return list(dict.fromkeys(candidates))


def infer_label(label_value: Optional[str]) -> Optional[int]:
    if label_value is None:
        return None

    normalized = normalize_text(label_value)
    if any(ignored in normalized for ignored in IGNORED_LABEL_PATTERNS):
        return None

    for pattern, label in LABEL_INCLUDE_PATTERNS.items():
        if pattern in normalized:
            return label

    return None


class CBISDDMSplitter:
    def __init__(self, dataset_root: Path, output_dir: Path, val_size: float = 0.15):
        self.dataset_root = dataset_root.expanduser().resolve()
        self.output_dir = output_dir.expanduser().resolve()
        self.val_size = val_size
        self.csv_root = self.dataset_root / "csv"
        self.jpeg_root = self.dataset_root / "jpeg"
        self.log_path = self.output_dir / "prepare_cbis_ddsm_dataset.log"
        configure_logging(self.log_path)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.image_index = self._index_images()
        self.dicom_map = self._load_dicom_info()

    def _index_images(self) -> Dict[str, List[Path]]:
        if not self.jpeg_root.exists():
            raise FileNotFoundError(f"JPEG directory not found: {self.jpeg_root}")

        image_paths: Dict[str, List[Path]] = {}
        self.logger.info("Indexing JPEG image files from %s", self.jpeg_root)

        for path in self.jpeg_root.rglob("*"):
            if path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                key = path.name.lower()
                image_paths.setdefault(key, []).append(path)

        self.logger.info("Found %d image files", sum(len(v) for v in image_paths.values()))
        return image_paths

    def _load_dicom_info(self) -> Dict[str, Path]:
        dicom_csv = self.csv_root / "dicom_info.csv"
        mapping: Dict[str, Path] = {}

        if not dicom_csv.exists():
            self.logger.info("dicom_info.csv not found, proceeding without DICOM mapping.")
            return mapping

        self.logger.info("Loading DICOM metadata from %s", dicom_csv)
        try:
            df = pd.read_csv(dicom_csv, low_memory=False)
        except Exception as exc:
            self.logger.warning("Failed to read dicom_info.csv: %s", exc)
            return mapping

        column_map = {col.lower(): col for col in df.columns}
        candidate_columns = [column_map[col] for col in column_map if col in DICOM_COLUMNS]

        for _, row in df.iterrows():
            for col in candidate_columns:
                value = row.get(col)
                if not isinstance(value, str) or not value.strip():
                    continue
                for candidate in extract_image_path_candidates(value):
                    path = self._resolve_image_candidate(candidate)
                    if path is not None:
                        mapping[normalize_text(candidate)] = path
                        break

        self.logger.info("Loaded %d DICOM image mappings", len(mapping))
        return mapping

    def _resolve_image_candidate(self, candidate: str) -> Optional[Path]:
        candidate_normalized = normalize_text(candidate)
        if not candidate_normalized:
            return None

        candidate_path = Path(candidate)
        if candidate_path.is_absolute() and candidate_path.exists():
            return candidate_path

        if candidate_normalized in self.dicom_map:
            return self.dicom_map[candidate_normalized]

        if candidate_path.name.lower() in self.image_index:
            paths = self.image_index[candidate_path.name.lower()]
            if len(paths) == 1:
                return paths[0]
            for path in paths:
                if candidate_path.parent.name and candidate_path.parent.name.lower() in {part.lower() for part in path.parts}:
                    return path
                if candidate_path.stem.lower() == path.stem.lower() and path.suffix.lower() == candidate_path.suffix.lower():
                    return path

        candidate_lower = candidate.lower().replace("\\", "/")
        if candidate_lower.startswith("jpeg/"):
            candidate_lower = candidate_lower[len("jpeg/"):]

        for path_list in self.image_index.values():
            for image_path in path_list:
                image_relative = str(image_path.relative_to(self.jpeg_root)).replace("\\", "/").lower()
                if candidate_lower in image_relative:
                    return image_path

        return None

    def _find_image_path(self, image_value: str) -> Optional[Path]:
        if not isinstance(image_value, str) or not image_value.strip():
            return None

        for candidate in extract_image_path_candidates(image_value):
            path = self._resolve_image_candidate(candidate)
            if path is not None and path.exists():
                return path

        return None

    def _read_metadata_files(self, filenames: Iterable[str]) -> pd.DataFrame:
        rows: List[pd.DataFrame] = []
        for filename in filenames:
            path = self.csv_root / filename
            if not path.exists():
                self.logger.warning("Metadata file not found and will be skipped: %s", path)
                continue
            try:
                df = pd.read_csv(path, low_memory=False)
                rows.append(df)
                self.logger.info("Loaded metadata file: %s (%d rows)", path, len(df))
            except Exception as exc:
                self.logger.warning("Failed to read metadata file %s: %s", path, exc)
        if not rows:
            return pd.DataFrame()
        return pd.concat(rows, ignore_index=True)

    def _extract_image_column(self, row: pd.Series) -> Optional[str]:
        for col in IMAGE_COLUMNS:
            if col in row.index and isinstance(row[col], str) and row[col].strip():
                return row[col].strip()
        for col in row.index:
            if isinstance(col, str) and "image" in col.lower() and isinstance(row[col], str) and row[col].strip():
                return row[col].strip()
        return None

    def _extract_label_column(self, row: pd.Series) -> Optional[str]:
        for col in LABEL_COLUMNS:
            if col in row.index and isinstance(row[col], str) and row[col].strip():
                return row[col].strip()
        for col in row.index:
            if isinstance(col, str) and ("path" in col.lower() or "assess" in col.lower() or "biopsy" in col.lower()):
                value = row[col]
                if isinstance(value, str) and value.strip():
                    return value.strip()
        return None

    def _build_examples(self, dataframe: pd.DataFrame) -> List[Tuple[Path, int]]:
        examples: List[Tuple[Path, int]] = []
        seen: Set[Tuple[str, int]] = set()

        for _, row in dataframe.iterrows():
            image_value = self._extract_image_column(row)
            if not image_value:
                continue

            label_text = self._extract_label_column(row)
            label = infer_label(label_text)
            if label is None:
                continue

            image_path = self._find_image_path(image_value)
            if image_path is None or not image_path.exists():
                self.logger.debug("Skipping missing image for row: %s", image_value)
                continue

            if not self._validate_image(image_path):
                self.logger.warning("Skipping corrupted or unreadable image: %s", image_path)
                continue

            key = (str(image_path.resolve()), label)
            if key in seen:
                continue

            seen.add(key)
            examples.append((image_path.resolve(), label))

        return examples

    def _validate_image(self, image_path: Path) -> bool:
        if not image_path.exists():
            return False
        image = cv2.imread(str(image_path))
        return image is not None

    def _create_dataframe(self, examples: List[Tuple[Path, int]]) -> pd.DataFrame:
        return pd.DataFrame(
            [(str(path), label) for path, label in examples],
            columns=["image_path", "label"],
        )

    def _save_csv(self, dataframe: pd.DataFrame, filename: str) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / filename
        dataframe.to_csv(output_path, index=False)
        self.logger.info("Saved %s (%d rows)", output_path, len(dataframe))
        return output_path

    def prepare(self) -> Dict[str, Path]:
        self.logger.info("Preparing CBIS-DDSM annotations in %s", self.dataset_root)

        train_df = self._read_metadata_files(TRAIN_CSV_FILENAMES)
        test_df = self._read_metadata_files(TEST_CSV_FILENAMES)

        if train_df.empty and test_df.empty:
            raise RuntimeError("No CBIS-DDSM metadata files could be loaded.")

        train_examples = self._build_examples(train_df)
        test_examples = self._build_examples(test_df)

        if not train_examples:
            raise RuntimeError("No valid training metadata examples were found.")

        self.logger.info("Found %d valid train examples and %d valid test examples", len(train_examples), len(test_examples))

        train_df = self._create_dataframe(train_examples)

        if len(train_df["label"].unique()) < 2:
            raise RuntimeError("Training metadata does not contain at least two label classes.")

        train_df, val_df = train_test_split(
            train_df,
            test_size=self.val_size,
            stratify=train_df["label"],
            random_state=42,
        )

        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)

        self._save_csv(train_df, "train.csv")
        self._save_csv(val_df, "val.csv")

        if test_examples:
            test_df = self._create_dataframe(test_examples).reset_index(drop=True)
            self._save_csv(test_df, "test.csv")

        self.logger.info("Dataset preparation completed successfully.")
        self.logger.info("Train distribution: %s", self._distribution(train_df))
        self.logger.info("Val distribution: %s", self._distribution(val_df))

        return {
            "train_csv": self.output_dir / "train.csv",
            "val_csv": self.output_dir / "val.csv",
            "test_csv": self.output_dir / "test.csv" if test_examples else None,
        }

    @staticmethod
    def _distribution(dataframe: pd.DataFrame) -> Dict[str, int]:
        return dataframe["label"].value_counts().to_dict()


def main() -> None:
    args = parse_args()
    splitter = CBISDDMSplitter(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        val_size=args.val_size,
    )
    result = splitter.prepare()

    print("Dataset preparation summary:")
    print(f"  Train CSV: {result['train_csv']}")
    print(f"  Val CSV: {result['val_csv']}")
    if result["test_csv"]:
        print(f"  Test CSV: {result['test_csv']}")


if __name__ == "__main__":
    main()
