from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


def configure_logging(log_path: Path) -> None:
    """Configure logging to stdout and file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(str(log_path), encoding="utf-8")],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare CBIS-DDSM dataset annotations.")

    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
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


def infer_label_from_text(value: Optional[str]) -> Optional[int]:
    """Map pathology/assessment text to labels.

    Returns 1 for malignant, 0 for benign/benign_without_callback, and None if unknown/ignored.
    """
    if value is None:
        return None
    v = normalize_text(value)
    if "malignant" in v:
        return 1
    if "benign_without_callback" in v or "benign without callback" in v:
        return 0
    if "benign" in v:
        return 0
    return None


class CBISDDMSplitter:
    """Efficiently prepare CBIS-DDSM train/val CSVs using O(N) image indexing.

    The implementation indexes JPEG images once into hash maps and uses direct
    lookups (no nested scanning) when resolving image references from CSV metadata.
    """

    def __init__(self, dataset_root: Path, output_dir: Path, val_size: float = 0.15):
        self.dataset_root = dataset_root.expanduser().resolve()
        self.output_dir = output_dir.expanduser().resolve()
        self.val_size = val_size
        self.csv_root = self.dataset_root / "csv"
        self.jpeg_root = self.dataset_root / "jpeg"
        self.log_path = self.output_dir / "prepare_cbis_ddsm_dataset.log"
        configure_logging(self.log_path)
        self.logger = logging.getLogger(self.__class__.__name__)

        self._ensure_roots()
        # Build image indices once (O(N_images))
        self.name_map, self.rel_map = self._index_images()

    def _ensure_roots(self) -> None:
        if not self.csv_root.exists():
            raise FileNotFoundError(f"CSV directory not found: {self.csv_root}")
        if not self.jpeg_root.exists():
            raise FileNotFoundError(f"JPEG directory not found: {self.jpeg_root}")

    def _index_images(self) -> Tuple[Dict[str, List[Path]], Dict[str, Path]]:
        """Index all JPEG images under `jpeg_root`.

        Returns:
            name_map: mapping from filename (lowercase) -> list of absolute Paths
            rel_map: mapping from relative path under jpeg_root (posix, lowercase) -> Path
        """
        name_map: Dict[str, List[Path]] = {}
        rel_map: Dict[str, Path] = {}
        self.logger.info("Indexing JPEG images under %s", self.jpeg_root)
        count = 0
        for path in self.jpeg_root.rglob("*"):
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                count += 1
                name = path.name.lower()
                name_map.setdefault(name, []).append(path)
                try:
                    rel = path.relative_to(self.jpeg_root).as_posix().lower()
                except Exception:
                    rel = path.as_posix().lower()
                rel_map[rel] = path

        self.logger.info("Indexed %d image files", count)
        return name_map, rel_map

    def _read_csv_files(self, filenames: List[str]) -> pd.DataFrame:
        rows: List[pd.DataFrame] = []
        for fn in filenames:
            p = self.csv_root / fn
            if not p.exists():
                self.logger.warning("Missing metadata file, skipping: %s", p)
                continue
            try:
                df = pd.read_csv(p, low_memory=False)
                rows.append(df)
                self.logger.info("Loaded %s (%d rows)", p.name, len(df))
            except Exception as exc:
                self.logger.warning("Failed to read %s: %s", p, exc)
        if not rows:
            return pd.DataFrame()
        return pd.concat(rows, ignore_index=True)

    def _extract_image_candidates_from_value(self, value: str) -> List[str]:
        """Produce a short list of candidate keys (rel path and filename) for fast lookup."""
        v = str(value).strip()
        if not v:
            return []
        candidates: List[str] = []
        # normalize separators
        v_posix = v.replace("\\", "/")
        # If value already looks like a relative path under jpeg/, strip leading 'jpeg/' if present
        if v_posix.lower().startswith("jpeg/"):
            rel = v_posix[len("jpeg/"):].lstrip("/")
            candidates.append(rel.lower())
        # full relative path candidate
        if "/" in v_posix:
            candidates.append(v_posix.lstrip("/").lower())
            candidates.append(Path(v_posix).name.lower())
        else:
            candidates.append(v_posix.lower())
        # try adding common jpg extensions if missing
        if not any(v.lower().endswith(ext) for ext in [".jpg", ".jpeg"]):
            candidates.extend([v.lower() + ".jpg", v.lower() + ".jpeg"])  # type: ignore[arg-type]
        # dedupe while preserving order
        seen = set()
        out = []
        for c in candidates:
            if c and c not in seen:
                seen.add(c)
                out.append(c)
        return out

    def _resolve_image(self, value: str) -> Optional[Path]:
        """Resolve a metadata image reference value to an absolute JPEG Path using indices.

        Uses direct dictionary lookups only (no scanning of all images).
        """
        for cand in self._extract_image_candidates_from_value(value):
            # exact relative path match
            if cand in self.rel_map:
                return self.rel_map[cand]
            # filename match
            if cand in self.name_map:
                paths = self.name_map[cand]
                if len(paths) == 1:
                    return paths[0]
                # multiple matches: try to disambiguate by checking if candidate contains a folder prefix
                # (e.g., uid/filename)
                if "/" in value.replace("\\", "/"):
                    # attempt to match the relative parts
                    vparts = value.replace("\\", "/").lower().split("/")
                    for p in paths:
                        rel = p.relative_to(self.jpeg_root).as_posix().lower()
                        if all(part in rel for part in vparts if part):
                            return p
                # fallback to first path
                return paths[0]
        return None

    def _infer_image_and_label_from_row(self, row: pd.Series) -> Optional[Tuple[Path, int]]:
        """Extract image reference and label from a metadata row.

        The function inspects common metadata columns without performing expensive
        image scans.
        """
        # possible image columns — look for any column name containing 'image' or 'file'
        image_value = None
        for col in row.index:
            if not isinstance(col, str):
                continue
            if "image" in col.lower() or "file" in col.lower() or "filepath" in col.lower():
                val = row.get(col)
                if isinstance(val, str) and val.strip():
                    image_value = val.strip()
                    break

        if not image_value:
            return None

        # infer label from common columns. Prefer explicit pathology/biopsy/assessment
        # columns and avoid selecting image/file columns that also contain 'path'.
        label_value = None
        preferred_label_cols = {
            "pathology",
            "pathology_status",
            "pathology description",
            "pathology_type",
            "biopsy result",
            "biopsy_result",
            "assessment",
            "assessment code",
        }

        for col in row.index:
            if not isinstance(col, str):
                continue
            lname = col.lower().strip()
            if lname in preferred_label_cols:
                val = row.get(col)
                if isinstance(val, str) and val.strip():
                    label_value = val.strip()
                    break

        # fallback: search for columns containing path/biopsy/assessment but skip image/file columns
        if label_value is None:
            for col in row.index:
                if not isinstance(col, str):
                    continue
                lname = col.lower()
                if any(k in lname for k in ("path", "biopsy", "assessment")) and "image" not in lname and "file" not in lname:
                    val = row.get(col)
                    if isinstance(val, str) and val.strip():
                        label_value = val.strip()
                        break

        label = infer_label_from_text(label_value)
        if label is None:
            return None

        img_path = self._resolve_image(image_value)
        if img_path is None:
            self.logger.debug("Could not resolve image for metadata value: %s", image_value)
            return None

        return img_path.resolve(), label

    def _validate_image(self, p: Path) -> bool:
        if not p.exists():
            return False
        try:
            img = cv2.imread(str(p))
            return img is not None
        except Exception:
            return False

    def prepare(self) -> Dict[str, Path]:
        """Main entrypoint: generate train/val CSVs at the configured output directory."""
        self.logger.info("Preparing dataset from %s", self.dataset_root)

        train_meta = self._read_csv_files(TRAIN_CSV_FILENAMES)
        test_meta = self._read_csv_files(TEST_CSV_FILENAMES)

        if train_meta.empty and test_meta.empty:
            raise RuntimeError("No metadata CSVs could be loaded from csv/ directory.")

        examples: List[Tuple[str, int]] = []
        seen = set()

        # build examples from training metadata (O(N_rows) with O(1) image lookups)
        for _, row in train_meta.iterrows():
            res = self._infer_image_and_label_from_row(row)
            if res is None:
                continue
            img_path, label = res
            if not img_path.exists():
                self.logger.debug("Image does not exist, skipping: %s", img_path)
                continue
            if not self._validate_image(img_path):
                self.logger.warning("Unreadable image, skipping: %s", img_path)
                continue
            key = (str(img_path), label)
            if key in seen:
                continue
            seen.add(key)
            examples.append((str(img_path), label))

        if not examples:
            raise RuntimeError("No valid examples were found in training metadata.")

        df = pd.DataFrame(examples, columns=["image_path", "label"])

        # Basic validations
        # 1) Check duplicates
        before = len(df)
        df = df.drop_duplicates()
        after = len(df)
        if before != after:
            self.logger.info("Dropped %d duplicate rows", before - after)

        # 2) Missing labels
        if df["label"].isnull().any():
            raise RuntimeError("Missing labels detected in prepared examples.")

        # 3) Ensure all image files exist
        missing = [p for p in df["image_path"] if not Path(p).exists()]
        if missing:
            raise RuntimeError(f"Some images are missing from disk (first example): {missing[0]}")

        # Stratified split
        if len(df["label"].unique()) < 2:
            raise RuntimeError("Need at least two classes to perform stratified split.")

        train_df, val_df = train_test_split(
            df, test_size=self.val_size, stratify=df["label"], random_state=42
        )

        # Ensure required output format: image_path,label
        self.output_dir.mkdir(parents=True, exist_ok=True)
        train_out = self.output_dir / "train.csv"
        val_out = self.output_dir / "val.csv"
        train_df.to_csv(train_out, index=False, columns=["image_path", "label"])
        val_df.to_csv(val_out, index=False, columns=["image_path", "label"])

        # Print/Log distributions and counts
        train_dist = train_df["label"].value_counts().to_dict()
        val_dist = val_df["label"].value_counts().to_dict()
        self.logger.info("Train distribution: %s", train_dist)
        self.logger.info("Val distribution: %s", val_dist)
        self.logger.info("Train samples: %d", len(train_df))
        self.logger.info("Val samples: %d", len(val_df))

        return {"train_csv": train_out, "val_csv": val_out}


def main() -> None:
    args = parse_args()
    splitter = CBISDDMSplitter(dataset_root=args.dataset_root, output_dir=args.output_dir, val_size=args.val_size)
    result = splitter.prepare()

    print("Dataset preparation summary:")
    print(f"  Train CSV: {result['train_csv']}")
    print(f"  Val CSV: {result['val_csv']}")


if __name__ == "__main__":
    main()
