"""Prepare CBIS-DDSM annotation CSVs from metadata and indexed image paths.

The script builds lookup tables once, resolves rows through cached lookups,
and emits train/validation CSV files in the standard ``image_path,label``
format.
"""

from __future__ import annotations

import argparse
import logging
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
PREFERRED_IMAGE_COLUMNS = [
    "image file path",
    "cropped image file path",
    "roi mask file path",
    "cropped image file",
    "roi_mask_file_path",
]
PREFERRED_LABEL_COLUMNS = [
    "pathology",
    "pathology_status",
    "pathology description",
    "pathology_type",
    "biopsy result",
    "biopsy_result",
    "assessment",
    "assessment code",
]


def configure_logging(log_path: Path) -> None:
    """Configure INFO-level logging for stdout and the preparation log file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(str(log_path), encoding="utf-8")],
        force=True,
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for dataset preparation."""
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

    parser.add_argument(
        "--debug-resolver",
        "--debug_resolver",
        dest="debug_resolver",
        action="store_true",
        help="Print resolver debug information for first rows (diagnostics).",
    )

    return parser.parse_args()


def normalize_text(value: Optional[str]) -> str:
    """Normalize whitespace and casing for metadata text comparisons."""
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).strip().lower())


def infer_label_from_text(value: Optional[str]) -> Optional[int]:
    """Map pathology or assessment text to a binary label.

    Returns 1 for malignant, 0 for benign or benign_without_callback, and None if unknown.
    """
    if value is None:
        return None

    normalized_value = normalize_text(value)
    if "malignant" in normalized_value:
        return 1
    if "benign_without_callback" in normalized_value or "benign without callback" in normalized_value:
        return 0
    if "benign" in normalized_value:
        return 0
    return None


class CBISDDMSplitter:
    """Prepare CBIS-DDSM train/val CSVs using one-time image indexing.

    The splitter builds lookup tables once and then resolves metadata rows
    through direct dictionary access instead of repeated filesystem scanning.
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
        # Build the image lookup tables once.
        self.name_map, self.rel_map, self.uid_map, self.stem_map = self._index_images()
        self._uid_cache = {}
        for uid, files in self.uid_map.items():
            for file_list in files.values():
                if file_list:
                    sorted_files = sorted(file_list, key=lambda p: p.as_posix())
                    self._uid_cache[uid] = sorted_files[0]
                    break
        # CLI-controlled diagnostic mode.
        self.debug = False

    def _ensure_roots(self) -> None:
        if not self.csv_root.exists():
            raise FileNotFoundError(f"CSV directory not found: {self.csv_root}")
        if not self.jpeg_root.exists():
            raise FileNotFoundError(f"JPEG directory not found: {self.jpeg_root}")

    def _index_images(self) -> Tuple[Dict[str, List[Path]], Dict[str, Path], Dict[str, Dict[str, List[Path]]], Dict[str, List[Path]]]:
        """Index all JPEG images under `jpeg_root`.

        Returns:
            name_map: filename (lowercase) -> list of absolute Paths
            rel_map: relative path under jpeg_root (posix, lowercase) -> Path
        """
        name_map: Dict[str, List[Path]] = {}
        rel_map: Dict[str, Path] = {}
        # uid_map: first relative directory -> filename -> list[Path]
        uid_map: Dict[str, Dict[str, List[Path]]] = {}
        # stem_map: filename stem -> list[Path]
        stem_map: Dict[str, List[Path]] = {}
        self.logger.info("Indexing JPEG images under %s", self.jpeg_root)
        count = 0
        for path in self.jpeg_root.rglob("*"):
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                count += 1
                name = path.name.lower()
                name_map.setdefault(name, []).append(path)
                stem = path.stem.lower()
                stem_map.setdefault(stem, []).append(path)
                try:
                    rel = path.relative_to(self.jpeg_root).as_posix().lower()
                except Exception:
                    rel = path.as_posix().lower()
                rel_map[rel] = path
                # uid part (first directory under jpeg/), if present
                parts = rel.split("/")
                if len(parts) > 1:
                    uid = parts[0]
                    uid_map.setdefault(uid, {}).setdefault(name, []).append(path)

        for name in list(name_map):
            name_map[name] = sorted(name_map[name], key=lambda p: p.as_posix())
        for stem in list(stem_map):
            stem_map[stem] = sorted(stem_map[stem], key=lambda p: p.as_posix())
        for uid in uid_map:
            for name in list(uid_map[uid]):
                uid_map[uid][name] = sorted(uid_map[uid][name], key=lambda p: p.as_posix())

        self.logger.info("Indexed %d image files", count)
        return name_map, rel_map, uid_map, stem_map

    def _read_csv_files(self, filenames: List[str]) -> pd.DataFrame:
        """Load and concatenate metadata CSV files when available."""
        rows: List[pd.DataFrame] = []
        for filename in filenames:
            csv_path = self.csv_root / filename
            if not csv_path.exists():
                self.logger.warning("Missing metadata file, skipping: %s", csv_path)
                continue
            try:
                dataframe = pd.read_csv(csv_path, low_memory=False)
                rows.append(dataframe)
                self.logger.info("Loaded %s (%d rows)", csv_path.name, len(dataframe))
            except Exception as exc:
                self.logger.warning("Failed to read %s: %s", csv_path, exc)
        if not rows:
            return pd.DataFrame()
        return pd.concat(rows, ignore_index=True)

    def _extract_image_candidates_from_value(self, value: str) -> List[str]:
        """Produce a short list of candidate image keys for fast lookup."""
        raw_value = str(value).strip()
        if not raw_value:
            return []

        candidates: List[str] = []
        normalized_value = raw_value.replace("\\", "/")

        if normalized_value.lower().startswith("jpeg/"):
            relative_path = normalized_value[len("jpeg/"):].lstrip("/")
            candidates.append(relative_path.lower())

        if "/" in normalized_value:
            candidates.append(normalized_value.lstrip("/").lower())
            candidates.append(Path(normalized_value).name.lower())
        else:
            candidates.append(normalized_value.lower())

        if not any(raw_value.lower().endswith(ext) for ext in [".jpg", ".jpeg"]):
            candidates.extend([raw_value.lower() + ".jpg", raw_value.lower() + ".jpeg"])  # type: ignore[arg-type]

        seen_candidates = set()
        unique_candidates = []
        for candidate in candidates:
            if candidate and candidate not in seen_candidates:
                seen_candidates.add(candidate)
                unique_candidates.append(candidate)
        return unique_candidates

    def _resolve_with_reason(self, value: str) -> Tuple[Optional[Path], str, Optional[str]]:
        """Resolve image and return (path, reason, matched_candidate).

        Tries direct candidates first, then a small set of transformed candidates
        that handle common CBIS-DDSM naming differences. All lookups use dicts
        built at indexing time (O(1) per candidate).
        """
        if not isinstance(value, str) or not value.strip():
            return None, "empty_value", None

        vraw = value.strip()
        vpos = vraw.replace("\\", "/").lstrip("/").lower()

        # ----- Fast-path for CBIS-DDSM metadata -----
        parts = vpos.split("/")

        # Metadata format:
        # Mass-Training.../UID1/UID2/000000.dcm
        # JPEG folder is UID2
        if len(parts) >= 4 and parts[-1] == "000000.dcm":
            uid = parts[-2]
            cached_path = self._uid_cache.get(uid)
            if cached_path is not None:
                return cached_path, "uid fast-path", uid
        # --------------------------------------------

        tested = []
        # primary candidates
        candidates = self._extract_image_candidates_from_value(vraw)
        for cand in candidates:
            tested.append(cand)
            if cand in self.rel_map:
                return self.rel_map[cand], f"rel_map exact match for '{cand}'", cand
            if cand in self.name_map:
                paths = self.name_map[cand]
                if len(paths) == 1:
                    return paths[0], f"unique name_map match for '{cand}'", cand
                # try uid_map disambiguation
                if "/" in vpos:
                    uid = vpos.split("/")[0]
                    fname = Path(vpos).name.lower()
                    if uid in self.uid_map and fname in self.uid_map[uid]:
                        return self.uid_map[uid][fname][0], f"uid_map disambiguation (uid={uid}) for '{cand}'", cand
                # try stem_map unique
                stem = Path(cand).stem.lower()
                if stem in self.stem_map and len(self.stem_map[stem]) == 1:
                    return self.stem_map[stem][0], f"stem_map unique for stem '{stem}'", cand
                # fallback to first
                return paths[0], f"fallback first name_map entry for '{cand}'", cand

        # secondary transformed candidates (automatic fixes)
        transforms: List[str] = []
        # if absolute path contains 'jpeg/', extract rel part
        if "jpeg/" in vpos:
            idx = vpos.find("jpeg/")
            rel = vpos[idx + len("jpeg/"):]
            transforms.append(rel)
        # replace dicom extension with .jpg/.jpeg
        if vpos.endswith(".dcm"):
            stem = vpos.rsplit('.', 1)[0]
            transforms.extend([stem + ".jpg", stem + ".jpeg"])  # type: ignore[arg-type]
        # try last two path components (uid/filename)
        parts = vpos.split("/")
        if len(parts) >= 2:
            last_two = "/".join(parts[-2:])
            transforms.append(last_two)
        # try filename without leading numbers/prefix like '1-' or '2-'
        filename = Path(vpos).name
        filename_noprefix = re.sub(r'^\d+[-_.]', '', filename)
        if filename_noprefix != filename:
            transforms.append(filename_noprefix)
        # try underscore/hyphen swap
        if "_" in filename:
            transforms.append(filename.replace("_", "-"))
        if "-" in filename:
            transforms.append(filename.replace("-", "_"))

        # dedupe transforms
        seen_t = set()
        tlist = []
        for t in transforms:
            t = t.lower()
            if t and t not in seen_t:
                seen_t.add(t)
                tlist.append(t)

        for cand in tlist:
            tested.append(cand)
            if cand in self.rel_map:
                return self.rel_map[cand], f"transformed rel_map match for '{cand}'", cand
            if cand in self.name_map:
                paths = self.name_map[cand]
                if len(paths) == 1:
                    return paths[0], f"transformed unique name_map match for '{cand}'", cand
                # try uid disambiguation
                if "/" in cand:
                    uid = cand.split("/")[0]
                    fname = Path(cand).name.lower()
                    if uid in self.uid_map and fname in self.uid_map[uid]:
                        return self.uid_map[uid][fname][0], f"transformed uid_map disambiguation (uid={uid}) for '{cand}'", cand
                stem = Path(cand).stem.lower()
                if stem in self.stem_map and len(self.stem_map[stem]) == 1:
                    return self.stem_map[stem][0], f"transformed stem_map unique for stem '{stem}'", cand
                return paths[0], f"transformed fallback first name_map for '{cand}'", cand

        return None, f"no match after testing candidates: {tested}", None

    def _resolve_image(self, value: str) -> Optional[Path]:
        """Resolve a metadata image reference value to an absolute JPEG Path using indices.

        Uses direct dictionary lookups only (no scanning of all images).
        """
        p, reason, cand = self._resolve_with_reason(value)
        if p is not None:
            return p
        return None

    def _iter_row_values(self, row):
        """Yield column/value pairs for a metadata row."""
        if hasattr(row, "_fields"):
            for field_name in row._fields:
                yield field_name, getattr(row, field_name)
        else:
            for field_name in row.index:
                yield field_name, row.get(field_name)

    def _normalize_column_name(self, column_name: object) -> str:
        """Normalize a column name into a lowercase, whitespace-safe form."""
        if not isinstance(column_name, str):
            return ""
        return column_name.lower().replace("_", " ").strip()

    def _find_first_text_value(
        self,
        row,
        preferred_column_names: List[str],
        fallback_keywords: Tuple[str, ...] = (),
        excluded_keywords: Tuple[str, ...] = (),
    ) -> Optional[str]:
        """Find the first non-empty text value using preferred names and keyword fallbacks."""
        for column_name, value in self._iter_row_values(row):
            if not isinstance(column_name, str):
                continue
            normalized_name = self._normalize_column_name(column_name)
            if normalized_name in preferred_column_names and isinstance(value, str) and value.strip():
                return value.strip()

        for column_name, value in self._iter_row_values(row):
            if not isinstance(column_name, str):
                continue
            normalized_name = self._normalize_column_name(column_name)
            if not isinstance(value, str) or not value.strip():
                continue
            if any(keyword in normalized_name for keyword in fallback_keywords) and not any(
                excluded_keyword in normalized_name for excluded_keyword in excluded_keywords
            ):
                return value.strip()
        return None

    def _infer_image_and_label_from_row(self, row) -> Optional[Tuple[Path, int]]:
        """Extract the image reference and binary label from a metadata row."""
        image_value = self._find_first_text_value(
            row,
            preferred_column_names=PREFERRED_IMAGE_COLUMNS,
            fallback_keywords=("image", "file", "filepath"),
            excluded_keywords=(),
        )

        if not image_value:
            if self.debug:
                self._debug_fail_count = getattr(self, "_debug_fail_count", 0)
                if self._debug_fail_count < 20:
                    self.logger.info("FAIL [NO IMAGE] row missing image reference")
                    self._debug_fail_count += 1
            return None

        label_value = self._find_first_text_value(
            row,
            preferred_column_names=PREFERRED_LABEL_COLUMNS,
            fallback_keywords=("path", "biopsy", "assessment"),
            excluded_keywords=("image", "file"),
        )

        label = infer_label_from_text(label_value)
        if label is None:
            if self.debug:
                self._debug_fail_count = getattr(self, "_debug_fail_count", 0)
                if self._debug_fail_count < 20:
                    self.logger.info("FAIL [LABEL] label_value=%s", label_value)
                    self._debug_fail_count += 1
            return None

        img_path = self._resolve_image(image_value)
        if self.debug:
            candidates = self._extract_image_candidates_from_value(image_value)
            self.logger.info("Resolver debug for metadata: %s", image_value)
            for cand in candidates:
                reason = ""
                matched = None
                if cand in self.rel_map:
                    matched = self.rel_map[cand]
                    reason = "exact rel_map match"
                elif cand in self.name_map:
                    paths = self.name_map[cand]
                    if len(paths) == 1:
                        matched = paths[0]
                        reason = "unique name_map match"
                    else:
                        # attempt uid disambiguation
                        vpos = image_value.replace("\\", "/").lstrip("/").lower()
                        if "/" in vpos:
                            vparts = vpos.split("/")
                            uid = vparts[0]
                            fname = Path(vpos).name.lower()
                            if uid in self.uid_map and fname in self.uid_map[uid]:
                                matched = self.uid_map[uid][fname][0]
                                reason = f"uid_map disambiguation (uid={uid})"
                        if matched is None:
                            stem = Path(cand).stem.lower()
                            if stem in self.stem_map and len(self.stem_map[stem]) == 1:
                                matched = self.stem_map[stem][0]
                                reason = "stem_map unique"
                            else:
                                reason = "multiple name_map entries; fallback to first"
                                matched = paths[0]
                else:
                    reason = "no match in rel_map or name_map"

                self.logger.info("  candidate=%s -> matched=%s reason=%s", cand, str(matched) if matched is not None else None, reason)

        if img_path is None:
            if self.debug:
                self._debug_fail_count = getattr(self, "_debug_fail_count", 0)
                if self._debug_fail_count < 20:
                    self.logger.info("FAIL [RESOLVER] image=%s", image_value)
                    self.logger.debug("Could not resolve image for metadata value: %s", image_value)
                    self._debug_fail_count += 1
            return None

        return img_path.resolve(), label

    def prepare(self) -> Dict[str, Path]:
        """Generate the train/validation CSV files in the configured output directory."""
        self.logger.info("Preparing dataset from %s", self.dataset_root)

        train_meta = self._read_csv_files(TRAIN_CSV_FILENAMES)
        test_meta = self._read_csv_files(TEST_CSV_FILENAMES)

        self.logger.info("Starting to build examples...")

        if self.debug:
            self.logger.info("=== SINGLE ROW DEBUG TEST ===")
            row = train_meta.iloc[0]
            res = self._infer_image_and_label_from_row(row)
            self.logger.info("TEST RESULT: %s", res)

        # Diagnostic pass is optional and only affects logging output.
        if self.debug:
            self.logger.info("--- Debug: first 10 metadata image identifiers ---")
            meta_ids = []
            for _, row in train_meta.iterrows():
                imgv = None
                for col in row.index:
                    if isinstance(col, str) and col.lower().strip() in ("image file path", "cropped image file path", "roi mask file path"):
                        val = row.get(col)
                        if isinstance(val, str) and val.strip():
                            imgv = val.strip()
                            break
                if not imgv:
                    for col in row.index:
                        if isinstance(col, str) and ("image" in col.lower() or "file" in col.lower() or "filepath" in col.lower()):
                            val = row.get(col)
                            if isinstance(val, str) and val.strip():
                                imgv = val.strip()
                                break
                if imgv:
                    meta_ids.append(imgv)
                if len(meta_ids) >= 10:
                    break
            for mid in meta_ids:
                self.logger.info("  %s", mid)

            self.logger.info("--- Debug: first 10 indexed rel_map keys ---")
            for k in list(self.rel_map.keys())[:10]:
                self.logger.info("  %s", k)
            self.logger.info("--- Debug: first 10 indexed filename keys ---")
            for k in list(self.name_map.keys())[:10]:
                self.logger.info("  %s", k)

            total_rows = 0
            resolved_rows = 0
            failed_rows = 0
            examples = []
            for _, row in train_meta.iterrows():
                total_rows += 1
                def _get_field(r, name):
                    for col in r.index:
                        if isinstance(col, str) and col.lower().strip() == name:
                            v = r.get(col)
                            if isinstance(v, str) and v.strip():
                                return v.strip()
                    return None

                image_fp = _get_field(row, "image file path")
                cropped_fp = _get_field(row, "cropped image file path")
                roi_fp = _get_field(row, "roi mask file path")
                if not image_fp:
                    for col in row.index:
                        if isinstance(col, str) and ("image" in col.lower() or "file" in col.lower() or "filepath" in col.lower()):
                            val = row.get(col)
                            if isinstance(val, str) and val.strip():
                                image_fp = val.strip()
                                break

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

                # normalized lookup key
                normalized_key = None
                if image_fp:
                    candlist = self._extract_image_candidates_from_value(image_fp)
                    normalized_key = candlist[0] if candlist else None

                matched_path = None
                reason = None
                if image_fp:
                    p, reason, cand = self._resolve_with_reason(image_fp)
                    matched_path = str(p) if p is not None else None

                if matched_path is not None and label is not None:
                    examples.append((str(p.resolve()), label))
                    resolved_rows += 1
                else:
                    failed_rows += 1

                if total_rows <= 20:
                    self.logger.info("--- Metadata row %d diagnostics ---", total_rows)
                    self.logger.info("  image_file_path: %s", image_fp)
                    self.logger.info("  cropped_image_file_path: %s", cropped_fp)
                    self.logger.info("  roi_mask_file_path: %s", roi_fp)
                    self.logger.info("  normalized_lookup_key: %s", normalized_key)
                    self.logger.info("  label_value: %s -> mapped_label: %s", label_value, label)
                    self.logger.info("  matched: %s", matched_path)
                    self.logger.info("  reason: %s", reason)

            # Summary
            self.logger.info("--- Resolver diagnostics summary ---")
            self.logger.info("  total metadata rows: %d", total_rows)
            self.logger.info("  resolved rows: %d", resolved_rows)
            self.logger.info("  failed rows: %d", failed_rows)
            pct = (resolved_rows / total_rows * 100.0) if total_rows else 0.0
            self.logger.info("  resolution percentage: %.2f%%", pct)
            examples = list({(p, l) for p, l in examples})

        if train_meta.empty and test_meta.empty:
            raise RuntimeError("No metadata CSVs could be loaded from csv/ directory.")

        examples: List[Tuple[str, int]] = []
        seen_examples = set()
        processed_rows = len(train_meta)
        resolved_rows = 0

        # Build examples from training metadata using row-wise lookups only.
        for i, (_, row) in enumerate(train_meta.iterrows()):
            if i % 1000 == 0:
                self.logger.info("Processed %d rows", i)
            res = self._infer_image_and_label_from_row(row)
            if self.debug and i < 20:
                self.logger.info("ROW %d RESULT = %s", i, res)
            if res is None:
                continue
            resolved_rows += 1
            img_path, label = res
            example_key = (str(img_path), label)
            if example_key in seen_examples:
                continue
            seen_examples.add(example_key)
            examples.append((str(img_path), label))

        if not examples:
            raise RuntimeError("No valid examples were found in training metadata.")

        self.logger.info("Total rows processed: %d", processed_rows)
        self.logger.info("Resolved rows: %d", resolved_rows)
        self.logger.info("Valid samples collected: %d", len(examples))
        self.logger.info("Skipped samples: %d", processed_rows - resolved_rows)
        if self.debug:
            for i, sample in enumerate(examples[:10]):
                self.logger.info("Sample %d: %s", i, sample)
        assert examples, "No valid examples found - UID mapping still broken"

        df = pd.DataFrame(examples, columns=["image_path", "label"])

        # Basic validations.
        before = len(df)
        df = df.drop_duplicates(subset=["image_path", "label"]).reset_index(drop=True)
        after = len(df)
        if before != after:
            self.logger.info("Dropped %d duplicate rows", before - after)

        if df["label"].isnull().any():
            raise RuntimeError("Missing labels detected in prepared examples.")

        missing = [p for p in df["image_path"] if not Path(p).is_file()]
        if missing:
            raise RuntimeError(f"Some images are missing from disk (first example): {missing[0]}")

        n_classes = len(df["label"].unique())
        if n_classes < 2:
            raise RuntimeError("Need at least two classes to perform stratified split.")

        test_size = self.val_size
        if isinstance(self.val_size, float):
            min_test_samples = max(1, n_classes)
            requested_test_samples = math.ceil(self.val_size * len(df))
            if requested_test_samples < min_test_samples:
                test_size = min_test_samples / len(df)
                self.logger.warning(
                    "Adjusted validation split from %.3f to %.3f to keep stratification valid for %d classes.",
                    self.val_size,
                    test_size,
                    n_classes,
                )

        train_df, val_df = train_test_split(
            df, test_size=test_size, stratify=df["label"], random_state=42
        )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        train_out = self.output_dir / "train.csv"
        val_out = self.output_dir / "val.csv"
        train_df.to_csv(train_out, index=False, columns=["image_path", "label"])
        val_df.to_csv(val_out, index=False, columns=["image_path", "label"])

        train_dist = train_df["label"].value_counts().to_dict()
        val_dist = val_df["label"].value_counts().to_dict()
        self.logger.info("Train distribution: %s", train_dist)
        self.logger.info("Val distribution: %s", val_dist)
        self.logger.info("Train samples: %d", len(train_df))
        self.logger.info("Val samples: %d", len(val_df))

        return {"train_csv": train_out, "val_csv": val_out}


def main() -> None:
    """Run dataset preparation from the command line."""
    args = parse_args()
    splitter = CBISDDMSplitter(dataset_root=args.dataset_root, output_dir=args.output_dir, val_size=args.val_size)
    splitter.debug = getattr(args, 'debug_resolver', False)
    result = splitter.prepare()

    print("Dataset preparation summary:")
    print(f"  Train CSV: {result['train_csv']}")
    print(f"  Val CSV: {result['val_csv']}")


if __name__ == "__main__":
    main()
