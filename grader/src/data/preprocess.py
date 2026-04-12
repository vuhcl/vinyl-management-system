"""
grader/src/data/preprocess.py

Text preprocessing pipeline for the vinyl condition grader.
Reads unified.jsonl, applies text normalization and abbreviation
expansion, detects unverified media signals, assigns train/val/test
splits using adaptive stratification, and writes output JSONL files.

Transformation order (strictly enforced):
  1. Detect unverified media signals  — on raw text
  2. Detect Generic sleeve signals    — on raw text
  3. Lowercase
  4. Normalize whitespace
  5. Strip stray single-digit tokens    — boilerplate / price leftovers (optional)
  6. Expand abbreviations             — after lowercase
  7. Verify protected terms survive   — sanity check

The original `text` field is preserved. Cleaned text is written
to a new `text_clean` field. Labels are never modified.

Usage:
    python -m grader.src.data.preprocess
    python -m grader.src.data.preprocess --dry-run
"""

import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Optional

import mlflow
import yaml
from sklearn.model_selection import StratifiedShuffleSplit

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------
class Preprocessor:
    """
    Text preprocessing pipeline for vinyl condition grader.

    Config keys read from grader.yaml:
        preprocessing.lowercase
        preprocessing.normalize_whitespace
        preprocessing.strip_stray_numeric_tokens
        preprocessing.abbreviation_map
        preprocessing.min_text_length_discogs
        data.splits.train / val / test
        data.splits.random_seed
        paths.processed
        paths.splits
        mlflow.tracking_uri
        mlflow.experiment_name

    Config keys read from grading_guidelines.yaml:
        grades.Mint.hard_signals          — for unverified media detection
        grades.Generic.hard_signals       — for Generic sleeve detection
        grades[*].hard_signals            — protected terms derived from all signals
    """

    def __init__(self, config_path: str, guidelines_path: str) -> None:
        self.config = self._load_yaml(config_path)
        self.guidelines = self._load_yaml(guidelines_path)

        pp_cfg = self.config["preprocessing"]
        self.do_lowercase: bool = pp_cfg.get("lowercase", True)
        self.do_normalize_whitespace: bool = pp_cfg.get(
            "normalize_whitespace", True
        )
        self.do_strip_stray_numeric_tokens: bool = pp_cfg.get(
            "strip_stray_numeric_tokens", True
        )
        # Lone digits left after ingest boilerplate stripping (e.g. "$6 / …")
        # become junk TF-IDF terms; drop them unless they look like counts
        # ("2 lp", "2 of 3", "3 x lp"), fractions, prices, or size cues
        # ("2\" split", "2 \" split", "6 inch seam", double prime U+2033).
        self._stray_digit_token: re.Pattern[str] = re.compile(
            r"(?<![0-9/\"'$])\b\d\b"
            r"(?![0-9/\"'])"
            r'(?!\s*["\u201c\u201d\u2033])'
            r"(?!\s*inch(?:es)?\b)"
            r"(?!\s*(?:lp|lps|vinyl|record|records|disc|discs|cd|cds)\b)"
            r"(?!\s*of\b)"
            r"(?!\s*x\b)"
        )

        # Build ordered abbreviation list — order from config is preserved.
        # Using list of tuples, not dict, to guarantee expansion order.
        # Longer/more specific patterns must come before shorter ones
        # (e.g. "vg++" before "vg+") — enforced in grader.yaml ordering.
        self.abbreviation_pairs: list[tuple[str, str]] = [
            (abbr.lower(), expansion)
            for abbr, expansion in pp_cfg.get("abbreviation_map", {}).items()
        ]

        # Replace the entire abbreviation_patterns block with:
        self.abbreviation_patterns: list[tuple[re.Pattern, str]] = []
        for abbr, expansion in self.abbreviation_pairs:
            escaped = re.escape(abbr.lower())
            if abbr.endswith("+"):
                # Prevent vg+ from matching inside vg++
                # by requiring the next char is not also +
                pattern = re.compile(
                    r"(?<!\w)" + escaped + r"(?!\+)",
                    re.IGNORECASE,
                )
            else:
                pattern = re.compile(
                    r"(?<!\w)" + escaped + r"(?!\w)",
                    re.IGNORECASE,
                )
            self.abbreviation_patterns.append((pattern, expansion))

        # Unverified media signals — from config
        self.unverified_signals: list[str] = self.config.get(
            "preprocessing", {}
        ).get(
            "unverified_media_signals",
            [
                "untested",
                "unplayed",
                "sold as seen",
                "haven't played",
                "not played",
                "unable to test",
                "no turntable",
            ],
        )

        # Generic sleeve hard signals — from guidelines
        self.generic_signals: list[str] = [
            s.lower()
            for s in self.guidelines["grades"]["Generic"]["hard_signals"]
        ]

        # Protected terms — derived from all hard_signals and
        # supporting_signals across all grades. These must survive
        # all text transformations unchanged.
        self.protected_terms: set[str] = self._build_protected_terms()

        # Split config
        split_cfg = self.config["data"]["splits"]
        self.train_ratio: float = split_cfg["train"]
        self.val_ratio: float = split_cfg["val"]
        self.test_ratio: float = split_cfg["test"]
        self.random_seed: int = split_cfg.get("random_seed", 42)

        # Paths
        processed_dir = Path(self.config["paths"]["processed"])
        splits_dir = Path(self.config["paths"]["splits"])
        self.input_path = processed_dir / "unified.jsonl"
        self.output_path = processed_dir / "preprocessed.jsonl"
        self.split_paths = {
            "train": splits_dir / "train.jsonl",
            "val": splits_dir / "val.jsonl",
            "test": splits_dir / "test.jsonl",
        }
        splits_dir.mkdir(parents=True, exist_ok=True)

        # MLflow
        mlflow.set_tracking_uri(self.config["mlflow"]["tracking_uri"])
        mlflow.set_experiment(self.config["mlflow"]["experiment_name"])

        # Stats
        self._stats: dict = {}

    # -----------------------------------------------------------------------
    # Config loading
    # -----------------------------------------------------------------------
    @staticmethod
    def _load_yaml(path: str) -> dict:
        with open(path, "r") as f:
            return yaml.safe_load(f)

    # -----------------------------------------------------------------------
    # Protected terms
    # -----------------------------------------------------------------------
    def _build_protected_terms(self) -> set[str]:
        """
        Derive protected terms from all grade signal lists in guidelines.
        These terms carry grading signal and must survive normalization.
        Stored in lowercase for case-insensitive comparison.
        """
        terms: set[str] = set()
        grades = self.guidelines.get("grades", {})
        for grade_def in grades.values():
            for signal_list_key in [
                "hard_signals",
                "supporting_signals",
                "forbidden_signals",
            ]:
                for signal in grade_def.get(signal_list_key, []):
                    terms.add(signal.lower())
        return terms

    # -----------------------------------------------------------------------
    # Detection — must run on RAW text
    # -----------------------------------------------------------------------
    def detect_unverified_media(self, text: str) -> bool:
        """
        Returns False (unverifiable) if any unverified media signal
        is found in the raw text. Case-insensitive.

        Must be called before any text transformation.
        """
        text_lower = text.lower()
        return not any(
            signal in text_lower for signal in self.unverified_signals
        )

    def detect_generic_sleeve(self, text: str) -> bool:
        """
        Returns True if any Generic hard signal is found in raw text.
        Used to re-confirm Generic sleeve labels from text alone.

        Note: sleeve_label may already be Generic from ingestion.
        This is a secondary text-based detection for records where
        the API condition field was ambiguous.

        Must be called before any text transformation.
        """
        text_lower = text.lower()
        return any(signal in text_lower for signal in self.generic_signals)

    # -----------------------------------------------------------------------
    # Text normalization
    # -----------------------------------------------------------------------
    def _lowercase(self, text: str) -> str:
        return text.lower() if self.do_lowercase else text

    def _normalize_whitespace(self, text: str) -> str:
        if not self.do_normalize_whitespace:
            return text
        # Collapse multiple whitespace characters into a single space
        # and strip leading/trailing whitespace
        return re.sub(r"\s+", " ", text).strip()

    def _strip_stray_numeric_tokens(self, text: str) -> str:
        if not self.do_strip_stray_numeric_tokens:
            return text
        text = self._stray_digit_token.sub(" ", text)
        return re.sub(r"\s+", " ", text).strip() if self.do_normalize_whitespace else text.strip()

    def _expand_abbreviations(self, text: str) -> str:
        """
        Expand abbreviations using ordered regex patterns.
        Order is preserved from grader.yaml abbreviation_map.
        Longer patterns (vg++) are applied before shorter ones (vg+)
        to prevent partial match corruption.
        """
        for pattern, expansion in self.abbreviation_patterns:
            text = pattern.sub(expansion, text)
        return text

    def _verify_protected_terms(self, original: str, cleaned: str) -> list[str]:
        """
        Sanity check — verify that protected terms present in the
        original text are still present in the cleaned text.

        Returns list of terms that were lost during transformation.
        A non-empty list indicates a preprocessing bug.
        """
        original_lower = original.lower()
        lost = []
        for term in self.protected_terms:
            if term in original_lower and term not in cleaned.lower():
                lost.append(term)
        return lost

    def clean_text(self, text: str) -> str:
        """
        Apply full text normalization pipeline in correct order.
        Runs on text AFTER detection steps have completed.

        Steps:
          1. Lowercase
          2. Normalize whitespace
          3. Strip stray single-digit tokens (when enabled)
          4. Expand abbreviations
        """
        cleaned = self._lowercase(text)
        cleaned = self._normalize_whitespace(cleaned)
        cleaned = self._strip_stray_numeric_tokens(cleaned)
        cleaned = self._expand_abbreviations(cleaned)
        return cleaned

    # -----------------------------------------------------------------------
    # Record processing
    # -----------------------------------------------------------------------
    def process_record(self, record: dict) -> dict:
        """
        Process a single unified record. Returns a new dict with:
          - text_clean:       normalized, expanded text
          - media_verifiable: re-detected from raw text
          - All original fields preserved unchanged

        Detection runs on original text.
        Cleaning runs after detection.
        """
        raw_text = record.get("text", "")

        # Step 1 & 2: detection on raw text
        media_verifiable = self.detect_unverified_media(raw_text)
        text_based_generic = self.detect_generic_sleeve(raw_text)

        # Step 3-6: text normalization
        text_clean = self.clean_text(raw_text)

        # Step 6: protected term sanity check
        lost_terms = self._verify_protected_terms(raw_text, text_clean)
        if lost_terms:
            logger.warning(
                "Protected terms lost during cleaning for item_id=%s: %s",
                record.get("item_id", "?"),
                lost_terms,
            )
            self._stats["protected_terms_lost"] += 1

        # Build output record — original fields preserved, new fields appended
        processed = {**record}
        processed["text_clean"] = text_clean
        processed["media_verifiable"] = media_verifiable

        # If text-based Generic detection fires but sleeve_label is not
        # already Generic, log for review — do not silently override label.
        # Label integrity is paramount; discrepancies are flagged, not fixed.
        if text_based_generic and record.get("sleeve_label") != "Generic":
            logger.debug(
                "Generic signal in text but sleeve_label=%r for item_id=%s. "
                "Label preserved — review may be needed.",
                record.get("sleeve_label"),
                record.get("item_id", "?"),
            )
            self._stats["generic_text_label_mismatch"] += 1

        return processed

    # -----------------------------------------------------------------------
    # Adaptive stratification
    # -----------------------------------------------------------------------
    def _compute_imbalance(self, labels: list[str]) -> float:
        """
        Compute imbalance ratio for a label list.
        Ratio = max_class_count / min_class_count.
        Higher ratio = more imbalanced.
        """
        counts = Counter(labels)
        if len(counts) < 2:
            return 1.0
        return max(counts.values()) / min(counts.values())

    def select_stratify_key(self, records: list[dict]) -> str:
        """
        Adaptively select the stratification key based on which
        target has higher class imbalance.

        Logs the decision and both imbalance ratios to MLflow.
        """
        sleeve_labels = [r["sleeve_label"] for r in records]
        media_labels = [r["media_label"] for r in records]

        sleeve_imbalance = self._compute_imbalance(sleeve_labels)
        media_imbalance = self._compute_imbalance(media_labels)

        stratify_key = (
            "sleeve_label"
            if sleeve_imbalance >= media_imbalance
            else "media_label"
        )

        logger.info(
            "Adaptive stratification — sleeve imbalance: %.2f | "
            "media imbalance: %.2f | stratifying on: %s",
            sleeve_imbalance,
            media_imbalance,
            stratify_key,
        )

        self._stats["sleeve_imbalance_ratio"] = sleeve_imbalance
        self._stats["media_imbalance_ratio"] = media_imbalance
        self._stats["stratify_key"] = stratify_key

        return stratify_key

    # -----------------------------------------------------------------------
    # Train/val/test split
    # -----------------------------------------------------------------------
    def split_records(self, records: list[dict]) -> dict[str, list[dict]]:
        """
        Assign train/val/test split to each record using adaptive
        stratified sampling.

        Strategy:
          1. Select stratification key based on imbalance ratio
          2. Attempt stratified split using StratifiedShuffleSplit
          3. If any stratum has < 2 samples, fall back to random split
             for affected records and log a warning

        Returns dict mapping split name → list of records.
        Each record has a "split" field added.
        """
        stratify_key = self.select_stratify_key(records)
        labels = [r[stratify_key] for r in records]

        # Identify strata with fewer than 2 samples — cannot be stratified
        label_counts = Counter(labels)
        too_rare = {label for label, count in label_counts.items() if count < 2}

        if too_rare:
            logger.warning(
                "Strata with < 2 samples — falling back to random split "
                "for these classes: %s",
                too_rare,
            )
            self._stats["rare_strata_fallback"] = list(too_rare)

        # Separate rare and stratifiable records
        rare_records = [r for r in records if r[stratify_key] in too_rare]
        strat_records = [r for r in records if r[stratify_key] not in too_rare]
        strat_labels = [r[stratify_key] for r in strat_records]

        splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}

        if strat_records:
            # First split: train vs (val + test)
            val_test_ratio = self.val_ratio + self.test_ratio
            splitter = StratifiedShuffleSplit(
                n_splits=1,
                test_size=val_test_ratio,
                random_state=self.random_seed,
            )
            train_idx, val_test_idx = next(
                splitter.split(strat_records, strat_labels)
            )

            train_records = [strat_records[i] for i in train_idx]
            val_test_records = [strat_records[i] for i in val_test_idx]
            val_test_labels = [strat_labels[i] for i in val_test_idx]

            # Second split: val vs test from the val_test pool
            val_ratio_adjusted = self.val_ratio / val_test_ratio
            splitter2 = StratifiedShuffleSplit(
                n_splits=1,
                test_size=1.0 - val_ratio_adjusted,
                random_state=self.random_seed,
            )
            val_idx, test_idx = next(
                splitter2.split(val_test_records, val_test_labels)
            )

            splits["train"] = train_records
            splits["val"] = [val_test_records[i] for i in val_idx]
            splits["test"] = [val_test_records[i] for i in test_idx]

        # Distribute rare records proportionally using random assignment
        if rare_records:
            import random

            rng = random.Random(self.random_seed)
            for record in rare_records:
                split_name = rng.choices(
                    ["train", "val", "test"],
                    weights=[
                        self.train_ratio,
                        self.val_ratio,
                        self.test_ratio,
                    ],
                )[0]
                splits[split_name].append(record)

        # Tag each record with its split name
        for split_name, split_records in splits.items():
            for record in split_records:
                record["split"] = split_name

        logger.info(
            "Split sizes — train: %d | val: %d | test: %d",
            len(splits["train"]),
            len(splits["val"]),
            len(splits["test"]),
        )

        return splits

    # -----------------------------------------------------------------------
    # Loading and saving
    # -----------------------------------------------------------------------
    def load_unified(self) -> list[dict]:
        """Load the unified JSONL file produced by harmonize_labels.py."""
        if not self.input_path.exists():
            raise FileNotFoundError(
                f"Unified dataset not found at {self.input_path}. "
                "Run harmonize_labels.py first."
            )
        records = []
        with open(self.input_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        logger.info("Loaded %d records from %s", len(records), self.input_path)
        return records

    def save_preprocessed(self, records: list[dict]) -> None:
        """Write full preprocessed dataset with split field to JSONL."""
        with open(self.output_path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info(
            "Saved %d preprocessed records to %s",
            len(records),
            self.output_path,
        )

    def save_splits(self, splits: dict[str, list[dict]]) -> None:
        """Write individual train/val/test JSONL files."""
        for split_name, split_records in splits.items():
            path = self.split_paths[split_name]
            with open(path, "w", encoding="utf-8") as f:
                for record in split_records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            logger.info("Saved %d records to %s", len(split_records), path)

    # -----------------------------------------------------------------------
    # MLflow logging
    # -----------------------------------------------------------------------
    def _log_mlflow(self, splits: dict[str, list[dict]]) -> None:
        mlflow.log_params(
            {
                "lowercase": self.do_lowercase,
                "normalize_whitespace": self.do_normalize_whitespace,
                "strip_stray_numeric_tokens": self.do_strip_stray_numeric_tokens,
                "train_ratio": self.train_ratio,
                "val_ratio": self.val_ratio,
                "test_ratio": self.test_ratio,
                "random_seed": self.random_seed,
                "stratify_key": self._stats.get("stratify_key", "unknown"),
                "n_abbreviations": len(self.abbreviation_pairs),
            }
        )
        mlflow.log_metrics(
            {
                "total_processed": self._stats["total_processed"],
                "protected_terms_lost": self._stats["protected_terms_lost"],
                "generic_text_label_mismatch": self._stats[
                    "generic_text_label_mismatch"
                ],
                "sleeve_imbalance_ratio": self._stats["sleeve_imbalance_ratio"],
                "media_imbalance_ratio": self._stats["media_imbalance_ratio"],
                "n_train": len(splits["train"]),
                "n_val": len(splits["val"]),
                "n_test": len(splits["test"]),
            }
        )

    # -----------------------------------------------------------------------
    # Orchestration
    # -----------------------------------------------------------------------
    def run(self, dry_run: bool = False) -> dict[str, list[dict]]:
        """
        Full preprocessing pipeline:
          1. Load unified.jsonl
          2. Process each record (detect + clean)
          3. Adaptive stratified split
          4. Save preprocessed.jsonl and split files
          5. Log metrics to MLflow

        Args:
            dry_run: process and split but do not write files
                     or log to MLflow.

        Returns:
            Dict mapping split name → list of processed records.
        """
        self._stats = {
            "total_processed": 0,
            "protected_terms_lost": 0,
            "generic_text_label_mismatch": 0,
            "sleeve_imbalance_ratio": 0.0,
            "media_imbalance_ratio": 0.0,
            "stratify_key": "",
            "rare_strata_fallback": [],
        }

        with mlflow.start_run(run_name="preprocess"):
            records = self.load_unified()

            # Process each record
            processed: list[dict] = []
            for record in records:
                processed.append(self.process_record(record))
                self._stats["total_processed"] += 1

            logger.info(
                "Processed %d records. Protected term losses: %d. "
                "Generic/label mismatches: %d.",
                self._stats["total_processed"],
                self._stats["protected_terms_lost"],
                self._stats["generic_text_label_mismatch"],
            )

            # Adaptive stratified split
            splits = self.split_records(processed)

            if dry_run:
                logger.info(
                    "Dry run — skipping file writes and MLflow logging."
                )
                return splits

            # Save outputs
            self.save_preprocessed(processed)
            self.save_splits(splits)
            self._log_mlflow(splits)

        return splits


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocess unified vinyl grader dataset"
    )
    parser.add_argument(
        "--config",
        default="grader/configs/grader.yaml",
        help="Path to grader config YAML",
    )
    parser.add_argument(
        "--guidelines",
        default="grader/configs/grading_guidelines.yaml",
        help="Path to grading guidelines YAML",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process and split without writing output files",
    )
    args = parser.parse_args()

    preprocessor = Preprocessor(
        config_path=args.config,
        guidelines_path=args.guidelines,
    )
    preprocessor.run(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
