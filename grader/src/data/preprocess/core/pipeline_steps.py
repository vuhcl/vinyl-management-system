"""Record processing, splitting, I/O, and pipeline orchestration."""

from __future__ import annotations

import json
import logging
import random
from collections import Counter

from sklearn.model_selection import StratifiedShuffleSplit

from grader.src.mlflow_tracking import mlflow_pipeline_step_run_ctx

logger = logging.getLogger(__name__)


class _PreprocessorSteps:
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
        media_evidence_strength = self.detect_media_evidence_strength(raw_text)
        text_based_generic = self.detect_generic_sleeve(raw_text)

        # Step 3-5: text normalization
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
        processed["media_evidence_strength"] = media_evidence_strength

        dq = self.compute_description_quality(
            raw_text,
            text_clean,
            sleeve_label=str(record.get("sleeve_label") or ""),
            media_label=str(record.get("media_label") or ""),
        )
        processed.update(dq)

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
        too_rare = {
            label for label, count in label_counts.items() if count < 2
        }

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

    def _write_test_thin_jsonl(self, thin_records: list[dict]) -> None:
        """Eval-only split: rows not eligible for train/val/test adequacy pool."""
        path = self.split_paths["test_thin"]
        with open(path, "w", encoding="utf-8") as f:
            for record in thin_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info(
            "Saved %d thin-note eval records to %s",
            len(thin_records),
            path,
        )

    def _remove_stale_test_thin_file(self) -> None:
        path = self.split_paths["test_thin"]
        if path.exists():
            path.unlink()
            logger.info(
                "Removed %s — thin-note split only when "
                "description_adequacy + drop_insufficient_from_training are on",
                path,
            )

    def run(self, dry_run: bool = False) -> dict[str, list[dict]]:
        """
        Full preprocessing pipeline:
          1. Load unified.jsonl
          2. Process each record (detect + clean)
          3. Adaptive stratified split (adequate rows only when thin filter on)
          4. Save preprocessed.jsonl, train/val/test, and optional test_thin.jsonl
          5. Save ``class_distribution_splits.txt`` under ``paths.reports``
          6. Log metrics to MLflow

        Args:
            dry_run: process and split but do not write files
                     or log to MLflow.

        Returns:
            Dict mapping split name → list of processed records.
        """
        self._stats = self._fresh_pipeline_stats()

        with mlflow_pipeline_step_run_ctx(self.config, "preprocess") as mlf:
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

            split_pool = processed
            if (
                self.description_adequacy_enabled
                and self.drop_insufficient_from_training
            ):
                split_pool = [r for r in processed if r["adequate_for_training"]]
                self._stats["n_excluded_from_splits"] = len(processed) - len(
                    split_pool
                )
                self._stats["n_adequate_for_training"] = len(split_pool)
                logger.info(
                    "Description adequacy — eligible for splits: %d | "
                    "excluded (thin notes): %d",
                    len(split_pool),
                    self._stats["n_excluded_from_splits"],
                )
                self._save_description_adequacy_report(processed, split_pool)
            else:
                self._stats["n_adequate_for_training"] = len(processed)

            if not split_pool:
                raise ValueError(
                    "No records left for train/val/test splits after "
                    "description_adequacy filtering. Relax "
                    "preprocessing.description_adequacy or set "
                    "drop_insufficient_from_training: false."
                )

            # Adaptive stratified split (training-eligible rows only when filtering)
            splits = self.split_records(split_pool)

            thin_records: list[dict] = []
            if (
                self.description_adequacy_enabled
                and self.drop_insufficient_from_training
            ):
                thin_records = [
                    r for r in processed if not r["adequate_for_training"]
                ]
                for r in thin_records:
                    r["split"] = "test_thin"
                self._stats["n_test_thin"] = len(thin_records)
            else:
                self._remove_stale_test_thin_file()

            out_splits = dict(splits)
            if (
                self.description_adequacy_enabled
                and self.drop_insufficient_from_training
            ):
                out_splits["test_thin"] = thin_records

            if dry_run:
                logger.info(
                    "Dry run — skipping file writes and MLflow logging."
                )
                return out_splits

            # Save outputs
            self.save_preprocessed(processed)
            self.save_splits(splits)
            self._write_test_thin_jsonl(thin_records)
            self._save_class_distribution_splits_report(processed, out_splits)
            if mlf:
                self._log_mlflow(splits)

        return out_splits
