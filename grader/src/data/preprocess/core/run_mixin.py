"""End-to-end preprocess orchestration (`run`)."""

from __future__ import annotations

import logging

from grader.src.mlflow_tracking import mlflow_pipeline_step_run_ctx

logger = logging.getLogger(__name__)


class PreprocessorRunMixin:
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
