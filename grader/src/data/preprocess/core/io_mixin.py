"""Load unified JSONL; write preprocessed and split files."""

from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)


class PreprocessorIOMixin:
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

