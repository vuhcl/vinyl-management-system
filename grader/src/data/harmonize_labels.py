"""
grader/src/data/harmonize_labels.py

Merges processed Discogs, optional sale-history and release-marketplace JSONL,
and eBay JP JSONL into a single unified dataset. Validates schema conformance, checks grade
validity against the canonical schema, deduplicates within and
across sources, reports class distribution, and flags rare classes.

This module is a merge + validation step. Optional
``data.harmonization.exclude_thin_notes`` reuses preprocess description-adequacy
rules so unified.jsonl can match the training-eligible pool.
No feature engineering or splitting — those stay in preprocess and pipeline.py.

Usage:
    python -m grader.src.data.harmonize_labels
    python -m grader.src.data.harmonize_labels --dry-run
"""

import copy
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import mlflow

from grader.src.config_io import load_yaml_mapping
from grader.src.mlflow_tracking import (
    mlflow_log_artifacts_enabled,
    mlflow_pipeline_step_run_ctx,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Required fields for schema validation
# ---------------------------------------------------------------------------
REQUIRED_FIELDS = [
    "item_id",
    "source",
    "text",
    "sleeve_label",
    "media_label",
    "label_confidence",
]


# ---------------------------------------------------------------------------
# LabelHarmonizer
# ---------------------------------------------------------------------------
class LabelHarmonizer:
    """
    Merges Discogs and eBay JP processed JSONL files into a single
    unified dataset with validated, canonical grade labels.

    Config keys read from grader.yaml:
        paths.processed                     — directory of source JSONL files
        data.harmonization.min_samples_per_class
        data.harmonization.output_path
        data.harmonization.report_path
        data.harmonization.exclude_thin_notes
        mlflow.tracking_uri
        mlflow.experiment_name

    Config keys read from grading_guidelines.yaml:
        sleeve_grades                        — valid sleeve label values
        media_grades                         — valid media label values
    """

    def __init__(
        self,
        config_path: str,
        guidelines_path: str,
        config: Optional[dict[str, Any]] = None,
    ) -> None:
        self._config_path = config_path
        self._guidelines_path = guidelines_path
        if config is not None:
            self.config = copy.deepcopy(config)
        else:
            self.config = load_yaml_mapping(config_path)
        self.guidelines = load_yaml_mapping(guidelines_path)

        # Valid grade sets
        self.sleeve_grades: set[str] = set(self.guidelines["sleeve_grades"])
        self.media_grades: set[str] = set(self.guidelines["media_grades"])

        # Harmonization config
        harmonization_cfg = self.config["data"]["harmonization"]
        self.min_samples: int = harmonization_cfg.get(
            "min_samples_per_class", 20
        )
        self.output_path = Path(harmonization_cfg["output_path"])
        self.report_path = Path(harmonization_cfg["report_path"])
        self.exclude_thin_notes: bool = bool(
            harmonization_cfg.get("exclude_thin_notes", False)
        )

        # Source file paths
        processed_dir = Path(self.config["paths"]["processed"])
        self.source_paths = {
            "discogs": processed_dir / "discogs_processed.jsonl",
            "discogs_release_marketplace": processed_dir
            / "discogs_release_marketplace.jsonl",
            "sale_history": processed_dir / "discogs_sale_history.jsonl",
            "ebay_jp": processed_dir / "ebay_processed.jsonl",
        }

        # Ensure output directories exist
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.parent.mkdir(parents=True, exist_ok=True)

        # MLflow: ``run()`` uses ``mlflow_pipeline_step_run_ctx`` — configure there
        # when a step run is actually opened, not on every Harmonizer construction.

        # Stats — reset on each run()
        self._stats: dict = {}

    # -----------------------------------------------------------------------
    # Loading
    # -----------------------------------------------------------------------
    def load_discogs_processed_sources(self) -> list[dict]:
        """
        Discogs training rows may come from seller inventory JSONL plus optional
        ``discogs_release_marketplace.jsonl`` (website scrape per release).
        Missing optional file is treated as empty (no warning).
        """
        main = self.load_jsonl(self.source_paths["discogs"])
        extra_path = self.source_paths["discogs_release_marketplace"]
        if not extra_path.exists():
            return main
        extra = self.load_jsonl(extra_path)
        return main + extra

    def load_jsonl(self, path: Path) -> list[dict]:
        """
        Load all records from a JSONL file.
        Skips malformed lines with a logged warning.
        """
        if not path.exists():
            logger.warning("Source file not found — skipping: %s", path)
            return []

        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(
                        "Malformed JSON at %s line %d — skipping. Error: %s",
                        path.name,
                        line_num,
                        e,
                    )

        logger.info("Loaded %d records from %s", len(records), path.name)
        return records

    # -----------------------------------------------------------------------
    # Schema validation
    # -----------------------------------------------------------------------
    def validate_record(self, record: dict) -> Optional[str]:
        """
        Check that a record has all required fields with non-null values.
        Returns a drop reason string if invalid, None if valid.
        """
        for field in REQUIRED_FIELDS:
            if field not in record or record[field] is None:
                return f"missing_field_{field}"
            if isinstance(record[field], str) and not record[field].strip():
                return f"empty_field_{field}"
        return None

    # -----------------------------------------------------------------------
    # Grade validity
    # -----------------------------------------------------------------------
    def validate_grades(self, record: dict) -> Optional[str]:
        """
        Verify sleeve_label and media_label are valid canonical grades
        for their respective targets.

        Rules:
          - sleeve_label must be in sleeve_grades (includes Generic)
          - media_label must be in media_grades (excludes Generic)
          - Generic is never valid as media_label
        """
        sleeve = record["sleeve_label"]
        media = record["media_label"]

        if sleeve not in self.sleeve_grades:
            return f"invalid_sleeve_grade:{sleeve}"

        # Check Generic before general media validation —
        # Generic is not in media_grades so general check would fire first
        if media == "Generic":
            return "generic_as_media_grade"

        if media not in self.media_grades:
            return f"invalid_media_grade:{media}"

        return None

    # -----------------------------------------------------------------------
    # Deduplication
    # -----------------------------------------------------------------------
    def deduplicate(self, records: list[dict], source: str) -> list[dict]:
        """
        Remove duplicate item_ids within a single source.
        Keeps first occurrence. Logs duplicate count.
        """
        seen: set[str] = set()
        deduped: list[dict] = []
        duplicates = 0

        for record in records:
            item_id = record.get("item_id", "")
            key = f"{source}:{item_id}"
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            deduped.append(record)

        if duplicates:
            logger.warning(
                "Removed %d duplicate item_ids from source: %s",
                duplicates,
                source,
            )

        self._stats["duplicates_removed"][source] = duplicates
        return deduped

    def deduplicate_cross_source(self, records: list[dict]) -> list[dict]:
        """
        Detect and log cross-source duplicates — same item appearing
        in both Discogs and eBay JP data.

        Cross-source deduplication uses title + artist as a fuzzy key
        since item_ids differ across platforms. Keeps Discogs record
        (higher label_confidence = 1.0) when a cross-source match is found.

        These are expected to be rare. This step is informational —
        duplicates are logged but only removed on exact title+artist match.
        """
        seen: dict[str, str] = {}  # key → source
        deduped: list[dict] = []
        cross_source_dups = 0

        for record in records:
            artist = record.get("artist", "").lower().strip()
            title = record.get("title", "").lower().strip()
            source = record.get("source", "")

            if artist and title:
                key = f"{artist}||{title}"
                if key in seen and seen[key] != source:
                    cross_source_dups += 1
                    logger.debug(
                        "Cross-source duplicate — title: %r, "
                        "keeping: %s, dropping: %s",
                        title,
                        seen[key],
                        source,
                    )
                    # Keep Discogs record (already in deduped), skip this one
                    if source != "discogs":
                        continue
                seen[key] = source

            deduped.append(record)

        if cross_source_dups:
            logger.warning(
                "Found %d cross-source duplicates (title+artist match). "
                "Kept Discogs records.",
                cross_source_dups,
            )

        self._stats["cross_source_duplicates"] = cross_source_dups
        return deduped

    def _filter_thin_notes(self, records: list[dict]) -> tuple[list[dict], int]:
        """
        Keep only rows with adequate_for_training per preprocess rules.
        Returns (filtered_records, excluded_count).
        """
        from grader.src.data.preprocess import Preprocessor

        pre = Preprocessor(self._config_path, self._guidelines_path)
        if not pre.description_adequacy_enabled:
            logger.warning(
                "data.harmonization.exclude_thin_notes is true but "
                "preprocessing.description_adequacy.enabled is false — "
                "no rows removed; enable adequacy to filter thin notes."
            )
            return records, 0

        kept: list[dict] = []
        for record in records:
            raw = record.get("text", "") or ""
            text_clean = pre.clean_text(raw)
            dq = pre.compute_description_quality(
                raw,
                text_clean,
                sleeve_label=str(record.get("sleeve_label") or ""),
                media_label=str(record.get("media_label") or ""),
            )
            if dq["adequate_for_training"]:
                kept.append(record)

        excluded = len(records) - len(kept)
        if excluded:
            logger.info(
                "Excluded %d thin-note (inadequate) rows — %d remain",
                excluded,
                len(kept),
            )
        return kept, excluded

    # -----------------------------------------------------------------------
    # Class distribution
    # -----------------------------------------------------------------------
    def compute_distribution(
        self, records: list[dict]
    ) -> dict[str, dict[str, int]]:
        """
        Compute per-grade counts for sleeve and media targets.

        Returns:
            {
                "sleeve": {"Mint": 142, "Near Mint": 487, ...},
                "media":  {"Mint": 142, "Near Mint": 501, ...},
            }
        """
        dist: dict[str, dict[str, int]] = {
            "sleeve": defaultdict(int),
            "media": defaultdict(int),
        }
        for record in records:
            dist["sleeve"][record["sleeve_label"]] += 1
            dist["media"][record["media_label"]] += 1

        return dist

    def flag_rare_classes(
        self, distribution: dict[str, dict[str, int]]
    ) -> list[str]:
        """
        Warn about any grade/target combination with fewer than
        min_samples_per_class samples.

        Returns list of warning strings for logging and reporting.
        Poor and Generic are expected to be rare — warnings are
        informational, not hard stops.
        """
        warnings = []
        for target, grade_counts in distribution.items():
            for grade, count in grade_counts.items():
                if count < self.min_samples:
                    msg = (
                        f"RARE CLASS — target: {target}, grade: {grade}, "
                        f"count: {count} (threshold: {self.min_samples})"
                    )
                    warnings.append(msg)
                    logger.warning(msg)
        return warnings

    # -----------------------------------------------------------------------
    # Distribution report
    # -----------------------------------------------------------------------
    def format_report(
        self,
        distribution: dict[str, dict[str, int]],
        stats: dict,
        warnings: list[str],
    ) -> str:
        """
        Format a human-readable class distribution report.
        Saved to grader/reports/class_distribution.txt.
        """
        sleeve_order = self.guidelines["sleeve_grades"]
        media_dist = distribution["media"]
        sleeve_dist = distribution["sleeve"]

        lines = [
            "=" * 60,
            "VINYL GRADER — CLASS DISTRIBUTION REPORT",
            "=" * 60,
            "",
            f"Total records:          {stats['total_saved']:>6}",
            f"  Discogs:              {stats['per_source'].get('discogs', 0):>6}",
            f"  Sale history:         {stats['per_source'].get('sale_history', 0):>6}",
            f"  eBay JP:              {stats['per_source'].get('ebay_jp', 0):>6}",
            "",
            f"Total dropped:          {stats['total_dropped']:>6}",
            "",
            "Drop reasons:",
        ]

        for reason, count in stats["drops"].items():
            lines.append(f"  {reason:<40} {count:>6}")

        lines += [
            "",
            "Duplicates removed (within source):",
            f"  Discogs:              "
            f"{stats['duplicates_removed'].get('discogs', 0):>6}",
            f"  Sale history:         "
            f"{stats['duplicates_removed'].get('sale_history', 0):>6}",
            f"  eBay JP:              "
            f"{stats['duplicates_removed'].get('ebay_jp', 0):>6}",
            f"Cross-source duplicates removed:      "
            f"{stats['cross_source_duplicates']:>6}",
            "",
            "-" * 60,
            f"{'Grade':<20} {'Sleeve':>8} {'Media':>8}",
            "-" * 60,
        ]

        for grade in sleeve_order:
            sleeve_count = sleeve_dist.get(grade, 0)
            media_count = (
                "-" if grade == "Generic" else media_dist.get(grade, 0)
            )
            lines.append(f"{grade:<20} {sleeve_count:>8} {str(media_count):>8}")

        sleeve_total = sum(sleeve_dist.values())
        media_total = sum(media_dist.values())
        lines += [
            "-" * 60,
            f"{'Total':<20} {sleeve_total:>8} {media_total:>8}",
            "",
        ]

        if warnings:
            lines += [
                "=" * 60,
                "RARE CLASS WARNINGS",
                "=" * 60,
            ]
            for w in warnings:
                lines.append(f"  {w}")
            lines.append("")

        lines += [
            "=" * 60,
            "Note: Poor and Generic are expected to be rare.",
            "Rule engine owns these grades — low sample count",
            "does not prevent grading of these conditions.",
            "=" * 60,
        ]

        return "\n".join(lines)

    def save_report(self, report_text: str) -> None:
        with open(self.report_path, "w", encoding="utf-8") as f:
            f.write(report_text)
        logger.info("Distribution report saved to %s", self.report_path)

    # -----------------------------------------------------------------------
    # Output
    # -----------------------------------------------------------------------
    def save_unified(self, records: list[dict]) -> None:
        """
        Write all validated records to the unified JSONL output file.
        Overwrites any existing file — harmonize is idempotent.
        """
        with open(self.output_path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info(
            "Saved %d unified records to %s",
            len(records),
            self.output_path,
        )

    # -----------------------------------------------------------------------
    # MLflow logging
    # -----------------------------------------------------------------------
    def _log_mlflow(
        self,
        distribution: dict[str, dict[str, int]],
    ) -> None:
        mlflow.log_params(
            {
                "sources": [
                    k
                    for k in self.source_paths
                    if k
                    not in (
                        "discogs_release_marketplace",
                    )
                ],
                "min_samples_per_class": self.min_samples,
                "exclude_thin_notes": self.exclude_thin_notes,
            }
        )
        metrics = {
            "total_saved": self._stats["total_saved"],
            "total_dropped": self._stats["total_dropped"],
            "cross_source_duplicates": self._stats[
                "cross_source_duplicates"
            ],
        }
        thin_drop = self._stats["drops"].get("thin_note_inadequate", 0)
        if thin_drop:
            metrics["thin_notes_excluded"] = thin_drop
        mlflow.log_metrics(metrics)
        for source, count in self._stats["per_source"].items():
            mlflow.log_metric(f"saved_{source}", count)
        for reason, count in self._stats["drops"].items():
            clean_reason = reason.replace(":", "_").replace(" ", "_")
            mlflow.log_metric(f"dropped_{clean_reason}", count)
        for target, grade_counts in distribution.items():
            for grade, count in grade_counts.items():
                clean_grade = grade.lower().replace(" ", "_")
                mlflow.log_metric(f"dist_{target}_{clean_grade}", count)
        if mlflow_log_artifacts_enabled(self.config):
            mlflow.log_artifact(str(self.report_path))

    # -----------------------------------------------------------------------
    # Orchestration
    # -----------------------------------------------------------------------
    def run(self, dry_run: bool = False) -> list[dict]:
        """
        Full harmonization pipeline:
          1. Load Discogs and eBay JP processed JSONL files
          2. Validate schema and grade validity per record
          3. Deduplicate within each source
          4. Merge sources
          5. Deduplicate across sources
          6. Compute and save class distribution report
          7. Flag rare classes
          8. Save unified JSONL
          9. Log metrics to MLflow

        Args:
            dry_run: if True, run all validation and reporting steps
                     but do not write output files or log to MLflow.

        Returns:
            List of validated, unified record dicts.
        """
        self._stats = {
            "total_fetched": 0,
            "total_dropped": 0,
            "total_saved": 0,
            "per_source": {},
            "drops": defaultdict(int),
            "duplicates_removed": {},
            "cross_source_duplicates": 0,
        }

        with mlflow_pipeline_step_run_ctx(
            self.config, "harmonize_labels"
        ) as mlf:
            all_records: list[dict] = []

            for source, path in self.source_paths.items():
                if source == "discogs_release_marketplace":
                    continue
                if source == "discogs":
                    raw_records = self.load_discogs_processed_sources()
                else:
                    raw_records = self.load_jsonl(path)
                self._stats["total_fetched"] += len(raw_records)

                valid_records: list[dict] = []
                for record in raw_records:
                    drop_reason = self.validate_record(record)
                    if drop_reason:
                        self._stats["drops"][drop_reason] += 1
                        self._stats["total_dropped"] += 1
                        continue

                    for _k in ("sleeve_label", "media_label"):
                        _v = record.get(_k)
                        if isinstance(_v, str):
                            record[_k] = _v.strip()

                    drop_reason = self.validate_grades(record)
                    if drop_reason:
                        self._stats["drops"][drop_reason] += 1
                        self._stats["total_dropped"] += 1
                        continue

                    valid_records.append(record)

                deduped = self.deduplicate(valid_records, source)
                self._stats["per_source"][source] = len(deduped)
                all_records.extend(deduped)

                logger.info(
                    "Source %-12s — loaded: %4d | valid: %4d | "
                    "after dedup: %4d",
                    source,
                    len(raw_records),
                    len(valid_records),
                    len(deduped),
                )

            all_records = self.deduplicate_cross_source(all_records)

            if self.exclude_thin_notes:
                all_records, thin_excluded = self._filter_thin_notes(
                    all_records
                )
                if thin_excluded:
                    self._stats["drops"]["thin_note_inadequate"] = (
                        thin_excluded
                    )
                    self._stats["total_dropped"] += thin_excluded

            self._stats["total_saved"] = len(all_records)

            logger.info(
                "Harmonization complete — total saved: %d | total dropped: %d",
                self._stats["total_saved"],
                self._stats["total_dropped"],
            )

            distribution = self.compute_distribution(all_records)
            warnings = self.flag_rare_classes(distribution)
            report_text = self.format_report(
                distribution, self._stats, warnings
            )
            print("\n" + report_text)

            if dry_run:
                logger.info(
                    "Dry run — skipping file writes and MLflow logging."
                )
                return all_records

            self.save_unified(all_records)
            self.save_report(report_text)
            if mlf:
                self._log_mlflow(distribution)

        return all_records


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Harmonize Discogs, optional sale-history / marketplace JSONL, "
        "and eBay JP processed data into a single unified dataset"
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
        help="Run validation and reporting without writing output files",
    )
    args = parser.parse_args()

    harmonizer = LabelHarmonizer(
        config_path=args.config,
        guidelines_path=args.guidelines,
    )
    harmonizer.run(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
