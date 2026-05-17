"""Class-distribution and adequacy reports; MLflow preprocess metrics."""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

import mlflow

logger = logging.getLogger(__name__)


class PreprocessorReportsMixin:
    @staticmethod
    def _label_distribution(
        records: list[dict],
    ) -> dict[str, dict[str, int]]:
        sleeve: Counter[str] = Counter()
        media: Counter[str] = Counter()
        for record in records:
            sleeve[str(record["sleeve_label"])] += 1
            media[str(record["media_label"])] += 1
        return {"sleeve": dict(sleeve), "media": dict(media)}

    def _rare_class_warnings_for_dist(
        self,
        distribution: dict[str, dict[str, int]],
        *,
        scope: str,
    ) -> list[str]:
        warnings: list[str] = []
        threshold = self._harmonization_min_samples
        for target, grade_counts in distribution.items():
            for grade, count in grade_counts.items():
                if count < threshold:
                    warnings.append(
                        f"RARE CLASS — scope: {scope}, target: {target}, "
                        f"grade: {grade}, count: {count} "
                        f"(threshold: {threshold})"
                    )
        return warnings

    def _format_grade_table_lines(
        self,
        distribution: dict[str, dict[str, int]],
    ) -> list[str]:
        sleeve_order = self.guidelines["sleeve_grades"]
        sleeve_dist = distribution["sleeve"]
        media_dist = distribution["media"]
        lines = [
            "-" * 60,
            f"{'Grade':<20} {'Sleeve':>8} {'Media':>8}",
            "-" * 60,
        ]
        for grade in sleeve_order:
            sleeve_count = sleeve_dist.get(grade, 0)
            media_count = (
                "-" if grade == "Generic" else media_dist.get(grade, 0)
            )
            lines.append(
                f"{grade:<20} {sleeve_count:>8} {str(media_count):>8}"
            )
        sleeve_total = sum(sleeve_dist.values())
        media_total = sum(media_dist.values())
        lines += [
            "-" * 60,
            f"{'Total':<20} {sleeve_total:>8} {media_total:>8}",
            "",
        ]
        return lines

    def _format_class_distribution_splits_report(
        self,
        processed: list[dict],
        out_splits: dict[str, list[dict]],
    ) -> str:
        lines: list[str] = [
            "=" * 60,
            "VINYL GRADER — CLASS DISTRIBUTION BY SPLIT (AFTER PREPROCESS)",
            "=" * 60,
            "",
            f"Total preprocessed rows (full pool): {len(processed):>10}",
        ]
        if (
            self.description_adequacy_enabled
            and self.drop_insufficient_from_training
        ):
            eligible = self._stats.get("n_adequate_for_training", 0)
            excl = self._stats.get("n_excluded_from_splits", 0)
            lines += [
                f"Eligible for train/val/test:       {eligible:>10}",
                f"Excluded from splits (thin):     {excl:>10}",
                "",
            ]

        by_source: Counter[str] = Counter(
            str(r.get("source") or "?") for r in processed
        )
        lines.append("By source (full pool):")
        for src in sorted(by_source.keys()):
            lines.append(f"  {src + ':':<40} {by_source[src]:>10}")
        lines.append("")
        lines.append("Split sizes:")
        for name in ("train", "val", "test", "test_thin"):
            if name not in out_splits:
                continue
            lines.append(f"  {name + ':':<40} {len(out_splits[name]):>10}")
        lines.append("")

        lines.append("-" * 60)
        lines.append("Full pool (all rows written to preprocessed.jsonl)")
        lines.append("-" * 60)
        full_dist = self._label_distribution(processed)
        lines.extend(self._format_grade_table_lines(full_dist))
        all_warnings = self._rare_class_warnings_for_dist(
            full_dist, scope="full_pool"
        )

        for split_name in ("train", "val", "test", "test_thin"):
            if split_name not in out_splits:
                continue
            rows = out_splits[split_name]
            lines.append("-" * 60)
            lines.append(f"Split: {split_name} ({len(rows)} rows)")
            lines.append("-" * 60)
            dist = self._label_distribution(rows)
            lines.extend(self._format_grade_table_lines(dist))
            all_warnings.extend(
                self._rare_class_warnings_for_dist(
                    dist, scope=f"split:{split_name}"
                )
            )

        if all_warnings:
            lines += [
                "=" * 60,
                "RARE CLASS WARNINGS",
                "=" * 60,
            ]
            for w in all_warnings:
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

    def _save_class_distribution_splits_report(
        self,
        processed: list[dict],
        out_splits: dict[str, list[dict]],
    ) -> None:
        path = self.reports_dir / "class_distribution_splits.txt"
        text = self._format_class_distribution_splits_report(
            processed, out_splits
        )
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        logger.info("Saved class distribution (splits) to %s", path)

    # -----------------------------------------------------------------------
    # MLflow logging
    # -----------------------------------------------------------------------
    def _log_mlflow(self, splits: dict[str, list[dict]]) -> None:
        mlflow.log_params(
            {
                "lowercase": self.do_lowercase,
                "normalize_whitespace": self.do_normalize_whitespace,
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
                "sleeve_imbalance_ratio": self._stats[
                    "sleeve_imbalance_ratio"
                ],
                "media_imbalance_ratio": self._stats["media_imbalance_ratio"],
                "n_train": len(splits["train"]),
                "n_val": len(splits["val"]),
                "n_test": len(splits["test"]),
                "n_adequate_for_training": self._stats.get(
                    "n_adequate_for_training", 0
                ),
                "n_excluded_from_splits": self._stats.get(
                    "n_excluded_from_splits", 0
                ),
                "n_test_thin": self._stats.get("n_test_thin", 0),
            }
        )

    def _save_description_adequacy_report(
        self,
        all_processed: list[dict],
        split_pool: list[dict],
    ) -> None:
        path = self.reports_dir / "description_adequacy_summary.txt"
        excl = [r for r in all_processed if not r["adequate_for_training"]]
        lines = [
            "Description adequacy (preprocessing)",
            "=" * 60,
            f"Total records:           {len(all_processed)}",
            f"Eligible for splits:     {len(split_pool)}",
            f"Excluded (thin notes):   {len(excl)}",
            "",
            "Excluded rows lack sleeve cues and/or playable-media cues "
            "(see preprocessing.description_adequacy in grader.yaml).",
            "They remain in preprocessed.jsonl with adequacy flags for audit.",
            "Eval-only split: grader/data/splits/test_thin.jsonl (same rows).",
            "",
        ]
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        logger.info("Wrote %s", path)

