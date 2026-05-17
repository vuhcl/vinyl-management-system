"""Pipeline stats and guideline signal helpers."""

from __future__ import annotations

from typing import Any

from ..listing_promo import protected_terms_from_grades


class PreprocessorSignalsMixin:
    @staticmethod
    def _fresh_pipeline_stats() -> dict[str, Any]:
        return {
            "total_processed": 0,
            "protected_terms_lost": 0,
            "generic_text_label_mismatch": 0,
            "sleeve_imbalance_ratio": 0.0,
            "media_imbalance_ratio": 0.0,
            "stratify_key": "",
            "rare_strata_fallback": [],
            "n_adequate_for_training": 0,
            "n_excluded_from_splits": 0,
            "n_test_thin": 0,
        }

    # -----------------------------------------------------------------------
    # Config loading
    # -----------------------------------------------------------------------
    # -----------------------------------------------------------------------
    # Protected terms
    # -----------------------------------------------------------------------
    @staticmethod
    def _collect_hard_signals(grade_def: dict[str, Any]) -> list[str]:
        """
        Aggregate every hard-signal phrase declared on a grade, across the
        legacy ``hard_signals`` list and the §13/§13b strict/cosignal
        variants (untargeted and per-target ``_sleeve`` / ``_media``
        keys). Deduplicated while preserving first-seen order.

        Used by preprocess-time detectors that care about "is any Generic
        / Mint hard phrase present?" and do not distinguish tiers.
        """
        if not isinstance(grade_def, dict):
            return []
        keys = (
            "hard_signals",
            "hard_signals_strict",
            "hard_signals_cosignal",
            "hard_signals_strict_sleeve",
            "hard_signals_strict_media",
            "hard_signals_cosignal_sleeve",
            "hard_signals_cosignal_media",
        )
        seen: set[str] = set()
        out: list[str] = []
        for key in keys:
            values = grade_def.get(key)
            if not isinstance(values, list):
                continue
            for signal in values:
                if not isinstance(signal, str):
                    continue
                s = signal.lower()
                if s in seen:
                    continue
                seen.add(s)
                out.append(s)
        return out

    def _build_protected_terms(self) -> set[str]:
        """
        Derive protected terms from all grade signal lists in guidelines.
        These terms carry grading signal and must survive normalization.
        Stored in lowercase for case-insensitive comparison.
        """
        grades = self.guidelines.get("grades", {})
        if not isinstance(grades, dict):
            return set()
        return protected_terms_from_grades(grades)
