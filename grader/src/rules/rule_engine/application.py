"""apply / apply_batch and coverage summaries."""

from __future__ import annotations

import logging
from typing import Optional

from grader.src.guidelines_identity import guidelines_version_from_mapping
from grader.src.schemas import GraderPrediction

from .constants import MEDIA, SLEEVE

logger = logging.getLogger(__name__)


class RuleEngineApplicationMixin:
    """End-to-end prediction application and batch stats."""

    def _collapse_excellent_to_near_mint(self, result: GraderPrediction) -> None:
        """
        When soft EX is disabled, map Excellent → Near Mint on final outputs and
        fold EX probability mass into NM in confidence dicts (if present).
        """
        if self._allow_excellent_soft_override:
            return
        collapsed: list[str] = []
        for target, pred_key in (
            ("sleeve", "predicted_sleeve_condition"),
            ("media", "predicted_media_condition"),
        ):
            if result.get(pred_key) != "Excellent":
                continue
            result[pred_key] = "Near Mint"
            collapsed.append(target)
            scores = result.get("confidence_scores", {}).get(target)
            if isinstance(scores, dict) and "Excellent" in scores:
                ex_prob = float(scores.pop("Excellent", 0.0))
                scores["Near Mint"] = float(scores.get("Near Mint", 0.0)) + ex_prob
        if collapsed:
            result["metadata"]["excellent_collapsed_to_near_mint"] = collapsed

    # -----------------------------------------------------------------------
    # Single prediction application
    # -----------------------------------------------------------------------
    def apply(
        self,
        prediction: GraderPrediction,
        text: str,
    ) -> GraderPrediction:
        """
        Apply rule engine to a single prediction dict.

        Modifies a copy of the prediction — original is never mutated.
        Runs full rule logic independently for sleeve and media.

        Args:
            prediction: :class:`~grader.src.schemas.GraderPrediction` from models or pyfunc adapter
            text:       preprocessed seller notes (text_clean field)

        Returns:
            Updated prediction dict with rule overrides applied
            and metadata flags set.
        """
        # Work on a shallow copy — preserve original
        result = {**prediction}
        result["confidence_scores"] = {
            "sleeve": dict(prediction["confidence_scores"]["sleeve"]),
            "media": dict(prediction["confidence_scores"]["media"]),
        }
        result["metadata"] = dict(prediction["metadata"])
        result["metadata"]["guidelines_version"] = guidelines_version_from_mapping(
            self.guidelines
        )

        text_lower = text.lower()

        # Step 1 — Contradiction detection (runs first, may suppress overrides)
        contradiction = self.check_contradiction(text_lower)
        result["metadata"]["contradiction_detected"] = contradiction

        if contradiction:
            logger.debug(
                "Contradiction detected for item_id=%s — "
                "suppressing all rule overrides.",
                prediction.get("item_id", "?"),
            )
            result["metadata"]["rule_override_applied"] = False
            result["metadata"]["rule_override_target"] = None
            self._collapse_excellent_to_near_mint(result)
            return result

        # Step 2 & 3 — Apply rules per target independently
        override_targets = []

        for target in [SLEEVE, MEDIA]:
            predicted_grade = (
                result["predicted_sleeve_condition"]
                if target == SLEEVE
                else result["predicted_media_condition"]
            )

            # Extract model confidence for predicted grade
            scores = result["confidence_scores"][target]
            model_confidence = scores.get(predicted_grade, 0.0)

            # Step 2a — Sleeve-specific guardrail:
            # seam/spine split should not remain Near Mint.
            override_grade = None
            if (
                target == SLEEVE
                and predicted_grade == "Near Mint"
                and self._match_signals(
                    text_lower, self._nm_sleeve_downgrade_patterns
                )
            ):
                override_grade = self._nm_sleeve_split_override(text_lower)

            # Step 2b — Hard signal override (Poor, Generic)
            hard_tier: Optional[str] = None
            hard_signal: Optional[str] = None
            if override_grade is None:
                self._last_hard_tier = None
                self._last_hard_signal = None
                override_grade = self.check_hard_override(text_lower, target)
                if override_grade is not None:
                    hard_tier = self._last_hard_tier
                    hard_signal = self._last_hard_signal

            # Step 3 — Soft signal override (if no hard override)
            if override_grade is None:
                override_grade = self.check_soft_override(
                    text_lower,
                    target,
                    model_confidence,
                    predicted_grade,
                )

            # Step 3b — Tight seam-split guardrail (after soft override):
            # If the model predicted Very Good Plus and the text indicates only
            # a small seam/spine split, do not allow a soft downgrade to Very Good.
            if (
                override_grade == "Very Good"
                and predicted_grade == "Very Good Plus"
                and self._nm_sleeve_split_override(text_lower) == "Very Good Plus"
            ):
                override_grade = None

            if override_grade is not None:
                # Apply override
                if target == SLEEVE:
                    result["predicted_sleeve_condition"] = override_grade
                else:
                    result["predicted_media_condition"] = override_grade
                override_targets.append(target)
                if hard_tier is not None:
                    tier_key = f"hard_override_tier_{target}"
                    sig_key = f"hard_override_signal_{target}"
                    result["metadata"][tier_key] = hard_tier
                    result["metadata"][sig_key] = hard_signal

        # Update metadata
        if override_targets:
            result["metadata"]["rule_override_applied"] = True
            if len(override_targets) == 2:
                result["metadata"]["rule_override_target"] = "both"
            else:
                result["metadata"]["rule_override_target"] = override_targets[0]
        else:
            result["metadata"]["rule_override_applied"] = False
            result["metadata"]["rule_override_target"] = None

        self._collapse_excellent_to_near_mint(result)
        return result

    def _nm_sleeve_split_override(self, text_lower: str) -> Optional[str]:
        """
        NM sleeve split downgrade rule:
          - only "small seam/spine split" can stay at VG+
          - any other split phrasing, or small split plus another defect cue, -> VG
        """
        if not self._match_signals(text_lower, self._nm_sleeve_downgrade_patterns):
            return None

        has_small_split = bool(
            self._match_signals(text_lower, self._nm_small_split_patterns)
        )
        if not has_small_split:
            return "Very Good"

        if self._match_signals(text_lower, self._nm_split_other_defect_patterns):
            return "Very Good"

        return "Very Good Plus"

    # -----------------------------------------------------------------------
    # Batch application
    # -----------------------------------------------------------------------
    def apply_batch(
        self,
        predictions: list[GraderPrediction],
        texts: list[str],
    ) -> list[GraderPrediction]:
        """
        Apply rule engine to a batch of predictions.

        Args:
            predictions: list of prediction dicts
            texts:       list of preprocessed text strings
                         (must be same length and order as predictions)

        Returns:
            List of updated prediction dicts.
        """
        if len(predictions) != len(texts):
            raise ValueError(
                f"predictions and texts must have the same length. "
                f"Got {len(predictions)} predictions and {len(texts)} texts."
            )

        results = []
        override_count = 0
        contradiction_count = 0

        for prediction, text in zip(predictions, texts):
            result = self.apply(prediction, text)
            results.append(result)

            if result["metadata"]["rule_override_applied"]:
                override_count += 1
            if result["metadata"]["contradiction_detected"]:
                contradiction_count += 1

        logger.info(
            "Rule engine batch complete — "
            "total: %d | overrides: %d | contradictions: %d",
            len(predictions),
            override_count,
            contradiction_count,
        )

        return results

    # -----------------------------------------------------------------------
    # Rule coverage report
    # -----------------------------------------------------------------------
    def summarize_results(self, results: list[GraderPrediction]) -> dict:
        """
        Aggregate override / contradiction stats from apply_batch outputs
        without re-running the rule engine.
        """
        stats = {
            "total": len(results),
            "overrides_applied": 0,
            "contradictions": 0,
            "override_sleeve": 0,
            "override_media": 0,
            "override_both": 0,
            "override_by_grade": {},
        }

        for result in results:
            meta = result["metadata"]

            if meta.get("contradiction_detected"):
                stats["contradictions"] += 1

            if meta.get("rule_override_applied"):
                stats["overrides_applied"] += 1
                target = meta.get("rule_override_target")

                if target == "both":
                    stats["override_both"] += 1
                elif target == SLEEVE:
                    stats["override_sleeve"] += 1
                    grade = result["predicted_sleeve_condition"]
                    stats["override_by_grade"][grade] = (
                        stats["override_by_grade"].get(grade, 0) + 1
                    )
                elif target == MEDIA:
                    stats["override_media"] += 1
                    grade = result["predicted_media_condition"]
                    stats["override_by_grade"][grade] = (
                        stats["override_by_grade"].get(grade, 0) + 1
                    )

        stats["override_rate"] = round(
            stats["overrides_applied"] / max(stats["total"], 1), 4
        )
        stats["contradiction_rate"] = round(
            stats["contradictions"] / max(stats["total"], 1), 4
        )

        return stats

    def coverage_report(
        self,
        predictions: list[GraderPrediction],
        texts: list[str],
    ) -> dict:
        """
        Compute rule engine coverage statistics over a dataset.
        Useful for evaluating how often rules fire and which grades
        are most frequently overridden.

        Returns:
            Dict with coverage statistics for MLflow logging.
        """
        if len(predictions) != len(texts):
            raise ValueError("predictions and texts must have the same length.")

        applied = self.apply_batch(predictions, texts)
        return self.summarize_results(applied)
