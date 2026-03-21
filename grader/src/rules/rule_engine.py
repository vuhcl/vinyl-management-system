"""
grader/src/rules/rule_engine.py

Post-processing rule engine for vinyl condition grading.
Applies grading rubric rules on top of model predictions.

Rule priority (strictly enforced):
  1. Contradiction detection — flag and suppress all overrides
  2. Hard signal overrides   — Mint, Poor, Generic (unconditional)
  3. Soft signal overrides   — other grades (confidence-gated)

Operates independently on sleeve and media targets.
Generic checks are skipped for media — Generic is sleeve-only.

Input:  prediction dict from baseline.py or transformer.py
Output: same schema with updated grade and metadata fields

Usage:
    from grader.src.rules.rule_engine import RuleEngine

    engine = RuleEngine(guidelines_path="grader/configs/grading_guidelines.yaml")
    prediction = engine.apply(prediction, text)
    predictions = engine.apply_batch(predictions, texts)
"""

import logging
import re
from typing import Optional

import yaml

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# Grades owned exclusively by the rule engine
HARD_OWNED_GRADES = {"Mint", "Poor", "Generic"}

# Targets
SLEEVE = "sleeve"
MEDIA = "media"


# ---------------------------------------------------------------------------
# RuleEngine
# ---------------------------------------------------------------------------
class RuleEngine:
    """
    Post-processing rule engine grounding model predictions in the
    official Discogs grading rubric and curated signal lists.

    All signal lists, thresholds, and grade ownership are loaded
    from grading_guidelines.yaml — nothing is hardcoded.

    Signal matching uses compiled word-boundary regex patterns for
    accuracy. Partial matches (e.g. "vg+" inside "vg++") are prevented
    by the boundary anchors.

    Args:
        guidelines_path: path to grading_guidelines.yaml
    """

    def __init__(self, guidelines_path: str) -> None:
        self.guidelines = self._load_yaml(guidelines_path)
        self.grade_defs = self.guidelines["grades"]

        # Pre-compile all signal patterns for performance
        # Structure: {grade: {signal_type: [compiled_pattern, ...]}}
        self._patterns: dict[str, dict[str, list[re.Pattern]]] = {}
        self._compile_patterns()

        # Pre-compile contradiction pairs
        self._contradiction_patterns: list[tuple[re.Pattern, re.Pattern]] = (
            self._compile_contradictions()
        )

        logger.info(
            "RuleEngine initialized — %d grades | %d contradiction pairs",
            len(self.grade_defs),
            len(self._contradiction_patterns),
        )

    # -----------------------------------------------------------------------
    # Config loading
    # -----------------------------------------------------------------------
    @staticmethod
    def _load_yaml(path: str) -> dict:
        with open(path, "r") as f:
            return yaml.safe_load(f)

    # -----------------------------------------------------------------------
    # Pattern compilation
    # -----------------------------------------------------------------------
    def _compile_signal(self, signal: str) -> re.Pattern:
        r"""
        Compile a single signal string to a word-boundary regex pattern.
        Case-insensitive. Handles multi-word signals correctly.

        Examples:
            "sealed"        → r"(?<!\w)sealed(?!\w)"
            "seam split"    → r"(?<!\w)seam split(?!\w)"
            "won't play"    → r"(?<!\w)won't play(?!\w)"
        """
        return re.compile(
            r"(?<!\w)" + re.escape(signal.lower()) + r"(?!\w)",
            re.IGNORECASE,
        )

    def _compile_patterns(self) -> None:
        """
        Pre-compile all signal patterns for all grades.
        Stored in self._patterns for reuse across all apply() calls.
        """
        for grade, grade_def in self.grade_defs.items():
            self._patterns[grade] = {}
            for signal_type in [
                "hard_signals",
                "supporting_signals",
                "forbidden_signals",
            ]:
                signals = grade_def.get(signal_type, [])
                self._patterns[grade][signal_type] = [
                    self._compile_signal(s) for s in signals
                ]

    def _compile_contradictions(
        self,
    ) -> list[tuple[re.Pattern, re.Pattern]]:
        """
        Compile contradiction pairs from guidelines.
        Each pair becomes two compiled patterns.
        """
        pairs = []
        for pair in self.guidelines.get("contradictions", []):
            if len(pair) == 2:
                pairs.append(
                    (
                        self._compile_signal(pair[0]),
                        self._compile_signal(pair[1]),
                    )
                )
        return pairs

    # -----------------------------------------------------------------------
    # Signal detection
    # -----------------------------------------------------------------------
    def _match_signals(
        self,
        text: str,
        patterns: list[re.Pattern],
    ) -> list[str]:
        """
        Return list of signal strings that match in the given text.
        """
        return [
            pattern.pattern  # return pattern string for logging
            for pattern in patterns
            if pattern.search(text)
        ]

    def detect_signals(
        self, text: str, grade: str, signal_type: str
    ) -> list[str]:
        """
        Detect all signals of a given type for a given grade in text.

        Args:
            text:        preprocessed seller notes (text_clean)
            grade:       canonical grade name
            signal_type: "hard_signals" | "supporting_signals" | "forbidden_signals"

        Returns:
            List of matched signal pattern strings.
        """
        patterns = self._patterns.get(grade, {}).get(signal_type, [])
        return self._match_signals(text, patterns)

    # -----------------------------------------------------------------------
    # Contradiction detection
    # -----------------------------------------------------------------------
    def check_contradiction(self, text: str) -> bool:
        """
        Check if text contains any contradictory signal pair.

        Returns True if a contradiction is found — rule overrides
        should be suppressed when this returns True.
        """
        text_lower = text.lower()
        for pattern_a, pattern_b in self._contradiction_patterns:
            if pattern_a.search(text_lower) and pattern_b.search(text_lower):
                logger.debug(
                    "Contradiction detected: %r AND %r in text",
                    pattern_a.pattern,
                    pattern_b.pattern,
                )
                return True
        return False

    # -----------------------------------------------------------------------
    # Hard signal override (Mint, Poor, Generic)
    # -----------------------------------------------------------------------
    def check_hard_override(
        self,
        text: str,
        target: str,
    ) -> Optional[str]:
        """
        Check if any hard-owned grade should override for a given target.
        Returns the override grade string or None.

        Hard signal logic:
          - Any single hard signal in the text triggers the override
          - Forbidden signals block the override even if hard signal present
          - Generic checks are skipped for media target

        Priority within hard grades: Mint > Generic > Poor
        (A sealed record with Poor signals → contradiction, handled upstream)
        """
        for grade in ["Mint", "Generic", "Poor"]:
            grade_def = self.grade_defs.get(grade, {})

            # Generic never applies to media
            applies_to = grade_def.get("applies_to", [SLEEVE, MEDIA])
            if target not in applies_to:
                continue

            hard_signals = self._patterns[grade]["hard_signals"]
            forbidden_signals = self._patterns[grade]["forbidden_signals"]

            # Check for any hard signal match
            hard_matches = self._match_signals(text, hard_signals)
            if not hard_matches:
                continue

            # Check for forbidden signals — block override if present
            forbidden_matches = self._match_signals(text, forbidden_signals)
            if forbidden_matches:
                logger.debug(
                    "Hard signal %r blocked by forbidden signal %r "
                    "for grade %s target=%s",
                    hard_matches[0],
                    forbidden_matches[0],
                    grade,
                    target,
                )
                continue

            logger.debug(
                "Hard override triggered — grade=%s target=%s signal=%r",
                grade,
                target,
                hard_matches[0],
            )
            return grade

        return None

    # -----------------------------------------------------------------------
    # Soft signal override (confidence-gated)
    # -----------------------------------------------------------------------
    def check_soft_override(
        self,
        text: str,
        target: str,
        model_confidence: float,
        predicted_grade: str,
    ) -> Optional[str]:
        """
        Check if a soft signal override should fire for a given target.

        Soft override fires only when:
          1. min_supporting signals are present for a candidate grade
          2. No forbidden signals for that grade are present
          3. Model confidence is below the grade's rule_confidence_threshold

        Evaluates all non-hard grades in ordinal order (best to worst).
        Returns the first grade that satisfies all conditions, or None.

        Args:
            text:             preprocessed seller notes
            target:           "sleeve" or "media"
            model_confidence: max predicted probability from model
            predicted_grade:  current model-predicted grade string

        Returns:
            Override grade string or None.
        """
        ordinal_map = self.guidelines.get("grade_ordinal_map", {})

        # Evaluate grades in ordinal order — best to worst
        candidate_grades = sorted(
            [g for g in self.grade_defs if g not in HARD_OWNED_GRADES],
            key=lambda g: ordinal_map.get(g, 99),
        )

        for grade in candidate_grades:
            grade_def = self.grade_defs[grade]

            # Check target applicability
            applies_to = grade_def.get("applies_to", [SLEEVE, MEDIA])
            if target not in applies_to:
                continue

            # Skip if model is already confident — trust the model
            threshold = grade_def.get("rule_confidence_threshold", 0.85)
            if model_confidence >= threshold:
                continue

            min_supporting = grade_def.get("min_supporting", 2)

            # Check forbidden signals — skip this grade if any present
            forbidden_matches = self._match_signals(
                text, self._patterns[grade]["forbidden_signals"]
            )
            if forbidden_matches:
                continue

            # Check supporting signals
            supporting_matches = self._match_signals(
                text, self._patterns[grade]["supporting_signals"]
            )
            if len(supporting_matches) < min_supporting:
                continue

            # All conditions met — but only override if different from
            # current prediction to avoid no-op overrides in metadata
            if grade != predicted_grade:
                logger.debug(
                    "Soft override triggered — grade=%s target=%s "
                    "supporting=%r model_confidence=%.3f threshold=%.2f",
                    grade,
                    target,
                    supporting_matches,
                    model_confidence,
                    threshold,
                )
                return grade

        return None

    # -----------------------------------------------------------------------
    # Single prediction application
    # -----------------------------------------------------------------------
    def apply(
        self,
        prediction: dict,
        text: str,
    ) -> dict:
        """
        Apply rule engine to a single prediction dict.

        Modifies a copy of the prediction — original is never mutated.
        Runs full rule logic independently for sleeve and media.

        Args:
            prediction: prediction dict from baseline.py or transformer.py
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

            # Step 2 — Hard signal override (Mint, Poor, Generic)
            override_grade = self.check_hard_override(text_lower, target)

            # Step 3 — Soft signal override (if no hard override)
            if override_grade is None:
                override_grade = self.check_soft_override(
                    text_lower,
                    target,
                    model_confidence,
                    predicted_grade,
                )

            if override_grade is not None:
                # Apply override
                if target == SLEEVE:
                    result["predicted_sleeve_condition"] = override_grade
                else:
                    result["predicted_media_condition"] = override_grade
                override_targets.append(target)

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

        return result

    # -----------------------------------------------------------------------
    # Batch application
    # -----------------------------------------------------------------------
    def apply_batch(
        self,
        predictions: list[dict],
        texts: list[str],
    ) -> list[dict]:
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
    def coverage_report(
        self,
        predictions: list[dict],
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

        stats = {
            "total": len(predictions),
            "overrides_applied": 0,
            "contradictions": 0,
            "override_sleeve": 0,
            "override_media": 0,
            "override_both": 0,
            "override_by_grade": {},
        }

        for prediction, text in zip(predictions, texts):
            result = self.apply(prediction, text)
            meta = result["metadata"]

            if meta["contradiction_detected"]:
                stats["contradictions"] += 1

            if meta["rule_override_applied"]:
                stats["overrides_applied"] += 1
                target = meta["rule_override_target"]

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
