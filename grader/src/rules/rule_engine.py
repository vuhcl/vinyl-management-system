"""
grader/src/rules/rule_engine.py

Post-processing rule engine for vinyl condition grading.
Applies grading rubric rules on top of model predictions.

Rule priority (strictly enforced):
  1. Contradiction detection — flag and suppress all overrides
  2. Hard signal overrides   — Poor, Generic (unconditional)
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

# Optional YAML keys — compiled when present (see _compile_patterns).
_SOFT_PATTERN_KEYS = (
    "supporting_signals",
    "forbidden_signals",
    "supporting_signals_sleeve",
    "supporting_signals_media",
    "forbidden_signals_sleeve",
    "forbidden_signals_media",
    # Exception phrases: matched substrings are stripped from text before the
    # forbidden check, allowing them to neutralise specific forbidden signals.
    # E.g. "sticker residue" as an exception cancels the "sticker" forbidden so
    # a VG+ override is not blocked just because residue was left behind.
    "forbidden_exceptions_sleeve",
    "forbidden_exceptions_media",
)

import yaml

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# Grades owned exclusively by the rule engine (hard + soft overrides apply)
HARD_OWNED_GRADES = {"Poor", "Generic"}

# Grades owned exclusively by the model — rule engine never overrides these.
# Mint hard-override fires correctly for a subset (sealed listings) but causes
# large-scale harm across the full dataset; model accuracy is higher overall.
MODEL_ONLY_GRADES = {"Mint"}


# Targets
SLEEVE = "sleeve"
MEDIA = "media"

# Poor hard_signals that describe playback / platter issues — apply to media only,
# not sleeve (seller notes often describe disc and jacket in one blob).
POOR_MEDIA_ONLY_HARD_SIGNALS = frozenset(
    {
        "skip",
        "skipping",
        "won't play",
        "does not play",
        "unplayable",
        "badly warped",  # warping is a playback/disc issue — not sleeve
        "broken",        # "broken" in seller notes usually describes packaging/OBI/seal;
        "cracked",       # "cracked" similarly — only applies to media (cracked disc)
    }
)

# Poor hard_signals that describe structural sleeve damage — apply to sleeve only,
# not media ("completely split" in seller notes almost always refers to a seam split,
# never a cracked record; disc splits are caught by "cracked"/"broken" above).
POOR_SLEEVE_ONLY_HARD_SIGNALS = frozenset(
    {
        "completely split",
    }
)


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

        self._nm_sleeve_downgrade_patterns = [
            self._compile_signal("seam split"),
            self._compile_signal("spine split"),
        ]
        self._nm_small_split_patterns = [
            self._compile_signal("small seam split"),
            self._compile_signal("small top seam split"),
            self._compile_signal("small seam split along"),
            self._compile_signal("small seam split on top"),
            self._compile_signal("small spine split"),
            self._compile_signal("small top spine split"),
        ]
        self._nm_split_other_defect_patterns = [
            self._compile_signal(s)
            for s in [
                "scratch",
                "scratches",
                "scuff",
                "scuffs",
                "wear",
                "wrinkle",
                "wrinkles",
                "ring wear",
                "ringwear",
                "stain",
                "stains",
                "foxing",
                "mold",
                "water damage",
                "corner",
                "bump",
                "cutout",
                "cut out",
                "tear",
                "tears",
                "fade",
                "writing",
                "sticker",
                "gouge",
                "gouges",
            ]
        ]

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
            hard = grade_def.get("hard_signals", [])
            self._patterns[grade]["hard_signals"] = [
                self._compile_signal(s) for s in hard
            ]
            for signal_type in _SOFT_PATTERN_KEYS:
                signals = grade_def.get(signal_type)
                if signals is not None:
                    self._patterns[grade][signal_type] = [
                        self._compile_signal(s) for s in signals
                    ]

    def _supporting_patterns(self, grade: str, target: str) -> list[re.Pattern]:
        pats = self._patterns.get(grade, {})
        if target == SLEEVE and "supporting_signals_sleeve" in pats:
            return pats["supporting_signals_sleeve"]
        if target == MEDIA and "supporting_signals_media" in pats:
            return pats["supporting_signals_media"]
        return pats.get("supporting_signals", [])

    def _exception_patterns(self, grade: str, target: str) -> list[re.Pattern]:
        """Patterns whose matches are stripped from text before the forbidden check."""
        pats = self._patterns.get(grade, {})
        if target == SLEEVE and "forbidden_exceptions_sleeve" in pats:
            return pats["forbidden_exceptions_sleeve"]
        if target == MEDIA and "forbidden_exceptions_media" in pats:
            return pats["forbidden_exceptions_media"]
        return pats.get("forbidden_exceptions", [])

    def _apply_exceptions(self, text: str, grade: str, target: str) -> str:
        """Strip exception phrases from text so they cannot trigger forbidden signals."""
        exc_pats = self._exception_patterns(grade, target)
        for pat in exc_pats:
            text = pat.sub(" ", text)
        return text

    def _forbidden_patterns(self, grade: str, target: str) -> list[re.Pattern]:
        pats = self._patterns.get(grade, {})
        if target == SLEEVE and "forbidden_signals_sleeve" in pats:
            return pats["forbidden_signals_sleeve"]
        if target == MEDIA and "forbidden_signals_media" in pats:
            return pats["forbidden_signals_media"]
        return pats.get("forbidden_signals", [])

    @staticmethod
    def _min_supporting_for(grade_def: dict, target: str) -> int:
        key = f"min_supporting_{target}"
        if grade_def.get(key) is not None:
            return int(grade_def[key])
        fallback = grade_def.get("min_supporting")
        if fallback is not None:
            return int(fallback)
        return 2

    @staticmethod
    def _max_supporting_for(grade_def: dict, target: str) -> Optional[int]:
        key = f"max_supporting_{target}"
        if grade_def.get(key) is not None:
            return int(grade_def[key])
        fallback = grade_def.get("max_supporting")
        if fallback is not None:
            return int(fallback)
        return None

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
        self,
        text: str,
        grade: str,
        signal_type: str,
        *,
        target: Optional[str] = None,
    ) -> list[str]:
        """
        Detect all signals of a given type for a given grade in text.

        Args:
            text:        preprocessed seller notes (text_clean)
            grade:       canonical grade name
            signal_type: "hard_signals" | "supporting_signals" | "forbidden_signals"
            target:      optional "sleeve" | "media" — for supporting/forbidden,
                selects per-target lists when defined in YAML.

        Returns:
            List of matched signal pattern strings.
        """
        if signal_type == "hard_signals":
            patterns = self._patterns.get(grade, {}).get("hard_signals", [])
        elif signal_type == "supporting_signals":
            if target:
                patterns = self._supporting_patterns(grade, target)
            else:
                patterns = self._patterns.get(grade, {}).get(
                    "supporting_signals", []
                )
        elif signal_type == "forbidden_signals":
            if target:
                patterns = self._forbidden_patterns(grade, target)
            else:
                patterns = self._patterns.get(grade, {}).get(
                    "forbidden_signals", []
                )
        else:
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

        Hard-owned grades: Generic and Poor only.
        Mint is model-owned — the sealed/unplayed hard signal fires correctly
        for a subset of listings but causes large-scale harm across the full
        dataset because "sealed" appears in non-Mint descriptions.

        Poor + sleeve: playback hard signals (``skip``, ``skipping``, etc.) are
        not evaluated — those apply to media only when both are in one note.
        """
        for grade in ["Generic", "Poor"]:
            grade_def = self.grade_defs.get(grade, {})

            # Generic never applies to media
            applies_to = grade_def.get("applies_to", [SLEEVE, MEDIA])
            if target not in applies_to:
                continue

            hard_signal_sources = grade_def.get("hard_signals", [])
            hard_patterns = self._patterns[grade]["hard_signals"]
            if grade == "Poor" and target == SLEEVE:
                hard_patterns = [
                    pat
                    for sig, pat in zip(hard_signal_sources, hard_patterns)
                    if sig not in POOR_MEDIA_ONLY_HARD_SIGNALS
                ]
            if grade == "Poor" and target == MEDIA:
                hard_patterns = [
                    pat
                    for sig, pat in zip(hard_signal_sources, hard_patterns)
                    if sig not in POOR_SLEEVE_ONLY_HARD_SIGNALS
                ]
            forbidden_signals = self._forbidden_patterns(grade, target)

            # Check for any hard signal match
            hard_matches = self._match_signals(text, hard_patterns)
            if not hard_matches:
                continue

            # Strip exception phrases before forbidden check
            text_for_forbidden = self._apply_exceptions(text, grade, target)
            # Check for forbidden signals — block override if present
            forbidden_matches = self._match_signals(text_for_forbidden, forbidden_signals)
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
          2. At most max_supporting distinct supporting signals match (if set);
             above the cap, the listing is treated as too defect-dense for
             that grade and a lower grade is considered instead
          3. No forbidden signals for that grade are present
          4. Model confidence is below the grade's rule_confidence_threshold

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
        # Skip hard-owned grades (Poor/Generic) and model-only grades (Mint)
        candidate_grades = sorted(
            [
                g for g in self.grade_defs
                if g not in HARD_OWNED_GRADES and g not in MODEL_ONLY_GRADES
            ],
            key=lambda g: ordinal_map.get(g, 99),
        )

        for grade in candidate_grades:
            grade_def = self.grade_defs[grade]

            # Check target applicability
            applies_to = grade_def.get("applies_to", [SLEEVE, MEDIA])
            if target not in applies_to:
                continue

            grade_ord = ordinal_map.get(grade, 99)
            pred_ord = ordinal_map.get(predicted_grade, 99)

            # only_downgrade: skip if predicted grade is already this grade or worse
            if grade_def.get("only_downgrade") and pred_ord >= grade_ord:
                continue

            # Global upgrade-step cap: upgrades larger than max_soft_upgrade_steps
            # ordinal positions are blocked (e.g. Good → VG+ jumps 2 steps).
            max_upgrade_steps = self.guidelines.get("max_soft_upgrade_steps", 1)
            if pred_ord > grade_ord and (pred_ord - grade_ord) > max_upgrade_steps:
                continue

            # Skip if model is already confident — trust the model
            threshold = grade_def.get("rule_confidence_threshold", 0.85)
            if model_confidence >= threshold:
                continue

            min_supporting = self._min_supporting_for(grade_def, target)

            # Check forbidden signals — skip this grade if any present.
            # Strip exception phrases first so they don't trigger forbidden signals.
            text_for_forbidden = self._apply_exceptions(text, grade, target)
            forbidden_matches = self._match_signals(
                text_for_forbidden, self._forbidden_patterns(grade, target)
            )
            if forbidden_matches:
                continue

            supporting_matches = self._match_signals(
                text, self._supporting_patterns(grade, target)
            )
            if len(supporting_matches) < min_supporting:
                continue

            max_supporting = self._max_supporting_for(grade_def, target)
            if max_supporting is not None and len(supporting_matches) > max_supporting:
                logger.debug(
                    "Soft override skipped — grade=%s target=%s supporting_count=%d "
                    "exceeds max_supporting=%d",
                    grade,
                    target,
                    len(supporting_matches),
                    max_supporting,
                )
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
            if override_grade is None:
                override_grade = self.check_hard_override(text_lower, target)

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
    def summarize_results(self, results: list[dict]) -> dict:
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

        applied = self.apply_batch(predictions, texts)
        return self.summarize_results(applied)

