"""Hard and soft grade overrides."""

from __future__ import annotations

import logging
import re
from typing import Optional

from .constants import (
    HARD_OWNED_GRADES,
    MEDIA,
    MODEL_ONLY_GRADES,
    SLEEVE,
    _HARD_OWNED_ORDER_DEFAULT,
)

logger = logging.getLogger(__name__)


class RuleEngineOverrideMixin:
    """Hard/soft override evaluation and diagnostics."""

    guidelines: dict
    grade_defs: dict
    _allow_excellent_soft_override: bool
    _last_hard_tier: Optional[str]
    _last_hard_signal: Optional[str]

    # -----------------------------------------------------------------------
    # Hard signal override (Mint, Poor, Generic)
    # -----------------------------------------------------------------------
    def _matched_hard_signals(
        self,
        text: str,
        pairs: list[tuple[str, re.Pattern]],
    ) -> list[str]:
        """Return source strings (not regex patterns) that match in text."""
        return [src for src, pat in pairs if pat.search(text)]

    def _evaluate_hard_match(
        self,
        text: str,
        grade: str,
        target: str,
    ) -> Optional[tuple[str, str]]:
        """
        Return ``(tier, fired_source)`` for the first hard signal that
        triggers, ignoring forbidden checks. ``tier`` is either
        ``"strict"`` or ``"cosignal"``.

        A strict signal fires on a single match. A cosignal fires only
        when **at least one other distinct signal string** from the same
        grade's strict *or* cosignal list also matches in the text
        (distinctness is by the source string, not by match position).
        Returns ``None`` if no hard signal fires.
        """
        strict, cosignal = self._resolve_hard_patterns(grade, target)
        if not strict and not cosignal:
            return None

        strict_matches = self._matched_hard_signals(text, strict)
        cosignal_matches = self._matched_hard_signals(text, cosignal)

        if strict_matches:
            return ("strict", strict_matches[0])

        # Cosignal tier — require at least two distinct matched signals
        # from the same grade's strict or cosignal lists combined.
        distinct_all = set(strict_matches) | set(cosignal_matches)
        if cosignal_matches and len(distinct_all) >= 2:
            return ("cosignal", cosignal_matches[0])

        return None

    def check_hard_override(
        self,
        text: str,
        target: str,
    ) -> Optional[str]:
        """
        Check if any hard-owned grade should override for a given target.
        Returns the override grade string or None.

        Hard signal logic:
          - **Strict** hard signals fire on any single match (legacy
            ``hard_signals`` is treated as strict for back-compat).
          - **Cosignal** hard signals fire only when another distinct
            signal from the same grade's strict or cosignal list also
            matches — the extra evidence guards against risky phrasings
            (e.g. ``skipping`` qualified by "minor").
          - Forbidden signals still block the override either way.
          - Generic checks are skipped for media target.

        Per-target bifurcation (e.g. sleeve-only vs media-only hard
        signals) is driven entirely by YAML via
        ``hard_signals_*_sleeve`` / ``hard_signals_*_media`` keys. The
        Python frozensets that used to encode this split were removed in
        favour of the single-source-of-truth contract documented in the
        rule_engine package docstring.

        Hard-owned grades: Generic and Poor only. Mint is model-owned —
        the sealed/unplayed hard signal fires correctly for a subset of
        listings but causes large-scale harm across the full dataset
        because "sealed" appears in non-Mint descriptions.

        Grades are tried in ``_HARD_OWNED_ORDER_DEFAULT`` (Poor, then
        Generic) so sleeve **Poor** (catastrophic jacket) wins over
        **Generic** (housing type) when both could match.
        """
        for grade in _HARD_OWNED_ORDER_DEFAULT:
            grade_def = self.grade_defs.get(grade, {})
            applies_to = grade_def.get("applies_to", [SLEEVE, MEDIA])
            if target not in applies_to:
                continue

            match = self._evaluate_hard_match(text, grade, target)
            if match is None:
                continue
            tier, fired = match

            forbidden_signals = self._forbidden_patterns(grade, target)
            text_for_forbidden = self._apply_exceptions(text, grade, target)
            forbidden_matches = self._match_signals(
                text_for_forbidden, forbidden_signals
            )
            if forbidden_matches:
                logger.debug(
                    "Hard signal %r (%s tier) blocked by forbidden %r "
                    "for grade %s target=%s",
                    fired,
                    tier,
                    forbidden_matches[0],
                    grade,
                    target,
                )
                continue

            logger.debug(
                "Hard override triggered — grade=%s target=%s "
                "signal=%r tier=%s",
                grade,
                target,
                fired,
                tier,
            )
            # Stash the tier on the last-evaluation cache so ``apply``
            # can surface it in metadata without a second scan.
            self._last_hard_tier = tier
            self._last_hard_signal = fired
            return grade

        return None

    # -----------------------------------------------------------------------
    # Debug helpers — diagnostics for the missed-rule-owned exporter
    # -----------------------------------------------------------------------
    def would_hard_signal_match(
        self,
        text: str,
        target: str,
        grade: str,
    ) -> bool:
        """
        Return True iff the hard-signal evaluation for ``grade`` would
        trigger on ``text`` for the given target, **ignoring forbidden
        signals and exceptions**. Strict signals require a single match,
        cosignals require the usual two-signal corroboration — the
        forbidden layer is the only thing skipped.

        Use this together with :meth:`would_hard_override_fire` to
        distinguish "missing pattern" (both False) from "blocked by a
        forbidden" (pre True, post False) when auditing rule-owned
        false negatives.
        """
        grade_def = self.grade_defs.get(grade, {})
        applies_to = grade_def.get("applies_to", [SLEEVE, MEDIA])
        if target not in applies_to:
            return False
        return self._evaluate_hard_match(text.lower(), grade, target) is not None

    def would_hard_override_fire(
        self,
        text: str,
        target: str,
        grade: str,
    ) -> bool:
        """
        Return True iff :meth:`check_hard_override` would assign
        ``grade`` for this text/target — i.e. a hard signal matches
        **and** no forbidden blocks it.
        """
        return self.check_hard_override(text.lower(), target) == grade

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
                g
                for g in self.grade_defs
                if g not in HARD_OWNED_GRADES and g not in MODEL_ONLY_GRADES
            ],
            key=lambda g: ordinal_map.get(g, 99),
        )

        for grade in candidate_grades:
            if (
                grade == "Excellent"
                and not self._allow_excellent_soft_override
            ):
                continue
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
