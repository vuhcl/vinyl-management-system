"""YAML-driven pattern compilation for soft/hard signals."""

from __future__ import annotations

import re
from typing import Optional

from .constants import MEDIA, SLEEVE, _HARD_PATTERN_KEYS, _SOFT_PATTERN_KEYS


class RuleEnginePatternMixin:
    """Compile and resolve regex patterns from grading guidelines."""

    _patterns: dict[str, dict[str, list[re.Pattern]]]
    _hard_sources: dict[str, dict[str, list[str]]]
    _hard_patterns: dict[str, dict[str, list[re.Pattern]]]
    grade_defs: dict

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

        For hard-signal keys we store both the source strings (for
        logging / diagnostics) and compiled patterns side-by-side:

        ``self._hard_sources[grade][key]  = ["skip", "skipping", ...]``
        ``self._hard_patterns[grade][key] = [re.Pattern, ...]``

        Soft / forbidden / exception keys live in ``self._patterns``
        as before (compiled pattern lists).
        """
        self._hard_sources = {}
        self._hard_patterns = {}

        for grade, grade_def in self.grade_defs.items():
            self._patterns[grade] = {}
            self._hard_sources[grade] = {}
            self._hard_patterns[grade] = {}

            for hkey in _HARD_PATTERN_KEYS:
                signals = grade_def.get(hkey)
                if signals is None:
                    continue
                source_list = [str(s) for s in signals]
                self._hard_sources[grade][hkey] = source_list
                self._hard_patterns[grade][hkey] = [
                    self._compile_signal(s) for s in source_list
                ]
            # Keep legacy ``hard_signals`` exposed on ``_patterns`` for any
            # external callers (e.g. ``detect_signals``) that look it up
            # through the old pattern map.
            self._patterns[grade]["hard_signals"] = (
                self._hard_patterns[grade].get("hard_signals", [])
            )

            for signal_type in _SOFT_PATTERN_KEYS:
                signals = grade_def.get(signal_type)
                if signals is not None:
                    self._patterns[grade][signal_type] = [
                        self._compile_signal(s) for s in signals
                    ]

    def _resolve_hard_patterns(
        self,
        grade: str,
        target: str,
    ) -> tuple[list[tuple[str, re.Pattern]], list[tuple[str, re.Pattern]]]:
        """
        Resolve the (strict, cosignal) hard-signal pairs for a grade on
        a given target applying the fallback chain from the plan §13/§13b:

        1. Per-target ``hard_signals_{strict,cosignal}_{sleeve,media}``.
        2. Untargeted ``hard_signals_{strict,cosignal}``.
        3. Legacy ``hard_signals`` (treated as strict).

        Each list contains ``(source_string, compiled_pattern)`` tuples
        so the caller can report which entry actually fired.
        """
        sources = self._hard_sources.get(grade, {})
        patterns = self._hard_patterns.get(grade, {})

        def _pair(key: str) -> list[tuple[str, re.Pattern]]:
            srcs = sources.get(key, [])
            pats = patterns.get(key, [])
            return list(zip(srcs, pats))

        strict_key = f"hard_signals_strict_{target}"
        cosig_key = f"hard_signals_cosignal_{target}"

        strict = _pair(strict_key) or _pair("hard_signals_strict")
        cosignal = _pair(cosig_key) or _pair(
            "hard_signals_cosignal"
        )

        if not strict and not cosignal:
            # Back-compat: legacy ``hard_signals`` behaves as strict-only.
            strict = _pair("hard_signals")

        return strict, cosignal

    def _supporting_patterns(
        self, grade: str, target: str
    ) -> list[re.Pattern]:
        pats = self._patterns.get(grade, {})
        if target == SLEEVE and "supporting_signals_sleeve" in pats:
            return pats["supporting_signals_sleeve"]
        if target == MEDIA and "supporting_signals_media" in pats:
            return pats["supporting_signals_media"]
        return pats.get("supporting_signals", [])

    def _exception_patterns(
        self, grade: str, target: str
    ) -> list[re.Pattern]:
        """Strip matches from text before the forbidden check."""
        pats = self._patterns.get(grade, {})
        if target == SLEEVE and "forbidden_exceptions_sleeve" in pats:
            return pats["forbidden_exceptions_sleeve"]
        if target == MEDIA and "forbidden_exceptions_media" in pats:
            return pats["forbidden_exceptions_media"]
        return pats.get("forbidden_exceptions", [])

    def _apply_exceptions(
        self, text: str, grade: str, target: str
    ) -> str:
        """Strip exception phrases so they cannot trigger forbidden signals."""
        exc_pats = self._exception_patterns(grade, target)
        for pat in exc_pats:
            text = pat.sub(" ", text)
        return text

    def _forbidden_patterns(
        self, grade: str, target: str
    ) -> list[re.Pattern]:
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
