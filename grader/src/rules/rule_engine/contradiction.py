"""Contradiction pairs and generic signal matching."""

from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


class RuleEngineContradictionMixin:
    """Contradiction compilation and signal detection helpers."""

    guidelines: dict
    _contradiction_patterns: list[tuple[re.Pattern, re.Pattern]]
    _patterns: dict[str, dict[str, list[re.Pattern]]]

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
            signal_type: "hard_signals" | "supporting_signals" |
                "forbidden_signals"
            target: optional "sleeve" | "media" — for supporting/forbidden,
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
