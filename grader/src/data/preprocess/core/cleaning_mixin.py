"""Lowercase, promo strip, abbreviation expansion, TF-IDF-normalize."""

from __future__ import annotations

import re
from typing import Any

from ..listing_promo import load_promo_noise_patterns, strip_listing_promo_noise


class PreprocessorCleaningMixin:
    def _lowercase(self, text: str) -> str:
        return text.lower() if self.do_lowercase else text

    def _normalize_whitespace(self, text: str) -> str:
        if not self.do_normalize_whitespace:
            return text
        # Collapse multiple whitespace characters into a single space
        # and strip leading/trailing whitespace
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _strip_leading_numeric_boilerplate(text: str) -> str:
        """
        Drop a lone leading catalog/index digit before obvious condition
        boilerplate (e.g. ``6 sealed, new hype sticker`` → ``sealed, …``).
        Does not strip counts that are part of a format description
        (``2 lp``, ``7\"``, ``disk 2 of 3``, ``6 inch``, ``3 inches``).
        """
        s = text.strip()
        m = re.match(
            r"^(\d{1,2})\s+",
            s,
            flags=re.IGNORECASE,
        )
        if not m:
            return text
        rest = s[m.end() :].lstrip()
        rest_lower = rest.lower()
        if re.match(r'^(?:\d{1,2}\s*["\u201d]|\d+\s*/\d+)', rest_lower):
            return text
        if re.match(
            r"^(?:\d+\s+lp\b|\d+\s+inch\b|\d+\s+inches\b|disk\s+\d+\s+of\b)",
            rest_lower,
        ):
            return text
        if re.match(
            r"^(?:sealed|new\s+hype|new\b|nm\b|mint\b|vg\+|vg\b|ex\b|poor\b|good\b)",
            rest_lower,
        ):
            return rest
        return text

    def _expand_abbreviations(self, text: str) -> str:
        """
        Expand abbreviations using ordered regex patterns.
        Order is preserved from grader.yaml abbreviation_map.
        Longer patterns (vg++) are applied before shorter ones (vg+)
        to prevent partial match corruption.

        When an abbreviation is immediately followed by a letter (e.g.
        ``vg+original`` with no space), append a single space so the
        expansion does not glue to the next word (``… plus original``).
        """
        for pattern, expansion in self.abbreviation_patterns:

            def _repl(m: re.Match[str], exp: str = expansion) -> str:
                s = m.string
                j = m.end()
                if j < len(s) and s[j].isalpha():
                    return f"{exp} "
                return exp

            text = pattern.sub(_repl, text)
        return text

    def _verify_protected_terms(
        self, original: str, cleaned: str
    ) -> list[str]:
        """
        Sanity check — verify that protected terms present in the
        original text as **whole tokens** still appear that way in the cleaned
        text (``\\b`` word boundaries + ``re.escape`` per term).

        Returns list of terms that were lost during transformation.
        A non-empty list indicates a preprocessing bug or an aggressive strip
        that removed a real defect token.
        """
        lost: list[str] = []
        for term, pat in self._protected_term_token_patterns.items():
            if pat.search(original) and not pat.search(cleaned):
                lost.append(term)
        return lost

    def clean_text(self, text: str) -> str:
        """
        Apply full text normalization pipeline in correct order.
        Runs on text AFTER detection steps have completed.

        Steps:
          1. Lowercase
          2. Normalize whitespace
          3. Strip listing promo / shipping boilerplate
          4. Optionally strip leading catalog digit (``strip_stray_numeric_tokens``)
          5. Expand abbreviations
        """
        cleaned = self._lowercase(text)
        cleaned = self._normalize_whitespace(cleaned)
        cleaned = strip_listing_promo_noise(
            cleaned,
            self.promo_noise_patterns,
            protected_term_patterns=self._protected_term_token_patterns,
        )
        if self.strip_stray_numeric_tokens:
            cleaned = self._strip_leading_numeric_boilerplate(cleaned)
        cleaned = self._expand_abbreviations(cleaned)
        return cleaned

    @classmethod
    def normalize_text_for_tfidf(
        cls,
        text: str,
        *,
        preprocessing_cfg: dict[str, Any],
        protected_term_patterns: dict[str, re.Pattern[str]] | None = None,
    ) -> str:
        """
        Match ``clean_text`` through promo stripping and leading-digit cleanup,
        but **omit** abbreviation expansion so TF-IDF sees the same tokens as
        ``text_clean`` from preprocess (which already expanded abbrevs).

        Pass the same ``protected_term_patterns`` as ``Preprocessor`` uses
        (from ``build_protected_term_token_patterns(guidelines)``) so TF-IDF
        skips the same gated ``###`` / ``[]`` / ``***`` spans as training.
        """
        pp = preprocessing_cfg
        s = text.strip()
        if pp.get("lowercase", True):
            s = s.lower()
        if pp.get("normalize_whitespace", True):
            s = re.sub(r"\s+", " ", s).strip()
        phrases = load_promo_noise_patterns(pp)
        s = strip_listing_promo_noise(
            s, phrases, protected_term_patterns=protected_term_patterns
        )
        if bool(pp.get("strip_stray_numeric_tokens", True)):
            s = cls._strip_leading_numeric_boilerplate(s)
        return s

