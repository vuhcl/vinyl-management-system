"""Signals, detection, and text cleaning for the preprocess pipeline."""

from __future__ import annotations

import re
from typing import Any

from ..listing_promo import (
    load_promo_noise_patterns,
    protected_terms_from_grades,
    strip_listing_promo_noise,
)


class _PreprocessorCleaning:
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

    def detect_unverified_media(self, text: str) -> bool:
        """
        Returns False (unverifiable) if any unverified media signal
        is found in the raw text. Case-insensitive.

        Must be called before any text transformation.
        """
        text_lower = text.lower()
        # 1) Hard unverified signals always win.
        if any(signal in text_lower for signal in self.unverified_signals):
            return False

        # 2) Sealed/Mint exemption: sealed implies Mint media by convention
        # in this project, even if playback isn't described.
        if any(sig in text_lower for sig in self.mint_hard_signals):
            return True

        # 3) If no playback/media cue exists in the comment, mark unverified.
        if any(cue in text_lower for cue in self.media_verifiable_cues):
            return True

        # 4) Comments that mention the media object + defect language
        # (e.g., "Vinyl has light surface marks", "labels bubbling")
        # count as verifiable media descriptions.
        has_media_subject = any(
            term in text_lower for term in self.media_subject_terms
        )
        has_media_condition = any(
            term in text_lower for term in self.media_condition_terms
        )
        if has_media_subject and has_media_condition:
            return True

        # 5) Explicit play-testing language (may not appear in guideline-derived cues).
        play_markers = (
            "plays perfectly",
            "plays great",
            "plays fine",
            "plays well",
            "plays cleanly",
            "plays through",
            "tested",
            "spin tested",
        )
        if any(m in text_lower for m in play_markers):
            return True

        return False

    def detect_media_evidence_strength(self, text: str) -> str:
        """
        Estimate how directly the comment describes playable media condition.

        Returns one of:
          - "none": no usable media evidence
          - "weak": indirect/limited media evidence
          - "strong": clear playback/media-condition evidence
        """
        text_lower = text.lower()
        if any(signal in text_lower for signal in self.unverified_signals):
            return "none"

        playback_hits = sum(
            1 for cue in self.media_verifiable_cues if cue in text_lower
        )
        has_media_subject = any(
            term in text_lower for term in self.media_subject_terms
        )
        has_media_condition = any(
            term in text_lower for term in self.media_condition_terms
        )

        if playback_hits >= 2:
            return "strong"
        if playback_hits >= 1 and (has_media_subject or has_media_condition):
            return "strong"
        if has_media_subject and has_media_condition:
            return "weak"
        if any(sig in text_lower for sig in self.mint_hard_signals):
            # Sealed/unopened gives a convention-based high media label,
            # but textual evidence for actual playback condition is limited.
            return "weak"
        return "none"

    def detect_generic_sleeve(self, text: str) -> bool:
        """
        Returns True if any Generic hard signal is found in raw text.
        Used to re-confirm Generic sleeve labels from text alone.

        Note: sleeve_label may already be Generic from ingestion.
        This is a secondary text-based detection for records where
        the API condition field was ambiguous.

        Must be called before any text transformation.
        """
        text_lower = text.lower()
        return any(signal in text_lower for signal in self.generic_signals)

    def _count_distinct_grade_phrases(self, text_clean_lower: str) -> int:
        """How many canonical grade phrases appear (after abbreviation expansion)."""
        return sum(1 for phrase in self._grade_phrases if phrase in text_clean_lower)

    def sleeve_note_adequate(self, text_clean: str) -> bool:
        """
        True if the note plausibly describes jacket/sleeve/packaging,
        or uses multi-grade shorthand (e.g. NM/VG), or is long free text.
        """
        t = text_clean.strip().lower()
        if not t:
            return False
        if any(hint in t for hint in self.sleeve_evidence_terms):
            return True
        if self._count_distinct_grade_phrases(t) >= 2:
            return True
        if len(text_clean.strip()) >= self.min_chars_sleeve_fallback:
            return True
        return False

    def media_note_adequate(self, raw_text: str) -> bool:
        """True when media_evidence_strength is not ``none`` (see detect_*)."""
        return self.detect_media_evidence_strength(raw_text) != "none"

    def _mint_listing_sleeve_relaxed_ok(
        self, raw_text: str, text_clean: str
    ) -> bool:
        """Short sealed / Mint-ish copy counts as sleeve evidence."""
        blob = f"{raw_text} {text_clean}".lower()
        return any(h in blob for h in self.mint_sleeve_relax_substrings)

    def compute_description_quality(
        self,
        raw_text: str,
        text_clean: str,
        *,
        sleeve_label: str | None = None,
        media_label: str | None = None,
    ) -> dict:
        """
        Fields for training filter and inference UX.

        Returns keys: sleeve_note_adequate, media_note_adequate,
        adequate_for_training, description_quality_gaps (list[str]),
        description_quality_prompts (list[str]), needs_richer_note (bool).

        When ``mint_sleeve_label_relax_sleeve_note`` is enabled in config and
        ``sleeve_label`` is ``Mint``, sleeve adequacy also passes if the note
        contains Mint listing phrases (sealed, shrink, brand new, …).
        """
        if not self.description_adequacy_enabled:
            return {
                "sleeve_note_adequate": True,
                "media_note_adequate": True,
                "adequate_for_training": True,
                "description_quality_gaps": [],
                "description_quality_prompts": [],
                "needs_richer_note": False,
            }

        sleeve_ok = self.sleeve_note_adequate(text_clean)
        if (
            not sleeve_ok
            and self.mint_sleeve_label_relax_sleeve_note
            and str(sleeve_label or "").strip() == "Mint"
            and self._mint_listing_sleeve_relaxed_ok(raw_text, text_clean)
        ):
            sleeve_ok = True
        media_ok = self.media_note_adequate(raw_text)
        gaps: list[str] = []
        prompts: list[str] = []
        if not sleeve_ok:
            gaps.append("sleeve")
            prompts.append(self.prompt_sleeve)
        if not media_ok:
            gaps.append("media")
            prompts.append(self.prompt_media)

        if self.require_both_for_training:
            train_ok = sleeve_ok and media_ok
        else:
            train_ok = sleeve_ok or media_ok

        return {
            "sleeve_note_adequate": sleeve_ok,
            "media_note_adequate": media_ok,
            "adequate_for_training": train_ok,
            "description_quality_gaps": gaps,
            "description_quality_prompts": prompts,
            "needs_richer_note": bool(gaps),
        }

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
