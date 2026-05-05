"""Raw-text detection and description-quality helpers."""

from __future__ import annotations


class PreprocessorDetectionMixin:
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

