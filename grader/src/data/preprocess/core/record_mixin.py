"""Single-record preprocess path."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class PreprocessorRecordMixin:
    def process_record(self, record: dict) -> dict:
        """
        Process a single unified record. Returns a new dict with:
          - text_clean:       normalized, expanded text
          - media_verifiable: re-detected from raw text
          - All original fields preserved unchanged

        Detection runs on original text.
        Cleaning runs after detection.
        """
        raw_text = record.get("text", "")

        # Step 1 & 2: detection on raw text
        media_verifiable = self.detect_unverified_media(raw_text)
        media_evidence_strength = self.detect_media_evidence_strength(raw_text)
        text_based_generic = self.detect_generic_sleeve(raw_text)

        # Step 3-5: text normalization
        text_clean = self.clean_text(raw_text)

        # Step 6: protected term sanity check
        lost_terms = self._verify_protected_terms(raw_text, text_clean)
        if lost_terms:
            logger.warning(
                "Protected terms lost during cleaning for item_id=%s: %s",
                record.get("item_id", "?"),
                lost_terms,
            )
            self._stats["protected_terms_lost"] += 1

        # Build output record — original fields preserved, new fields appended
        processed = {**record}
        processed["text_clean"] = text_clean
        processed["media_verifiable"] = media_verifiable
        processed["media_evidence_strength"] = media_evidence_strength

        dq = self.compute_description_quality(
            raw_text,
            text_clean,
            sleeve_label=str(record.get("sleeve_label") or ""),
            media_label=str(record.get("media_label") or ""),
        )
        processed.update(dq)

        # If text-based Generic detection fires but sleeve_label is not
        # already Generic, log for review — do not silently override label.
        # Label integrity is paramount; discrepancies are flagged, not fixed.
        if text_based_generic and record.get("sleeve_label") != "Generic":
            logger.debug(
                "Generic signal in text but sleeve_label=%r for item_id=%s. "
                "Label preserved — review may be needed.",
                record.get("sleeve_label"),
                record.get("item_id", "?"),
            )
            self._stats["generic_text_label_mismatch"] += 1

        return processed

