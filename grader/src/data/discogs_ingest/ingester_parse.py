"""Parsing, boilerplate stripping, save, MLflow, orchestration."""
from __future__ import annotations

import json
import logging
import re
from typing import Optional

import mlflow

from grader.src.mlflow_tracking import mlflow_pipeline_step_run_ctx

from .normalize import normalize_seller_comment_text

logger = logging.getLogger(__name__)


class DiscogsIngesterParseMixin:
    def normalize_grade(self, raw_condition: str) -> Optional[str]:
        """
        Map a raw Discogs condition string to a canonical grade.
        Returns None if the string is not in the condition map.
        """
        return self.condition_map.get(raw_condition)

    def _normalize_seller_comment_text(self, text: str) -> str:
        return normalize_seller_comment_text(
            text,
            strip_urls=self.strip_seller_comment_urls,
            strip_emoji=self.strip_seller_comment_emoji,
        )

    def _detect_media_verifiable(self, text: str) -> bool:
        """
        Returns False if seller notes contain any unverified media signal.
        Checked case-insensitively.
        """
        text_lower = text.lower()
        return not any(signal in text_lower for signal in self.UNVERIFIED_MEDIA_SIGNALS)

    def _exempt_from_generic_note_drop(
        self, text: str, raw_sleeve: str, raw_media: str
    ) -> bool:
        """
        Keep boilerplate-heavy notes when media or sleeve is Mint, or when
        the comment clearly indicates sealed / new-in-shrink stock.
        """
        if self.normalize_grade(raw_media) == "Mint":
            return True
        if self.normalize_grade(raw_sleeve) == "Mint":
            return True
        tl = text.lower()
        return any(kw in tl for kw in self.preservation_keywords)

    def _matches_generic_note_patterns(self, text: str) -> bool:
        tl = text.lower()
        return any(p.lower() in tl for p in self.generic_note_patterns)

    def _has_item_specific_language(self, text: str) -> bool:
        """True if the note likely describes this copy (not only shop policy)."""
        tl = text.lower()
        return any(h.lower() in tl for h in self.item_specific_hints)

    @staticmethod
    def _split_comment_segments(text: str) -> list[str]:
        """
        Split seller comments into sentence- or line-like chunks for
        boilerplate removal (keeps order).
        """
        text = text.strip()
        if not text:
            return []
        segments: list[str] = []
        for block in re.split(r"\n\s*\n+", text):
            block = block.strip()
            if not block:
                continue
            for line in block.split("\n"):
                line = line.strip()
                if not line:
                    continue
                # Sentence boundaries inside a line
                for sent in re.split(r"(?<=[.!?])\s+", line):
                    s = sent.strip()
                    if s:
                        segments.append(s)
        return segments if segments else [text]

    def _segment_is_boilerplate_only(self, segment: str) -> bool:
        """
        True if this chunk is dominated by configured shop-boilerplate phrases
        and does not carry copy-specific or preservation cues.
        """
        low = segment.lower()
        if not any(p.lower() in low for p in self.generic_note_patterns):
            return False
        if any(h.lower() in low for h in self.item_specific_hints):
            return False
        if any(k.lower() in low for k in self.preservation_keywords):
            return False
        return True

    def _strip_boilerplate_substrings(self, segment: str) -> str:
        """
        Remove known boilerplate phrases inside mixed segments while preserving
        copy-specific condition details.
        """
        out = segment
        for pat in self.generic_note_patterns:
            p = (pat or "").strip()
            if not p:
                continue
            out = re.sub(re.escape(p), " ", out, flags=re.IGNORECASE)
        out = re.sub(r"\(\s*\)", " ", out)
        out = re.sub(r"\s+", " ", out).strip(" .;,-")
        return out

    def strip_boilerplate_from_notes(self, text: str) -> str:
        """
        Remove sentence/line chunks that match generic shop patterns only,
        so training text emphasizes condition-relevant language.

        Uses the same pattern / hint / preservation lists as the generic-note
        drop filter. Runs before drop checks when ``strip_boilerplate`` is on.
        """
        if not self.strip_boilerplate_enabled:
            return text.strip()

        raw = text.strip()
        if not raw:
            return ""

        segments = self._split_comment_segments(raw)
        kept: list[str] = []
        for seg in segments:
            if self._segment_is_boilerplate_only(seg):
                continue
            cleaned = self._strip_boilerplate_substrings(seg)
            if cleaned:
                kept.append(cleaned)

        out = " ".join(kept)
        out = re.sub(r"\s+", " ", out).strip()
        return out

    def parse_listing(self, listing: dict) -> Optional[dict]:
        """
        Extract unified schema fields from a raw Discogs listing dict.
        Returns None if the listing fails any filter condition.
        The raw listing structure is preserved in raw_sleeve / raw_media
        so labels can be re-derived if the condition map changes.
        """
        # Extract condition fields — strip so condition_map matches SQLite/API padding.
        raw_sleeve = (listing.get("sleeve_condition") or "").strip()
        raw_media = (listing.get("condition") or "").strip()  # Discogs: "condition" = media

        # Extract seller notes: normalize (URLs, emoji) then shop boilerplate.
        raw_comments = listing.get("comments", "") or ""
        work = self._normalize_seller_comment_text(raw_comments)
        text = self.strip_boilerplate_from_notes(work)

        # Apply filters — return None with logged reason on failure
        drop_reason = self._get_drop_reason(text, raw_sleeve, raw_media)
        if drop_reason:
            self._stats["drops"][drop_reason] = (
                self._stats["drops"].get(drop_reason, 0) + 1
            )
            return None

        # Normalize grades
        sleeve_label = self.normalize_grade(raw_sleeve)
        media_label = self.normalize_grade(raw_media)

        if sleeve_label is None or media_label is None:
            self._stats["drops"]["unknown_condition_string"] = (
                self._stats["drops"].get("unknown_condition_string", 0) + 1
            )
            logger.debug(
                "Unknown condition string — sleeve: %r, media: %r",
                raw_sleeve,
                raw_media,
            )
            return None

        # Extract optional metadata
        release = listing.get("release", {})
        artist = release.get("artist", "")
        title = release.get("title", "")
        year = release.get("year")
        country = release.get("country", "")
        release_format = (release.get("format") or "").strip()
        release_description = (release.get("description") or "").strip()

        return {
            "item_id": str(listing.get("id", "")),
            "source": "discogs",
            "text": text.strip(),
            "sleeve_label": sleeve_label,
            "media_label": media_label,
            "label_confidence": 1.0,
            "media_verifiable": self._detect_media_verifiable(text),
            "obi_condition": None,
            "raw_sleeve": raw_sleeve,
            "raw_media": raw_media,
            "artist": artist,
            "title": title,
            "year": int(year) if year else None,
            "country": country,
            # Used when vinyl_format_filter_stage is post_patch (pipeline vinyl filter).
            "release_format": release_format,
            "release_description": release_description,
        }

    def _get_drop_reason(
        self, text: str, raw_sleeve: str, raw_media: str
    ) -> Optional[str]:
        """
        Returns a drop reason string if the listing should be excluded,
        or None if the listing passes all filters.
        """
        if not text or not text.strip():
            return "missing_notes"
        if len(text.strip()) < 10:
            return "notes_too_short"
        if not raw_sleeve:
            return "missing_sleeve_condition"
        if not raw_media:
            return "missing_media_condition"

        if self.generic_note_filter_enabled:
            if (
                self._matches_generic_note_patterns(text)
                and not self._exempt_from_generic_note_drop(text, raw_sleeve, raw_media)
                and not self._has_item_specific_language(text)
            ):
                return "generic_seller_notes"

        return None

    # -----------------------------------------------------------------------
    # Processed output
    # -----------------------------------------------------------------------
    def save_processed(self, records: list[dict]) -> None:
        """
        Append processed records to the unified JSONL output file.
        JSONL chosen over CSV — seller notes contain arbitrary text
        (commas, quotes, newlines) that makes CSV parsing brittle.
        """
        # Overwrite — each run is a full snapshot (avoids duplicate lines on re-run).
        with open(self.processed_path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info("Saved %d records to %s", len(records), self.processed_path)

    # -----------------------------------------------------------------------
    # MLflow logging
    # -----------------------------------------------------------------------
    def _log_mlflow(self) -> None:
        mlflow.log_params(
            {
                "source": "discogs",
                "ingest_mode": "user_inventory",
                "target_per_grade": self.target_per_grade,
                "format_filter": self.format_filter,
                "vinyl_format_filter_stage": self.vinyl_format_filter_stage,
                "max_public_inventory_pages": self.max_public_inventory_pages,
                "inventory_per_page": self.inventory_per_page,
                "inventory_format_api_param": self.inventory_format_api_param,
                "inventory_send_limit_param": self.inventory_send_limit_param,
                "cache_only": self.cache_only,
                "inventory_seller_count": len(self.inventory_sellers),
                "strip_boilerplate": self.strip_boilerplate_enabled,
            }
        )
        mlflow.log_metrics(
            {
                "total_fetched": self._stats["total_fetched"],
                "total_dropped": self._stats["total_dropped"],
                "total_saved": self._stats["total_saved"],
            }
        )
        # Per-grade saved counts
        for grade, count in self._stats["per_grade"].items():
            metric_key = f"saved_{grade.lower().replace(' ', '_')}"
            mlflow.log_metric(metric_key, count)

        # Per-drop-reason counts
        for reason, count in self._stats["drops"].items():
            mlflow.log_metric(f"dropped_{reason}", count)

    # -----------------------------------------------------------------------
    # Orchestration
    # -----------------------------------------------------------------------
    def run(self, dry_run: bool = False) -> list[dict]:
        """
        Full ingestion pipeline:
          1. Fetch all listings per grade (with resume logic)
          2. Parse and filter each listing
          3. Save processed records to JSONL
          4. Log metrics to MLflow

        Args:
            dry_run: if True, fetch and parse but do not write output
                     or log to MLflow. Useful for validating auth and
                     API responses without side effects.

        Returns:
            List of processed record dicts.
        """
        # Reset stats
        self._stats = {
            "total_fetched": 0,
            "total_dropped": 0,
            "total_saved": 0,
            "per_grade": {},
            "drops": {},
        }
        self._inventory_format_param_rejected = False
        self._inventory_limit_param_rejected = False
        self._logged_inventory_api_page_size_mismatch = False

        with mlflow_pipeline_step_run_ctx(self.config, "ingest_discogs") as mlf:
            # Fetch raw listings
            all_listings = self.fetch_all()

            processed_records: list[dict] = []

            for canonical_grade, listings in all_listings.items():
                grade_records: list[dict] = []

                for listing in listings:
                    self._stats["total_fetched"] += 1
                    record = self.parse_listing(listing)

                    if record is None:
                        self._stats["total_dropped"] += 1
                        continue

                    grade_records.append(record)

                self._stats["per_grade"][canonical_grade] = len(grade_records)
                self._stats["total_saved"] += len(grade_records)
                processed_records.extend(grade_records)

                logger.info(
                    "Grade %-20s — fetched: %4d | saved: %4d | dropped: %4d",
                    canonical_grade,
                    len(listings),
                    len(grade_records),
                    len(listings) - len(grade_records),
                )

            # Summary
            logger.info(
                "Ingestion complete — fetched: %d | saved: %d | dropped: %d",
                self._stats["total_fetched"],
                self._stats["total_saved"],
                self._stats["total_dropped"],
            )

            if dry_run:
                logger.info("Dry run — skipping file write and MLflow logging.")
                return processed_records

            self.save_processed(processed_records)
            if mlf:
                self._log_mlflow()

        return processed_records
