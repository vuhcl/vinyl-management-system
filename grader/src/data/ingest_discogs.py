"""
grader/src/data/ingest_discogs.py

Discogs marketplace ingestion module.
Fetches vinyl listings with seller notes and condition grades,
normalizes to canonical grade schema, and saves to unified
JSONL format for downstream preprocessing.

Resume-safe: skips already-fetched raw pages on re-run.
Rate-limited: respects Discogs 60 req/min authenticated limit.
MLflow-tracked: logs data quality metrics after each run.

Usage:
    python -m grader.src.data.ingest_discogs
    python -m grader.src.data.ingest_discogs --dry-run
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import mlflow
import requests
import yaml

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rate Limiter
# ---------------------------------------------------------------------------
class RateLimiter:
    """
    Token bucket rate limiter.
    More robust than time.sleep — accounts for actual elapsed time
    so processing time between calls is not double-counted.
    """

    def __init__(self, calls_per_minute: int) -> None:
        self.min_interval: float = 60.0 / calls_per_minute
        self.last_call: float = 0.0

    def wait(self) -> None:
        elapsed = time.time() - self.last_call
        remaining = self.min_interval - elapsed
        if remaining > 0:
            time.sleep(remaining)
        self.last_call = time.time()


# ---------------------------------------------------------------------------
# DiscogsIngester
# ---------------------------------------------------------------------------
class DiscogsIngester:
    """
    Fetches Discogs marketplace vinyl listings and normalizes
    them to the unified training data schema.

    Config keys read from grader.yaml:
        discogs.token               — personal access token (via env var)
        discogs.base_url            — API base URL
        discogs.rate_limit_per_minute
        discogs.marketplace_endpoint
        paths.raw                   — raw output directory
        paths.processed             — processed output directory
        mlflow.tracking_uri
        mlflow.experiment_name

    Config keys read from grading_guidelines.yaml:
        discogs_condition_map       — raw condition string → canonical grade
        grades[*].hard_signals      — for unverified media detection
    """

    # Discogs condition strings that indicate unverified media.
    # These map to media_verifiable: False in output schema.
    UNVERIFIED_MEDIA_SIGNALS = {
        "untested",
        "unplayed",
        "sold as seen",
        "haven't played",
        "not played",
        "unable to test",
        "no turntable",
    }

    def __init__(self, config_path: str, guidelines_path: str) -> None:
        self.config = self._load_yaml(config_path)
        self.guidelines = self._load_yaml(guidelines_path)

        # Auth token — read from environment variable, never hardcoded
        token = os.environ.get("DISCOGS_TOKEN")
        if not token:
            raise EnvironmentError(
                "DISCOGS_TOKEN environment variable is not set. "
                "Set it before running ingestion."
            )

        self.base_url: str = self.config["discogs"]["base_url"]
        self.endpoint: str = self.config["discogs"]["marketplace_endpoint"]
        self.target_per_grade: int = self.config["data"]["discogs"].get(
            "target_per_grade", 500
        )

        # Build condition map from guidelines
        self.condition_map: dict[str, str] = self.guidelines[
            "discogs_condition_map"
        ]

        # HTTP session with auth headers
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Discogs token={token}",
                "User-Agent": "VinylCollectorAI/0.1 +https://github.com/user/vinyl_collector_ai",
            }
        )

        # Rate limiter
        self.rate_limiter = RateLimiter(
            calls_per_minute=self.config["discogs"]["rate_limit_per_minute"]
        )

        # Paths
        self.raw_dir = Path(self.config["paths"]["raw"]) / "discogs"
        self.processed_path = (
            Path(self.config["paths"]["processed"]) / "discogs_processed.jsonl"
        )
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_path.parent.mkdir(parents=True, exist_ok=True)

        # MLflow
        mlflow.set_tracking_uri(self.config["mlflow"]["tracking_uri"])
        mlflow.set_experiment(self.config["mlflow"]["experiment_name"])

        # Stats counters — reset on each run()
        self._stats: dict = {}

    # -----------------------------------------------------------------------
    # Config loading
    # -----------------------------------------------------------------------
    @staticmethod
    def _load_yaml(path: str) -> dict:
        with open(path, "r") as f:
            return yaml.safe_load(f)

    # -----------------------------------------------------------------------
    # HTTP
    # -----------------------------------------------------------------------
    def _get(self, url: str, params: dict) -> dict:
        """
        Rate-limited GET request with basic retry on transient errors.
        Raises on 4xx (except 429) after retries exhausted.
        """
        max_retries = 3
        for attempt in range(max_retries):
            self.rate_limiter.wait()
            try:
                response = self.session.get(url, params=params, timeout=30)

                if response.status_code == 429:
                    # Discogs rate limit hit — back off and retry
                    retry_after = int(response.headers.get("Retry-After", 60))
                    logger.warning(
                        "Rate limit hit (429). Backing off %ds.", retry_after
                    )
                    time.sleep(retry_after)
                    continue

                response.raise_for_status()
                return response.json()

            except requests.exceptions.Timeout:
                logger.warning(
                    "Request timeout (attempt %d/%d): %s",
                    attempt + 1,
                    max_retries,
                    url,
                )
                if attempt == max_retries - 1:
                    raise

            except requests.exceptions.HTTPError as e:
                logger.error("HTTP error: %s", e)
                raise

        raise RuntimeError(
            f"Failed to fetch {url} after {max_retries} retries."
        )

    # -----------------------------------------------------------------------
    # Raw page fetching with resume logic
    # -----------------------------------------------------------------------
    def _raw_page_path(self, condition_slug: str, page: int) -> Path:
        page_dir = self.raw_dir / condition_slug
        page_dir.mkdir(parents=True, exist_ok=True)
        return page_dir / f"page_{page:03d}.json"

    def _page_already_fetched(self, condition_slug: str, page: int) -> bool:
        return self._raw_page_path(condition_slug, page).exists()

    def _save_raw_page(
        self, condition_slug: str, page: int, data: dict
    ) -> None:
        path = self._raw_page_path(condition_slug, page)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.debug("Saved raw page: %s", path)

    def _load_raw_page(self, condition_slug: str, page: int) -> dict:
        path = self._raw_page_path(condition_slug, page)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def fetch_page(
        self,
        condition: str,
        condition_slug: str,
        page: int,
    ) -> dict:
        """
        Fetch one page of marketplace listings for a given condition.
        Returns raw API response dict. Saves to raw directory.
        Skips fetch and loads from disk if page already exists.
        """
        if self._page_already_fetched(condition_slug, page):
            logger.info(
                "Resuming — page already fetched: %s page %d",
                condition_slug,
                page,
            )
            return self._load_raw_page(condition_slug, page)

        url = f"{self.base_url}{self.endpoint}"
        params = {
            "format": "Vinyl",
            "condition": condition,
            "per_page": 100,
            "page": page,
        }

        logger.info("Fetching %s page %d ...", condition_slug, page)
        data = self._get(url, params)
        self._save_raw_page(condition_slug, page, data)
        return data

    def fetch_all(self) -> dict[str, list[dict]]:
        """
        Fetch listings for all Discogs condition grades up to
        target_per_grade. Skips Generic — it has no marketplace
        search equivalent (set via listing structure, not search param).

        Returns dict mapping canonical grade → list of raw listing dicts.
        """
        # Discogs condition strings to iterate over.
        # Excludes Generic — not a searchable marketplace condition.
        # Excludes Excellent — not a Discogs grade, sourced from eBay JP.
        discogs_conditions = {
            k: v
            for k, v in self.condition_map.items()
            if k != "Generic Sleeve" and v != "Excellent"
        }

        # Invert: canonical grade → list of raw Discogs condition strings
        # (Good maps from both "Good (G)" and "Good Plus (G+)")
        canonical_to_raw: dict[str, list[str]] = {}
        for raw_str, canonical in discogs_conditions.items():
            canonical_to_raw.setdefault(canonical, []).append(raw_str)

        all_listings: dict[str, list[dict]] = {
            grade: [] for grade in canonical_to_raw
        }

        for canonical_grade, raw_strings in canonical_to_raw.items():
            condition_slug = canonical_grade.lower().replace(" ", "_")
            collected = 0

            for raw_condition in raw_strings:
                if collected >= self.target_per_grade:
                    break

                page = 1
                while collected < self.target_per_grade:
                    data = self.fetch_page(raw_condition, condition_slug, page)
                    listings = data.get("listings", [])

                    if not listings:
                        logger.info(
                            "No more listings for %s (page %d).",
                            condition_slug,
                            page,
                        )
                        break

                    all_listings[canonical_grade].extend(listings)
                    collected += len(listings)

                    # Check if we've reached the last page
                    pagination = data.get("pagination", {})
                    if page >= pagination.get("pages", 1):
                        break

                    page += 1

            logger.info(
                "Fetched %d raw listings for grade: %s",
                len(all_listings[canonical_grade]),
                canonical_grade,
            )

        return all_listings

    # -----------------------------------------------------------------------
    # Parsing and normalization
    # -----------------------------------------------------------------------
    def normalize_grade(self, raw_condition: str) -> Optional[str]:
        """
        Map a raw Discogs condition string to a canonical grade.
        Returns None if the string is not in the condition map.
        """
        return self.condition_map.get(raw_condition)

    def _detect_media_verifiable(self, text: str) -> bool:
        """
        Returns False if seller notes contain any unverified media signal.
        Checked case-insensitively.
        """
        text_lower = text.lower()
        return not any(
            signal in text_lower for signal in self.UNVERIFIED_MEDIA_SIGNALS
        )

    def parse_listing(self, listing: dict) -> Optional[dict]:
        """
        Extract unified schema fields from a raw Discogs listing dict.
        Returns None if the listing fails any filter condition.
        The raw listing structure is preserved in raw_sleeve / raw_media
        so labels can be re-derived if the condition map changes.
        """
        # Extract condition fields
        raw_sleeve = listing.get("sleeve_condition", "")
        raw_media = listing.get(
            "condition", ""
        )  # Discogs uses "condition" for media

        # Extract seller notes
        text = listing.get("comments", "") or ""

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
        with open(self.processed_path, "a", encoding="utf-8") as f:
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
                "target_per_grade": self.target_per_grade,
                "format_filter": "Vinyl",
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

        with mlflow.start_run(run_name="ingest_discogs"):
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
            self._log_mlflow()

        return processed_records


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Discogs marketplace ingestion"
    )
    parser.add_argument(
        "--config",
        default="grader/configs/grader.yaml",
        help="Path to grader config YAML",
    )
    parser.add_argument(
        "--guidelines",
        default="grader/configs/grading_guidelines.yaml",
        help="Path to grading guidelines YAML",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and parse without writing output or logging to MLflow",
    )
    args = parser.parse_args()

    ingester = DiscogsIngester(
        config_path=args.config,
        guidelines_path=args.guidelines,
    )
    ingester.run(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
