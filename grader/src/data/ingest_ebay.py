"""
grader/src/data/ingest_ebay.py

eBay JP trusted seller ingestion module.
Fetches vinyl listings from a curated list of trusted Japanese
sellers, extracts grades from seller-specific ItemSpecifics fields,
harmonizes to canonical grade schema, and saves to unified JSONL
format for downstream preprocessing.

Supports two grade formats:
  - clean:      bare grade value (e.g. "VG+", "E-")  — Face Records
  - annotated:  grade + inline defect codes           — ELLA RECORDS
                (e.g. "E (Excellent) S (Stain) cornerbump")

Resume-safe: skips already-fetched raw pages on re-run.
Rate-limited: respects eBay API limits.
MLflow-tracked: logs data quality metrics after each run.

Usage:
    python -m grader.src.data.ingest_ebay
    python -m grader.src.data.ingest_ebay --dry-run
"""

import base64
import json
import logging
import os
import re
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
# Rate Limiter — identical to ingest_discogs.py
# Defined here to keep modules self-contained.
# Will be refactored into shared/utils.py when that module is created.
# ---------------------------------------------------------------------------
class RateLimiter:
    """
    Token bucket rate limiter.
    Accounts for actual elapsed time so processing time between
    calls is not double-counted against the rate limit.
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
# EbayIngester
# ---------------------------------------------------------------------------
class EbayIngester:
    """
    Fetches eBay JP vinyl listings from trusted sellers and normalizes
    them to the unified training data schema.

    Config keys read from grader.yaml:
        ebay.base_url                   — eBay Browse API base URL
        ebay.token_url                  — OAuth token endpoint
        ebay.rate_limit_per_minute
        data.ebay.trusted_sellers       — per-seller field alias map
        data.ebay.min_text_length       — minimum text length (eBay-specific)
        paths.raw                       — raw output directory
        paths.processed                 — processed output directory
        mlflow.tracking_uri
        mlflow.experiment_name

    Config keys read from grading_guidelines.yaml:
        ebay_jp_harmonization           — eBay JP grade → canonical grade
    """

    # Regex to extract the grade token from annotated values.
    # Matches the leading grade code before any space or parenthesis.
    # Examples:
    #   "E (Excellent) S (Stain)" → "E"
    #   "E+ (Excellent Plus)"     → "E+"
    #   "VG+"                     → "VG+"
    #   "M-"                      → "M-"
    #   "S"                       → "S"
    GRADE_TOKEN_RE = re.compile(r"^([A-Za-z][A-Za-z0-9+\-]*)(?:\s|$|\()")

    # Signals indicating unverified media condition
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

        # OAuth credentials — from environment variables, never hardcoded
        self.client_id = os.environ.get("EBAY_CLIENT_ID")
        self.client_secret = os.environ.get("EBAY_CLIENT_SECRET")
        if not self.client_id or not self.client_secret:
            raise EnvironmentError(
                "EBAY_CLIENT_ID and EBAY_CLIENT_SECRET environment variables "
                "must be set before running eBay ingestion."
            )

        self.base_url: str = self.config["ebay"]["base_url"]
        self.token_url: str = self.config["ebay"]["token_url"]

        # Per-seller config — field alias map and grade format
        self.seller_configs: dict = (
            self.config["data"]["ebay"]["trusted_sellers"]
        )

        # Minimum text length — lower than Discogs given sparse JP notes
        self.min_text_length: int = (
            self.config["data"]["ebay"].get("min_text_length", 3)
        )

        # Harmonization table from guidelines
        self.harmonization: dict = self.guidelines["ebay_jp_harmonization"]

        # HTTP session
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    "VinylCollectorAI/0.1 "
                    "+https://github.com/user/vinyl_collector_ai"
                ),
                "X-EBAY-C-MARKETPLACE-ID": "EBAY_US",
                "Content-Type": "application/json",
            }
        )

        # OAuth token state
        self._access_token: Optional[str] = None
        self._token_expiry: float = 0.0

        # Rate limiter
        self.rate_limiter = RateLimiter(
            calls_per_minute=self.config["ebay"]["rate_limit_per_minute"]
        )

        # Paths
        self.raw_dir = Path(self.config["paths"]["raw"]) / "ebay"
        self.processed_path = (
            Path(self.config["paths"]["processed"]) / "ebay_processed.jsonl"
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
    # OAuth — client credentials flow with auto-refresh
    # -----------------------------------------------------------------------
    def _get_access_token(self) -> str:
        """
        Fetch or refresh the eBay OAuth app token.
        Token expires every 2 hours — refreshed automatically when expired.
        Uses client credentials flow (no user context required).
        """
        if self._access_token and time.time() < self._token_expiry - 60:
            return self._access_token

        logger.info("Fetching eBay OAuth token ...")

        # Base64-encode client_id:client_secret per eBay OAuth spec
        credentials = base64.b64encode(
            f"{self.client_id}:{self.client_secret}".encode()
        ).decode()

        response = requests.post(
            self.token_url,
            headers={
                "Authorization": f"Basic {credentials}",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            data={
                "grant_type": "client_credentials",
                "scope": "https://api.ebay.com/oauth/api_scope",
            },
            timeout=30,
        )
        response.raise_for_status()
        token_data = response.json()

        self._access_token = token_data["access_token"]
        # expires_in is in seconds — store absolute expiry timestamp
        self._token_expiry = time.time() + token_data.get("expires_in", 7200)
        self.session.headers.update(
            {"Authorization": f"Bearer {self._access_token}"}
        )

        logger.info("eBay OAuth token acquired.")
        return self._access_token

    # -----------------------------------------------------------------------
    # HTTP
    # -----------------------------------------------------------------------
    def _get(self, url: str, params: dict) -> dict:
        """
        Rate-limited GET with token refresh and retry on transient errors.
        """
        self._get_access_token()

        max_retries = 3
        for attempt in range(max_retries):
            self.rate_limiter.wait()
            try:
                response = self.session.get(url, params=params, timeout=30)

                if response.status_code == 401:
                    # Token may have expired mid-run — force refresh and retry
                    logger.warning("401 Unauthorized — refreshing token.")
                    self._token_expiry = 0.0
                    self._get_access_token()
                    continue

                if response.status_code == 429:
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

        raise RuntimeError(f"Failed to fetch {url} after {max_retries} retries.")

    # -----------------------------------------------------------------------
    # Raw page fetching with resume logic
    # -----------------------------------------------------------------------
    def _raw_page_path(self, seller: str, offset: int) -> Path:
        seller_dir = self.raw_dir / seller
        seller_dir.mkdir(parents=True, exist_ok=True)
        return seller_dir / f"offset_{offset:06d}.json"

    def _page_already_fetched(self, seller: str, offset: int) -> bool:
        return self._raw_page_path(seller, offset).exists()

    def _save_raw_page(self, seller: str, offset: int, data: dict) -> None:
        path = self._raw_page_path(seller, offset)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.debug("Saved raw page: %s", path)

    def _load_raw_page(self, seller: str, offset: int) -> dict:
        path = self._raw_page_path(seller, offset)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def fetch_page(self, seller: str, offset: int) -> dict:
        """
        Fetch one page of listings for a given seller at a given offset.
        Saves raw response to disk. Skips fetch if page already exists.
        """
        if self._page_already_fetched(seller, offset):
            logger.info(
                "Resuming — page already fetched: %s offset %d", seller, offset
            )
            return self._load_raw_page(seller, offset)

        url = f"{self.base_url}/item_summary/search"
        params = {
            "q": "vinyl record",
            "filter": f"sellers:{{{seller}}},itemLocationCountry:JP",
            "limit": 200,
            "offset": offset,
        }

        logger.info("Fetching seller=%s offset=%d ...", seller, offset)
        data = self._get(url, params)
        self._save_raw_page(seller, offset, data)
        return data

    def fetch_all(self) -> dict[str, list[dict]]:
        """
        Fetch all listings for all trusted sellers.
        Paginates using offset until no more results.

        Returns dict mapping seller username → list of raw item dicts.
        """
        all_items: dict[str, list[dict]] = {
            seller: [] for seller in self.seller_configs
        }

        for seller in self.seller_configs:
            offset = 0
            while True:
                data = self.fetch_page(seller, offset)
                items = data.get("itemSummaries", [])

                if not items:
                    logger.info(
                        "No more items for seller %s at offset %d.",
                        seller,
                        offset,
                    )
                    break

                all_items[seller].extend(items)

                total = data.get("total", 0)
                offset += len(items)

                if offset >= total:
                    break

            logger.info(
                "Fetched %d raw items for seller: %s",
                len(all_items[seller]),
                seller,
            )

        return all_items

    # -----------------------------------------------------------------------
    # ItemSpecifics extraction
    # -----------------------------------------------------------------------
    def _extract_item_specifics(self, item: dict) -> dict[str, str]:
        """
        Parse the localizedAspects array into a flat {name: value} dict.
        eBay returns item specifics as a list of {name, value} objects.

        Example input:
            [{"name": "Sleeve Grading", "value": "VG+"},
             {"name": "Record Grading", "value": "E-"}]

        Returns:
            {"Sleeve Grading": "VG+", "Record Grading": "E-"}
        """
        aspects = item.get("localizedAspects", [])
        return {
            aspect["name"]: aspect["value"]
            for aspect in aspects
            if "name" in aspect and "value" in aspect
        }

    def _get_field(
        self,
        specifics: dict[str, str],
        field_name: str,
    ) -> Optional[str]:
        """
        Case-insensitive field lookup in item specifics.
        Handles minor field name variations without requiring exact match.
        """
        # Try exact match first
        if field_name in specifics:
            return specifics[field_name]
        # Fall back to case-insensitive match
        field_lower = field_name.lower()
        for key, value in specifics.items():
            if key.lower() == field_lower:
                return value
        return None

    # -----------------------------------------------------------------------
    # Grade extraction — clean and annotated formats
    # -----------------------------------------------------------------------
    def _extract_grade_token(self, value: str) -> tuple[str, str]:
        """
        Extract the grade token and defect suffix from an item specific value.

        For clean format:   "VG+"               → ("VG+", "")
        For annotated:      "E (Excellent) S (Stain) cornerbump"
                                                → ("E", "S (Stain) cornerbump")

        Returns (grade_token, defect_suffix).
        defect_suffix is the remainder after the grade token and any
        parenthetical expansion, stripped of leading/trailing whitespace.
        """
        value = value.strip()
        match = self.GRADE_TOKEN_RE.match(value)
        if not match:
            return value, ""

        grade_token = match.group(1)

        # Remove the grade token and any immediately following parenthetical
        # expansion e.g. "(Excellent Plus)" — these restate the grade in words
        # and add no new information for the NLP model.
        remainder = value[match.end():].strip()
        paren_expansion = re.match(r"^\([^)]+\)\s*", remainder)
        if paren_expansion:
            remainder = remainder[paren_expansion.end():].strip()

        return grade_token, remainder

    def harmonize_grade(self, raw_grade: str) -> Optional[tuple[str, float]]:
        """
        Map a raw eBay JP grade token to a canonical grade and
        its associated label_confidence from the harmonization table.

        Returns (canonical_grade, label_confidence) or None if unmapped.
        """
        entry = self.harmonization.get(raw_grade)
        if entry is None:
            # Try uppercase — sellers occasionally use lowercase
            entry = self.harmonization.get(raw_grade.upper())
        if entry is None:
            return None
        return entry["canonical"], entry["label_confidence"]

    # -----------------------------------------------------------------------
    # Text construction
    # -----------------------------------------------------------------------
    def _build_text(
        self,
        grade_format: str,
        sleeve_suffix: str,
        media_suffix: str,
        obi_suffix: str,
        seller_notes: str,
    ) -> str:
        """
        Construct the text field for a listing.

        For annotated sellers: concatenate all defect suffixes
        (sleeve, media, OBI) and seller notes into a single string.
        Each non-empty component is separated by a period.

        For clean sellers: seller notes only.

        Rationale: annotated seller defect codes (e.g. "S (Stain)",
        "cornerbump") are meaningful NLP signal and should be included.
        Clean sellers provide no such inline signal so we rely on notes.
        """
        if grade_format == "annotated":
            parts = [
                p.strip()
                for p in [sleeve_suffix, media_suffix, obi_suffix, seller_notes]
                if p and p.strip()
            ]
            return ". ".join(parts)
        else:
            return seller_notes.strip()

    # -----------------------------------------------------------------------
    # Filtering
    # -----------------------------------------------------------------------
    def _get_drop_reason(
        self,
        text: str,
        sleeve_grade: Optional[str],
        media_grade: Optional[str],
    ) -> Optional[str]:
        """
        Returns a drop reason string if the listing should be excluded,
        or None if it passes all filters.
        """
        if sleeve_grade is None:
            return "missing_sleeve_field"
        if media_grade is None:
            return "missing_media_field"
        if self.harmonize_grade(sleeve_grade) is None:
            return "unmapped_sleeve_grade"
        if self.harmonize_grade(media_grade) is None:
            return "unmapped_media_grade"
        if not text or len(text.strip()) < self.min_text_length:
            return "text_too_short"
        return None

    def _detect_media_verifiable(self, text: str) -> bool:
        text_lower = text.lower()
        return not any(
            signal in text_lower for signal in self.UNVERIFIED_MEDIA_SIGNALS
        )

    # -----------------------------------------------------------------------
    # Parsing
    # -----------------------------------------------------------------------
    def parse_item(self, item: dict, seller: str) -> Optional[dict]:
        """
        Extract unified schema fields from a raw eBay item dict.
        Returns None if the item fails any filter condition.

        Handles both clean (Face Records) and annotated (ELLA RECORDS)
        grade formats based on the per-seller config.
        """
        seller_cfg = self.seller_configs[seller]
        grade_format = seller_cfg.get("grade_format", "clean")

        # Extract item specifics
        specifics = self._extract_item_specifics(item)

        # Extract raw field values using seller-specific field names
        raw_sleeve_value = self._get_field(
            specifics, seller_cfg["sleeve_field"]
        )
        raw_media_value = self._get_field(
            specifics, seller_cfg["media_field"]
        )
        raw_obi_value = self._get_field(
            specifics, seller_cfg.get("obi_field", "")
        )

        # Extract grade tokens and defect suffixes
        sleeve_token, sleeve_suffix = (
            self._extract_grade_token(raw_sleeve_value)
            if raw_sleeve_value
            else (None, "")
        )
        media_token, media_suffix = (
            self._extract_grade_token(raw_media_value)
            if raw_media_value
            else (None, "")
        )
        _, obi_suffix = (
            self._extract_grade_token(raw_obi_value)
            if raw_obi_value
            else (None, "")
        )

        # Seller notes — may be absent for eBay JP listings
        seller_notes = item.get("shortDescription", "") or ""

        # Build text field
        text = self._build_text(
            grade_format,
            sleeve_suffix,
            media_suffix,
            obi_suffix,
            seller_notes,
        )

        # Apply filters
        drop_reason = self._get_drop_reason(text, sleeve_token, media_token)
        if drop_reason:
            self._stats["drops"][drop_reason] = (
                self._stats["drops"].get(drop_reason, 0) + 1
            )
            return None

        # Harmonize grades
        sleeve_result = self.harmonize_grade(sleeve_token)
        media_result = self.harmonize_grade(media_token)

        # Both are guaranteed non-None here — drop_reason would have caught them
        sleeve_label, sleeve_confidence = sleeve_result
        media_label, media_confidence = media_result

        # Use the lower of the two confidences as the listing's label_confidence
        # — if either mapping is uncertain, the whole record is less reliable
        label_confidence = min(sleeve_confidence, media_confidence)

        # OBI condition — raw value preserved as metadata string
        obi_condition = raw_obi_value if raw_obi_value else None

        # Extract metadata
        artist = self._get_field(specifics, "Artist") or ""
        title = item.get("title", "")
        year = None  # eBay item summaries don't reliably expose year
        country = self._get_field(specifics, "Country of Origin") or ""

        return {
            "item_id":          str(item.get("itemId", "")),
            "source":           "ebay_jp",
            "text":             text.strip(),
            "sleeve_label":     sleeve_label,
            "media_label":      media_label,
            "label_confidence": label_confidence,
            "media_verifiable": self._detect_media_verifiable(text),
            "obi_condition":    obi_condition,
            "raw_sleeve":       raw_sleeve_value or "",
            "raw_media":        raw_media_value or "",
            "artist":           artist,
            "title":            title,
            "year":             year,
            "country":          country,
        }

    # -----------------------------------------------------------------------
    # Processed output
    # -----------------------------------------------------------------------
    def save_processed(self, records: list[dict]) -> None:
        """
        Append processed records to the unified JSONL output file.
        Appends rather than overwrites — safe to call after Discogs
        ingestion has already written to the same processed directory.
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
                "source":        "ebay_jp",
                "sellers":       list(self.seller_configs.keys()),
                "min_text_len":  self.min_text_length,
            }
        )
        mlflow.log_metrics(
            {
                "total_fetched": self._stats["total_fetched"],
                "total_dropped": self._stats["total_dropped"],
                "total_saved":   self._stats["total_saved"],
            }
        )
        for seller, count in self._stats["per_seller"].items():
            mlflow.log_metric(f"saved_{seller}", count)
        for reason, count in self._stats["drops"].items():
            mlflow.log_metric(f"dropped_{reason}", count)

    # -----------------------------------------------------------------------
    # Orchestration
    # -----------------------------------------------------------------------
    def run(self, dry_run: bool = False) -> list[dict]:
        """
        Full ingestion pipeline:
          1. Fetch all listings for all trusted sellers (with resume logic)
          2. Parse and filter each item
          3. Save processed records to JSONL
          4. Log metrics to MLflow

        Args:
            dry_run: if True, fetch and parse but do not write output
                     or log to MLflow.

        Returns:
            List of processed record dicts.
        """
        self._stats = {
            "total_fetched": 0,
            "total_dropped": 0,
            "total_saved":   0,
            "per_seller":    {},
            "drops":         {},
        }

        with mlflow.start_run(run_name="ingest_ebay"):
            all_items = self.fetch_all()
            processed_records: list[dict] = []

            for seller, items in all_items.items():
                seller_records: list[dict] = []

                for item in items:
                    self._stats["total_fetched"] += 1
                    record = self.parse_item(item, seller)

                    if record is None:
                        self._stats["total_dropped"] += 1
                        continue

                    seller_records.append(record)

                self._stats["per_seller"][seller] = len(seller_records)
                self._stats["total_saved"] += len(seller_records)
                processed_records.extend(seller_records)

                logger.info(
                    "Seller %-20s — fetched: %4d | saved: %4d | dropped: %4d",
                    seller,
                    len(items),
                    len(seller_records),
                    len(items) - len(seller_records),
                )

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

    parser = argparse.ArgumentParser(description="eBay JP trusted seller ingestion")
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

    ingester = EbayIngester(
        config_path=args.config,
        guidelines_path=args.guidelines,
    )
    ingester.run(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
