"""
grader/src/data/ingest_discogs.py

Discogs marketplace ingestion module.
Fetches vinyl listings with seller notes and condition grades,
normalizes to canonical grade schema, and saves to unified
JSONL format for downstream preprocessing.

Uses GET /users/{username}/inventory (official API). The historical
``/marketplace/search`` URL is not supported by Discogs and returns 404.

Resume-safe: skips already-fetched raw pages on re-run.
Rate-limited: respects Discogs 60 req/min authenticated limit.
MLflow-tracked: logs data quality metrics after each run.

Usage:
    python -m grader.src.data.ingest_discogs
    python -m grader.src.data.ingest_discogs --dry-run
"""

import copy
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Optional

import mlflow
import requests
import yaml

from grader.src.mlflow_tracking import (
    configure_mlflow_from_config,
    mlflow_enabled,
    mlflow_start_run_ctx,
)
from grader.src.project_env import load_project_dotenv

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# Defaults for generic seller-note filter (overridable via grader.yaml).
_DEFAULT_GENERIC_NOTE_PATTERNS: tuple[str, ...] = (
    "check out our other",
    "feel free to message",
    "feel free to ask",
    "message us with any",
    "ships within",
    "ship within",
    "will ship within",
    "buy 3 or more items",
    "discount",
    "all products are shipped",
    "usually ships",
    "working days",
    "please refer to discogs grading",
    "refer to discogs grading",
    "see discogs grading",
    "graded per discogs",
    "all items are graded",
    "all records are visually graded",
    "visually graded under",
    "see pictures for condition",
    "see photos for condition",
    "see seller terms",
    "visit our store",
    "our other listings",
    "combined shipping",
    "discount on shipping",
    "us shipping",
    "buy 3 or more items",
    "buy 6 or more items",
    "get a 10% discount",
    "get a 20% discount",
    "discount price will be refund after payment",
    "all products are shipped",
    "shipped from our us hub",
    "from our us hub",
    # Common shop header/footer (profile / policies), e.g.:
    # "Secondhand item in our store. Please visit our profile … policies. … dinged."
    "secondhand item in our store",
    "brand new item in our store",
    "used item in our store",
    "please visit our profile",
    "visit our profile",
    "general information including",
    "shipping prices and policies",
    "shipping policies",
    "for items over $20",
    "for items over 20",
    "contact us for details and pictures",
    "details and pictures",
    "mentioned in the item description only if included",
    # Promo / sale banners (often bracketed on Discogs)
    "half off",
    "marked for deletion",
    "new low price",
    # Seller trust / policy spam
    "from a reputable seller",
    "reputable seller",
    "full refunds available",
    "refunds available if unhappy",
    "buy with confidence",
    # Bracketed provenance lines, e.g. "[From Pittsburgh George's Collection]"
    "from pittsburgh george",
    "collection]",
    # Price / shipping promo fragments in seller headers
    "$6 /",
)
_DEFAULT_ITEM_SPECIFIC_HINTS: tuple[str, ...] = (
    "scratch",
    "scuff",
    "hairline",
    "warp",
    "pop",
    "crackle",
    "skip",
    "seam split",
    "seam-split",
    "ring wear",
    "shelf wear",
    "crease",
    "stain",
    "writing",
    "tear",
    "corner",
    "bump",
    "ding",
    "plays well",
    "plays great",
    "plays perfectly",
    "tested",
    "cleaned",
    "static",
    "noise",
    "mark",
    "wear",
    "damage",
)
# Default marketplace format when ``data.discogs.format_filter`` is missing or blank.
DEFAULT_DISCOGS_FORMAT_FILTER: str = "Vinyl"

# Seller profile on the website supports up to 250 items per page, e.g.
# ``/seller/{user}/profile?limit=250&format=Vinyl``. We mirror that on
# ``api.discogs.com/users/{user}/inventory`` with both ``per_page`` and
# ``limit``. The JSON API has historically still responded with
# ``pagination.per_page: 100`` — check the response and logs.
_INVENTORY_WEB_MAX_PAGE_SIZE: int = 250

_DEFAULT_PRESERVATION_KEYWORDS: tuple[str, ...] = (
    "sealed",
    "shrink",
    "unopened",
    "brand new",
    "factory sealed",
    "never opened",
    "still in shrink",
    "n.o.s",
    "nos ",
    "new copy",
    "dead stock",
    "deadstock",
)


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
        data.discogs.inventory_sellers — Discogs usernames to page (public inventory)
        data.discogs.format_filter  — e.g. Vinyl (matched in release text)
        data.discogs.target_per_grade
        data.discogs.max_public_inventory_pages — cap for others' inventory (Discogs: 100)
        data.discogs.inventory_per_page — page size up to 250 (website seller profile)
        data.discogs.inventory_send_limit_param — also send limit= (website mirror)
        data.discogs.inventory_format_api_param — send format=… on inventory GET
        data.discogs.generic_note_filter — boilerplate drop, strip_boilerplate, patterns
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

    def __init__(
        self,
        config_path: str,
        guidelines_path: str,
        *,
        target_per_grade: Optional[int] = None,
        format_filter: Optional[str] = None,
        cache_only: bool = False,
        config: Optional[dict[str, Any]] = None,
    ) -> None:
        if config is not None:
            self.config = copy.deepcopy(config)
        else:
            self.config = self._load_yaml(config_path)
        self.guidelines = self._load_yaml(guidelines_path)

        load_project_dotenv()

        # Auth token — read from environment variable or repo-root .env
        token = os.environ.get("DISCOGS_TOKEN")
        if not token:
            raise EnvironmentError(
                "DISCOGS_TOKEN environment variable is not set. "
                "Set it before running ingestion."
            )

        self.base_url: str = self.config["discogs"]["base_url"]
        discogs_data = self.config["data"]["discogs"]
        self.inventory_sellers: list[str] = discogs_data.get("inventory_sellers", [])
        _cfg_format = discogs_data.get("format_filter")
        _fmt_src = format_filter if format_filter is not None else _cfg_format
        if isinstance(_fmt_src, str) and _fmt_src.strip():
            self.format_filter: str = _fmt_src.strip()
        else:
            self.format_filter = DEFAULT_DISCOGS_FORMAT_FILTER
        cfg_target = discogs_data.get("target_per_grade", 500)
        self.target_per_grade: int = (
            target_per_grade if target_per_grade is not None else cfg_target
        )
        # Discogs returns 403 beyond page 100 for *other users'* inventories
        # ("Pagination above 100 disabled for inventories besides your own").
        self.max_public_inventory_pages: int = int(
            discogs_data.get("max_public_inventory_pages", 100)
        )
        _pp = int(discogs_data.get("inventory_per_page", _INVENTORY_WEB_MAX_PAGE_SIZE))
        self.inventory_per_page: int = max(1, min(_pp, _INVENTORY_WEB_MAX_PAGE_SIZE))
        if _pp > _INVENTORY_WEB_MAX_PAGE_SIZE:
            logger.warning(
                "data.discogs.inventory_per_page=%d exceeds website max (%d); using %d.",
                _pp,
                _INVENTORY_WEB_MAX_PAGE_SIZE,
                _INVENTORY_WEB_MAX_PAGE_SIZE,
            )
        # Mirror seller profile query string ``limit=250``.
        self.inventory_send_limit_param: bool = bool(
            discogs_data.get("inventory_send_limit_param", True)
        )
        # Mirror seller profile default ordering (sort=listed,desc on the site).
        self.inventory_sort: Optional[str] = discogs_data.get("inventory_sort")
        self.inventory_sort_order: Optional[str] = discogs_data.get(
            "inventory_sort_order"
        )
        # Undocumented for inventory — many deployments accept it like marketplace search.
        self.inventory_format_api_param: bool = bool(
            discogs_data.get("inventory_format_api_param", True)
        )

        gnf = discogs_data.get("generic_note_filter") or {}
        self.generic_note_filter_enabled: bool = bool(gnf.get("enabled", True))
        _p = gnf.get("patterns")
        self.generic_note_patterns: list[str] = list(
            _DEFAULT_GENERIC_NOTE_PATTERNS if _p is None else _p
        )
        _h = gnf.get("item_specific_hints")
        self.item_specific_hints: list[str] = list(
            _DEFAULT_ITEM_SPECIFIC_HINTS if _h is None else _h
        )
        _k = gnf.get("preservation_keywords")
        self.preservation_keywords: list[str] = list(
            _DEFAULT_PRESERVATION_KEYWORDS if _k is None else _k
        )
        self.strip_boilerplate_enabled: bool = bool(gnf.get("strip_boilerplate", True))
        # If True, do not fetch from the network when a raw inventory page
        # is missing; only use already-cached JSON pages on disk.
        self.cache_only: bool = bool(cache_only)

        # Build condition map from guidelines
        self.condition_map: dict[str, str] = self.guidelines["discogs_condition_map"]

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

        if mlflow_enabled(self.config):
            configure_mlflow_from_config(self.config)

        # Stats counters — reset on each run()
        self._stats: dict = {}
        # If inventory ``format=`` triggers 400, drop it for the rest of the run.
        self._inventory_format_param_rejected: bool = False
        self._inventory_limit_param_rejected: bool = False
        self._logged_inventory_api_page_size_mismatch: bool = False

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

        raise RuntimeError(f"Failed to fetch {url} after {max_retries} retries.")

    # -----------------------------------------------------------------------
    # Raw page fetching with resume logic (seller inventory)
    # -----------------------------------------------------------------------
    def _inventory_dir(self, seller_slug: str) -> Path:
        # Separate cache per per_page so 100 vs 250 responses are not mixed.
        d = self.raw_dir / "inventory" / seller_slug / f"per_{self.inventory_per_page}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _raw_inventory_page_path(self, seller_slug: str, page: int) -> Path:
        return self._inventory_dir(seller_slug) / f"page_{page:03d}.json"

    def _inventory_page_exists(self, seller_slug: str, page: int) -> bool:
        return self._raw_inventory_page_path(seller_slug, page).exists()

    def _save_inventory_page(self, seller_slug: str, page: int, data: dict) -> None:
        path = self._raw_inventory_page_path(seller_slug, page)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.debug("Saved raw inventory page: %s", path)

    def _load_inventory_page(self, seller_slug: str, page: int) -> dict:
        path = self._raw_inventory_page_path(seller_slug, page)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _seller_slug(username: str) -> str:
        return username.replace("/", "_").lower()

    def _listing_matches_format_filter(self, listing: dict) -> bool:
        """
        Keep rows whose release metadata looks like vinyl (configurable).

        Note: Discogs inventory `release.format` is typically values like
        "LP, Album", "12\", EP, ..." or "CD, Album" (not literally the word
        "Vinyl"). When `data.discogs.format_filter` is "Vinyl", we therefore
        treat LP/EP/12\"/etc as vinyl-like and exclude CD/cassette/DVD-like
        formats.
        """
        rel = listing.get("release") or {}
        release_format = (rel.get("format") or "").strip()
        release_desc = (rel.get("description") or "").strip()
        blob = f"{release_format} {release_desc}".strip().lower()

        needle = (self.format_filter or DEFAULT_DISCOGS_FORMAT_FILTER).lower().strip()
        if needle in {"vinyl", "record", "records"}:
            rf = release_format.lower()

            # Exclude non-vinyl media types commonly present in `release.format`.
            non_vinyl_tokens = (
                "cd",
                "dvd",
                "sacd",
                "blu-ray",
                "cassette",
                "tape",
                "digital",
                "mp3",
                "file",
                "download",
                "stream",
            )
            if any(tok in rf for tok in non_vinyl_tokens):
                return False

            # Allow common vinyl-related Discogs format shorthands.
            vinyl_tokens = (
                "lp",
                "2xlp",
                "3xlp",
                '12"',
                '12"',
                '7"',
                '7"',
                "ep",
                "single",
                "comp",
                "album",
            )
            if any(tok in rf for tok in vinyl_tokens):
                return True

            # Fallback: if either format or description literally contains the word.
            return "vinyl" in blob

        # Generic substring match for other needles.
        return needle in blob

    def fetch_inventory_page(self, seller: str, page: int) -> dict:
        """
        Fetch one page of a seller's public inventory (``For Sale``).
        Same listing shape as legacy marketplace responses — compatible
        with ``parse_listing``.
        """
        slug = self._seller_slug(seller)

        if page > self.max_public_inventory_pages:
            logger.debug(
                "Skipping inventory page %d for %r — above public cap (%d).",
                page,
                seller,
                self.max_public_inventory_pages,
            )
            return {
                "listings": [],
                "pagination": {
                    "pages": self.max_public_inventory_pages,
                    "page": page,
                    "per_page": self.inventory_per_page,
                },
            }

        if self._inventory_page_exists(slug, page):
            logger.info(
                "Resuming — inventory page already fetched: %s page %d",
                seller,
                page,
            )
            return self._load_inventory_page(slug, page)

        if self.cache_only:
            logger.info(
                "Cache-only mode — missing inventory page (seller=%r page=%d). Skipping without network.",
                seller,
                page,
            )
            return {
                "listings": [],
                "pagination": {
                    "pages": self.max_public_inventory_pages,
                    "page": page,
                    "per_page": self.inventory_per_page,
                },
            }

        url = f"{self.base_url}/users/{seller}/inventory"
        params: dict[str, str | int] = {
            "status": "For Sale",
            "per_page": self.inventory_per_page,
            "page": page,
        }
        if self.inventory_send_limit_param and not self._inventory_limit_param_rejected:
            # Same name as seller profile: ?limit=250
            params["limit"] = self.inventory_per_page
        if (
            self.inventory_format_api_param
            and (self.format_filter or "").strip()
            and not self._inventory_format_param_rejected
        ):
            params["format"] = self.format_filter.strip()
        if self.inventory_sort:
            params["sort"] = self.inventory_sort
        if self.inventory_sort_order:
            params["sort_order"] = self.inventory_sort_order

        logger.info("Fetching inventory %s page %d ...", seller, page)
        attempt_params: dict[str, str | int] = dict(params)
        while True:
            try:
                data = self._get(url, attempt_params)
                break
            except requests.exceptions.HTTPError as e:
                code = getattr(e.response, "status_code", None)
                if (
                    code == 400
                    and "format" in attempt_params
                    and not self._inventory_format_param_rejected
                ):
                    logger.warning(
                        "Inventory API rejected format=%r (400) — omitting "
                        "format= for the rest of this run. Client-side vinyl "
                        "filter still applies.",
                        attempt_params.get("format"),
                    )
                    self._inventory_format_param_rejected = True
                    attempt_params.pop("format", None)
                    continue
                if (
                    code == 400
                    and "limit" in attempt_params
                    and not self._inventory_limit_param_rejected
                ):
                    logger.warning(
                        "Inventory API rejected limit=%r (400) — omitting "
                        "limit= for the rest of this run.",
                        attempt_params.get("limit"),
                    )
                    self._inventory_limit_param_rejected = True
                    attempt_params.pop("limit", None)
                    continue
                if code == 404:
                    logger.warning(
                        "No public inventory for user %r (404) — skipping.",
                        seller,
                    )
                    return {
                        "listings": [],
                        "pagination": {
                            "pages": 0,
                            "page": page,
                            "per_page": self.inventory_per_page,
                        },
                    }
                if code == 403:
                    detail = ""
                    try:
                        detail = (e.response.text or "")[:200]
                    except Exception:
                        pass
                    logger.warning(
                        "Inventory 403 for %r page %d — Discogs limits public "
                        "inventory to ~%d pages per seller. Stopping this seller. %s",
                        seller,
                        page,
                        self.max_public_inventory_pages,
                        detail.strip() or "(no body)",
                    )
                    return {
                        "listings": [],
                        "pagination": {
                            "pages": page - 1,
                            "page": page,
                            "per_page": self.inventory_per_page,
                        },
                    }
                raise

        self._maybe_warn_inventory_api_page_cap(data)
        self._save_inventory_page(slug, page, data)
        return data

    def _maybe_warn_inventory_api_page_cap(self, data: dict) -> None:
        """
        Log once if the JSON API reports a smaller page size than we requested
        (common: request 250, response pagination.per_page is 100).
        """
        if self._logged_inventory_api_page_size_mismatch:
            return
        pag = data.get("pagination") or {}
        raw_pp = pag.get("per_page")
        if raw_pp is None:
            return
        try:
            api_pp = int(raw_pp)
        except (TypeError, ValueError):
            return
        if api_pp >= self.inventory_per_page:
            return
        n_list = len(data.get("listings") or [])
        logger.warning(
            "Discogs API returned pagination.per_page=%d (%d listings) but we "
            "requested %d (per_page + limit, mirroring the seller profile UI). "
            "The website supports limit=250; api.discogs.com may still cap "
            "inventory at 100.",
            api_pp,
            n_list,
            self.inventory_per_page,
        )
        self._logged_inventory_api_page_size_mismatch = True

    def _target_canonical_media_grades(self) -> list[str]:
        """Canonical grades we try to fill (media), aligned with old search."""
        ordered: list[str] = []
        seen: set[str] = set()
        for raw, canonical in self.condition_map.items():
            # Generic is sleeve-only; Excellent is eBay-only in this setup.
            if raw == "Generic Sleeve" or canonical in {"Excellent", "Generic"}:
                continue
            if canonical not in seen:
                seen.add(canonical)
                ordered.append(canonical)
        return ordered

    def fetch_all(self) -> dict[str, list[dict]]:
        """
        Page configured sellers' inventories until each target canonical
        **media** grade has up to ``target_per_grade`` vinyl listings.

        Returns dict mapping canonical grade → list of raw listing dicts.
        """
        if not self.inventory_sellers:
            raise ValueError(
                "data.discogs.inventory_sellers must list at least one "
                "Discogs username. The official API does not expose "
                "/marketplace/search (404); ingestion uses "
                "/users/{username}/inventory instead. "
                "Add sellers under data.discogs.inventory_sellers in "
                "grader.yaml."
            )

        target_grades = self._target_canonical_media_grades()
        buckets: dict[str, list[dict]] = {g: [] for g in target_grades}

        def bucket_complete() -> bool:
            return all(len(buckets[g]) >= self.target_per_grade for g in target_grades)

        for seller in self.inventory_sellers:
            if bucket_complete():
                break
            page = 1
            while not bucket_complete():
                data = self.fetch_inventory_page(seller, page)
                listings = data.get("listings", [])

                if not listings:
                    logger.info("No more inventory for %s (page %d).", seller, page)
                    break

                for listing in listings:
                    if not self._listing_matches_format_filter(listing):
                        continue
                    media_raw = listing.get("condition") or ""
                    canonical = self.normalize_grade(media_raw)
                    if canonical is None or canonical not in buckets:
                        continue
                    if len(buckets[canonical]) >= self.target_per_grade:
                        continue
                    buckets[canonical].append(listing)

                pagination = data.get("pagination", {})
                total_pages = int(pagination.get("pages") or 1)
                effective_last = min(total_pages, self.max_public_inventory_pages)
                if page >= effective_last:
                    break
                page += 1

        for grade, lst in buckets.items():
            logger.info(
                "Fetched %d raw listings for grade: %s",
                len(lst),
                grade,
            )

        return buckets

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
        # Extract condition fields
        raw_sleeve = listing.get("sleeve_condition", "")
        raw_media = listing.get("condition", "")  # Discogs uses "condition" for media

        # Extract seller notes; strip shop boilerplate before filters & output
        raw_comments = listing.get("comments", "") or ""
        text = self.strip_boilerplate_from_notes(raw_comments)

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

        with mlflow_start_run_ctx(self.config, "ingest_discogs"):
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
            if mlflow_enabled(self.config):
                self._log_mlflow()

        return processed_records


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Discogs marketplace ingestion")
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
    parser.add_argument(
        "--target-per-grade",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Override grader.yaml data.discogs.target_per_grade "
            "(max raw listings per canonical grade; lower = faster smoke ingest)"
        ),
    )
    parser.add_argument(
        "--format",
        default=None,
        metavar="NAME",
        help=(
            "Override data.discogs.format_filter (API format= + listing match). "
            f"Omit to use YAML or built-in default ({DEFAULT_DISCOGS_FORMAT_FILTER!r})."
        ),
    )
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help=(
            "Only use cached raw inventory pages already on disk. "
            "If a page is missing, skip it instead of calling Discogs."
        ),
    )
    args = parser.parse_args()

    ingester = DiscogsIngester(
        config_path=args.config,
        guidelines_path=args.guidelines,
        target_per_grade=args.target_per_grade,
        format_filter=args.format,
        cache_only=args.cache_only,
    )
    ingester.run(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
