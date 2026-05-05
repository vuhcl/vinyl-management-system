"""Discogs marketplace ingester (public API)."""
from __future__ import annotations

import copy
import logging
import os
from pathlib import Path
from typing import Any, Optional

import requests

from grader.src.config_io import load_yaml_mapping
from grader.src.project_env import load_project_dotenv

from .constants import (
    DEFAULT_DISCOGS_FORMAT_FILTER,
    _DEFAULT_GENERIC_NOTE_PATTERNS,
    _DEFAULT_ITEM_SPECIFIC_HINTS,
    _DEFAULT_PRESERVATION_KEYWORDS,
    _INVENTORY_WEB_MAX_PAGE_SIZE,
)
from .ingester_fetch import DiscogsIngesterFetchMixin
from .ingester_parse import DiscogsIngesterParseMixin
from .rate_limit import RateLimiter

logger = logging.getLogger(__name__)


class DiscogsIngester(DiscogsIngesterFetchMixin, DiscogsIngesterParseMixin):
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
        data.discogs.vinyl_format_filter_stage — ``fetch`` (default) vs ``post_patch``
        data.discogs.generic_note_filter — boilerplate drop, strip_boilerplate, patterns,
            strip_urls, strip_emoji (normalize before ``strip_boilerplate``)
        paths.raw                   — raw output directory
        paths.processed             — processed output directory
        mlflow.tracking_uri
        mlflow.experiment_name

    Config keys read from grading_guidelines.yaml:
        discogs_condition_map       — raw condition string → canonical grade
        grades[*].hard_signals      — for unverified media detection

    ``offline_parse_only`` (constructor flag): when True, ``DISCOGS_TOKEN`` is not
    required and no HTTP session is created — use only ``parse_listing`` on
    inventory-shaped dicts (e.g. website scrape normalization).
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
        offline_parse_only: bool = False,
    ) -> None:
        if config is not None:
            self.config = copy.deepcopy(config)
        else:
            self.config = load_yaml_mapping(config_path)
        self.guidelines = load_yaml_mapping(guidelines_path)

        load_project_dotenv()

        self.offline_parse_only: bool = bool(offline_parse_only)

        # Auth token — read from environment variable or repo-root .env
        token = os.environ.get("DISCOGS_TOKEN")
        if not self.offline_parse_only:
            if not token:
                raise EnvironmentError(
                    "DISCOGS_TOKEN environment variable is not set. "
                    "Set it before running ingestion."
                )
        else:
            token = token or "offline_parse_only"

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
        self.strip_seller_comment_urls: bool = bool(gnf.get("strip_urls", True))
        self.strip_seller_comment_emoji: bool = bool(gnf.get("strip_emoji", True))
        # fetch: drop non-vinyl during inventory fetch (default). post_patch: keep
        # all formats until after label patches; pipeline runs vinyl_format filter
        # on discogs_processed.jsonl.
        self.vinyl_format_filter_stage: str = str(
            discogs_data.get("vinyl_format_filter_stage", "fetch")
        ).strip().lower()
        # If True, do not fetch from the network when a raw inventory page
        # is missing; only use already-cached JSON pages on disk.
        self.cache_only: bool = bool(cache_only)

        # Build condition map from guidelines
        self.condition_map: dict[str, str] = self.guidelines["discogs_condition_map"]

        # HTTP session with auth headers (not used when offline_parse_only)
        self.session: Optional[requests.Session] = None
        if not self.offline_parse_only:
            self.session = requests.Session()
            self.session.headers.update(
                {
                    "Authorization": f"Discogs token={token}",
                    "User-Agent": "VinylCollectorAI/0.1 +https://github.com/user/vinyl_collector_ai",
                }
            )

        # Rate limiter (only used by network fetches)
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

        # MLflow: configure only inside ``mlflow_pipeline_step_run_ctx`` when a
        # step run is opened — avoid global tracking setup for offline helpers
        # (e.g. ``ingest_sale_history``) and redundant init when step runs are off.

        # Stats counters — reset on each run()
        self._stats: dict = {}
        # If inventory ``format=`` triggers 400, drop it for the rest of the run.
        self._inventory_format_param_rejected: bool = False
        self._inventory_limit_param_rejected: bool = False
        self._logged_inventory_api_page_size_mismatch: bool = False

