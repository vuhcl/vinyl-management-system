"""HTTP + inventory paging for :class:`DiscogsIngester`."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import requests

from grader.src.data.vinyl_format import release_format_looks_like_physical_vinyl

from .constants import DEFAULT_DISCOGS_FORMAT_FILTER

logger = logging.getLogger(__name__)


class DiscogsIngesterFetchMixin:
    def _get(self, url: str, params: dict) -> dict:
        """
        Rate-limited GET request with basic retry on transient errors.
        Raises on 4xx (except 429) after retries exhausted.
        """
        if self.session is None:
            raise RuntimeError(
                "DiscogsIngester network calls are disabled (offline_parse_only=True)"
            )
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

        Discogs inventory ``release.format`` is often ``LP, Album``,
        ``12", EP``, ``CD, Vinyl, LP``, etc. When ``data.discogs.format_filter``
        is Vinyl-like, we use ``release_format_looks_like_physical_vinyl`` so
        multi-format box sets still count if any vinyl signal is present, while
        CD/DVD/cassette/digital-only rows are dropped.
        """
        rel = listing.get("release") or {}
        release_format = (rel.get("format") or "").strip()
        release_desc = (rel.get("description") or "").strip()
        blob = f"{release_format} {release_desc}".strip().lower()

        needle = (self.format_filter or DEFAULT_DISCOGS_FORMAT_FILTER).lower().strip()
        if needle in {"vinyl", "record", "records"}:
            return release_format_looks_like_physical_vinyl(
                release_format, release_desc
            )

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
                    if (
                        self.vinyl_format_filter_stage != "post_patch"
                        and not self._listing_matches_format_filter(listing)
                    ):
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
