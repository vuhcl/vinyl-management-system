"""
Discogs ↔ AOTY canonical album matching utilities.

Goal:
  - Build a mapping `discogs_master_id -> aoty_album_id` using
    AOTY (artist + album_title + year) and Discogs database search results.
  - Convert Discogs `release_id`s (from collection/wantlist) into the
    canonical AOTY `album_id` values used by the recommender.

Speed / API limits:
  - Optional on-disk cache under `cache_dir` (release→master, search JSON).
  - Minimum spacing between HTTP calls (`min_request_interval_s`).
  - Reads `X-Discogs-Ratelimit-Remaining` and backs off when low.
  - Retries on HTTP 429 using `Retry-After` when present.
  - Retries on HTTP 500/502/503/504 with exponential backoff (transient gateway errors).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import difflib
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import requests


DISCOGS_BASE_URL = "https://api.discogs.com"
_RELEASE_CACHE_NAME = "release_master.json"
_SEARCH_CACHE_NAME = "search_database.json"
_MASTER_CACHE_NAME = "master_detail.json"


def _normalize_text(s: str | None) -> str:
    if not s:
        return ""
    out = s.lower().strip()
    out = "".join(ch if ch.isalnum() else " " for ch in out)
    out = " ".join(out.split())
    return out


def _title_similarity(a: str, b: str) -> float:
    na = _normalize_text(a)
    nb = _normalize_text(b)
    if not na or not nb:
        return 0.0
    return difflib.SequenceMatcher(None, na, nb).ratio()


def _extract_year(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        year = int(value)
        return year if 1800 <= year <= 2100 else None
    if isinstance(value, str):
        s = value.strip()
        if s.isdigit() and len(s) == 4:
            return int(s)
    return None


def _atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    tmp.replace(path)


def _load_json_dict(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return raw if isinstance(raw, dict) else {}


def _search_cache_key(params: dict[str, Any]) -> str:
    blob = json.dumps(params, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(blob).hexdigest()


@dataclass(frozen=True)
class DiscogsMatchConfig:
    """Tuning for master search + HTTP behavior."""

    per_page: int = 20
    max_results: int = 20
    min_fuzzy_similarity: float = 0.35
    # Discogs: ~60 req/min for authenticated users; stay under on average.
    min_request_interval_s: float = 1.05
    # If header says fewer than this many requests left, sleep extra.
    rate_limit_remaining_sleep_below: int = 5
    rate_limit_extra_sleep_s: float = 2.0
    max_429_retries: int = 1
    # Transient errors from Discogs (502 Bad Gateway is common during load).
    max_transient_retries: int = 5
    transient_http_status_codes: tuple[int, ...] = (500, 502, 503, 504)
    transient_base_sleep_s: float = 2.0
    transient_max_sleep_s: float = 60.0


@dataclass
class DiscogsHttpHelper:
    """
    Shared session + throttle + optional disk cache for Discogs GETs used in
    matching. Pass one instance through build + map in a single ingest run,
    then call save_disk().
    """

    session: requests.Session
    token: str
    cfg: DiscogsMatchConfig
    cache_dir: Path | None = None
    _last_request_mono: float = field(default=0.0, repr=False)
    _release_master: dict[str, str | None] = field(
        default_factory=dict, repr=False
    )
    _search_payloads: dict[str, dict[str, Any]] = field(
        default_factory=dict, repr=False
    )
    _master_payloads: dict[str, dict[str, Any]] = field(
        default_factory=dict, repr=False
    )
    _dirty: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        if self.cache_dir is not None:
            self.cache_dir = Path(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._load_disk()

    def _release_path(self) -> Path:
        assert self.cache_dir is not None
        return self.cache_dir / _RELEASE_CACHE_NAME

    def _search_path(self) -> Path:
        assert self.cache_dir is not None
        return self.cache_dir / _SEARCH_CACHE_NAME

    def _master_path(self) -> Path:
        assert self.cache_dir is not None
        return self.cache_dir / _MASTER_CACHE_NAME

    def _load_disk(self) -> None:
        if self.cache_dir is None:
            return
        rel = _load_json_dict(self._release_path())
        for k, v in rel.items():
            self._release_master[str(k)] = None if v is None else str(v)
        sea = _load_json_dict(self._search_path())
        for k, v in sea.items():
            if isinstance(v, dict):
                self._search_payloads[str(k)] = v
        mas = _load_json_dict(self._master_path())
        for k, v in mas.items():
            if isinstance(v, dict):
                self._master_payloads[str(k)] = v

    def save_disk(self) -> None:
        if self.cache_dir is None or not self._dirty:
            return
        _atomic_write_json(self._release_path(), self._release_master)
        _atomic_write_json(self._search_path(), self._search_payloads)
        _atomic_write_json(self._master_path(), self._master_payloads)
        self._dirty = False

    def _throttle(self) -> None:
        if self.cfg.min_request_interval_s <= 0:
            return
        now = time.monotonic()
        wait = self.cfg.min_request_interval_s - (
            now - self._last_request_mono
        )
        if wait > 0:
            time.sleep(wait)

    def _post_request_headers(self, resp: requests.Response) -> None:
        rem = resp.headers.get("X-Discogs-Ratelimit-Remaining")
        if rem is None:
            return
        try:
            n = int(rem)
        except ValueError:
            return
        if n < self.cfg.rate_limit_remaining_sleep_below:
            time.sleep(self.cfg.rate_limit_extra_sleep_s)

    def _authorized_get(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        timeout_s: float = 30.0,
    ) -> requests.Response:
        self._throttle()
        self.session.headers["Authorization"] = f"Discogs token={self.token}"
        attempt_429 = 0
        attempt_transient = 0
        while True:
            resp = self.session.get(
                url, params=params or {}, timeout=timeout_s
            )
            self._last_request_mono = time.monotonic()
            if resp.status_code == 429 and attempt_429 < self.cfg.max_429_retries:
                ra = resp.headers.get("Retry-After", "60")
                try:
                    delay = float(ra)
                except ValueError:
                    delay = 60.0
                time.sleep(delay)
                attempt_429 += 1
                continue
            if (
                resp.status_code in self.cfg.transient_http_status_codes
                and attempt_transient < self.cfg.max_transient_retries
            ):
                delay = min(
                    self.cfg.transient_max_sleep_s,
                    self.cfg.transient_base_sleep_s
                    * (2**attempt_transient),
                )
                time.sleep(delay)
                attempt_transient += 1
                continue
            self._post_request_headers(resp)
            return resp

    def get_database_search(self, params: dict[str, Any]) -> dict[str, Any]:
        key = _search_cache_key(params)
        if key in self._search_payloads:
            return self._search_payloads[key]

        url = f"{DISCOGS_BASE_URL}/database/search"
        resp = self._authorized_get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, dict):
            data = {}
        self._search_payloads[key] = data
        self._dirty = True
        return data

    def get_release_payload(self, release_id: str) -> dict[str, Any]:
        rid = str(release_id)
        if rid in self._release_master:
            mid = self._release_master[rid]
            if mid is None:
                return {"master_id": None}
            try:
                return {"master_id": int(mid)}
            except ValueError:
                return {"master_id": mid}

        url = f"{DISCOGS_BASE_URL}/releases/{rid}"
        resp = self._authorized_get(url)
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, dict):
            data = {}
        mid_raw = data.get("master_id")
        if mid_raw is None:
            self._release_master[rid] = None
        else:
            self._release_master[rid] = str(mid_raw)
        self._dirty = True
        return data

    def get_master_payload(self, master_id: str) -> dict[str, Any]:
        """GET ``/masters/{id}`` (cached in memory + ``master_detail.json``)."""
        mid = str(master_id)
        if mid in self._master_payloads:
            return self._master_payloads[mid]

        url = f"{DISCOGS_BASE_URL}/masters/{mid}"
        resp = self._authorized_get(url)
        resp.raise_for_status()
        raw = resp.json()
        data = raw if isinstance(raw, dict) else {}
        self._master_payloads[mid] = data
        self._dirty = True
        return data


def parse_discogs_master_artist_title_year(
    data: dict[str, Any],
) -> tuple[str, str, int | None]:
    """Best-effort artist, album title, year from Discogs ``/masters/{id}`` JSON."""
    artists = data.get("artists") or []
    artist = ""
    if isinstance(artists, list) and artists:
        a0 = artists[0]
        if isinstance(a0, dict):
            artist = str(a0.get("name") or "").strip()
    title = str(data.get("title") or "").strip()
    year_raw = data.get("year")
    year: int | None
    try:
        year = int(year_raw) if year_raw is not None else None
    except (TypeError, ValueError):
        year = None
    if year is not None and not (1800 <= year <= 2100):
        year = None
    return artist, title, year


def save_master_to_aoty_json(path: Path, mapping: dict[str, str]) -> None:
    """Write ``{discogs_master_id: aoty_album_id}`` JSON."""
    _atomic_write_json(path, dict(sorted(mapping.items())))


def load_master_to_aoty_json(path: Path) -> dict[str, str]:
    raw = _load_json_dict(path)
    out: dict[str, str] = {}
    for k, v in raw.items():
        if isinstance(k, str) and v is not None:
            out[k] = str(v)
    return out


def _search_discogs_masters(
    *,
    http: DiscogsHttpHelper,
    artist: str,
    album_title: str,
    cfg: DiscogsMatchConfig,
) -> list[dict[str, Any]]:
    params = {
        "type": "master",
        "artist": artist,
        "release_title": album_title,
        "per_page": cfg.per_page,
        "page": 1,
    }
    data = http.get_database_search(params)
    results = data.get("results") or []
    out: list[dict[str, Any]] = []
    for idx, r in enumerate(results):
        rid = r.get("id")
        title = r.get("title") or r.get("master_title") or ""
        r_year = _extract_year(r.get("year") or r.get("date"))
        if rid is None:
            continue
        out.append(
            {
                "master_id": str(rid),
                "title": str(title),
                "year": r_year,
                "relevance_rank": idx,
            }
        )
        if len(out) >= cfg.max_results:
            break
    return out


def build_discogs_master_to_aoty_album_id_map(
    aoty_albums: pd.DataFrame,
    *,
    discogs_token: str | None = None,
    cfg: DiscogsMatchConfig = DiscogsMatchConfig(),
    session: requests.Session | None = None,
    cache_dir: Path | None = None,
    http: DiscogsHttpHelper | None = None,
    stats_out: dict[str, int] | None = None,
) -> dict[str, str]:
    """
    Return mapping: `discogs_master_id -> aoty_album_id`.

    Pass a shared `http` from ingest to avoid reloading cache between build
    and map steps. Otherwise provide `discogs_token` (and optional
    `cache_dir`).
    """
    required_cols = {"album_id", "artist", "album_title", "year"}
    missing = required_cols - set(aoty_albums.columns)
    if missing:
        raise ValueError(
            f"AOTY albums missing required columns: {sorted(missing)}"
        )

    own_http = http is None
    if http is None:
        if not discogs_token:
            raise ValueError("discogs_token required unless http is provided")
        http = DiscogsHttpHelper(
            session or requests.Session(),
            discogs_token,
            cfg,
            cache_dir,
        )

    search_memo: dict[tuple[str, str, int | None], list[dict[str, Any]]] = {}
    master_to_aoty: dict[str, str] = {}
    master_to_score: dict[str, tuple[int, float, int]] = {}

    if stats_out is not None:
        stats_out.clear()
        stats_out.update(
            {
                "aoty_album_rows_input": int(len(aoty_albums)),
                "aoty_rows_skipped_missing_artist_or_title": 0,
                "aoty_rows_no_discogs_search_results": 0,
                "aoty_rows_no_discogs_match_after_fuzzy": 0,
                "discogs_master_to_aoty_entries": 0,
            }
        )

    try:
        for row in aoty_albums.itertuples(index=False):
            aoty_album_id = str(getattr(row, "album_id"))
            artist = str(getattr(row, "artist") or "")
            album_title = str(getattr(row, "album_title") or "")
            aoty_year = _extract_year(getattr(row, "year"))

            if not artist or not album_title:
                if stats_out is not None:
                    stats_out["aoty_rows_skipped_missing_artist_or_title"] += 1
                continue

            cache_key = (artist, album_title, aoty_year)
            if cache_key not in search_memo:
                search_memo[cache_key] = _search_discogs_masters(
                    http=http,
                    artist=artist,
                    album_title=album_title,
                    cfg=cfg,
                )
            candidates = search_memo[cache_key]

            if not candidates:
                if stats_out is not None:
                    stats_out["aoty_rows_no_discogs_search_results"] += 1
                continue

            best: dict[str, Any] | None = None
            best_key: tuple[int, float, int] | None = None

            for cand in candidates:
                cand_master_id = str(cand["master_id"])
                cand_title = str(cand.get("title") or "")
                cand_year = cand.get("year")
                fuzzy = _title_similarity(album_title, cand_title)
                if fuzzy < cfg.min_fuzzy_similarity:
                    continue

                year_match = (
                    1
                    if aoty_year is not None and cand_year == aoty_year
                    else 0
                )
                key = (
                    year_match,
                    fuzzy,
                    -int(cand.get("relevance_rank", 0)),
                )
                if best_key is None or key > best_key:
                    best_key = key
                    best = {"master_id": cand_master_id, "title": cand_title}

            if best is None:
                if stats_out is not None:
                    stats_out["aoty_rows_no_discogs_match_after_fuzzy"] += 1
                continue

            master_id = str(best["master_id"])
            assert best_key is not None
            prev_score = master_to_score.get(master_id)
            if prev_score is None or best_key > prev_score:
                master_to_aoty[master_id] = aoty_album_id
                master_to_score[master_id] = best_key
    finally:
        if own_http:
            http.save_disk()

    if stats_out is not None:
        stats_out["discogs_master_to_aoty_entries"] = len(master_to_aoty)
        stats_out["aoty_album_rows_not_linked_to_discogs_master"] = (
            stats_out["aoty_rows_skipped_missing_artist_or_title"]
            + stats_out["aoty_rows_no_discogs_search_results"]
            + stats_out["aoty_rows_no_discogs_match_after_fuzzy"]
        )

    return master_to_aoty


def build_discogs_release_to_aoty_map(
    release_ids: Iterable[str],
    *,
    discogs_master_to_aoty: dict[str, str],
    http: DiscogsHttpHelper,
) -> dict[str, str]:
    """
    For each Discogs release id, resolve master → AOTY album id.

    Returns only releases that map successfully (same semantics as
    ``map_discogs_release_ids_to_aoty_album_ids`` keep set).
    """
    out: dict[str, str] = {}
    for rid in sorted({str(x) for x in release_ids}):
        master_id = _get_discogs_release_master_id(http=http, release_id=rid)
        if master_id is None:
            continue
        aoty_id = discogs_master_to_aoty.get(master_id)
        if aoty_id is not None:
            out[rid] = aoty_id
    return out


def apply_release_to_aoty_map(
    df: pd.DataFrame,
    release_to_aoty: dict[str, str],
    *,
    stats_out: dict[str, int] | None = None,
) -> pd.DataFrame:
    """
    Replace ``album_id`` (Discogs release id) using a precomputed
    ``release_to_aoty`` dict. Rows with unknown releases are dropped.
    """
    _empty_stats: dict[str, int] = {
        "input_rows": 0,
        "output_rows": 0,
        "rows_dropped": 0,
        "unique_discogs_releases": 0,
        "unique_releases_mapped_to_aoty": 0,
        "unique_releases_no_master": 0,
        "unique_releases_master_not_in_aoty_catalog": 0,
    }
    if df.empty:
        if stats_out is not None:
            stats_out.clear()
            stats_out.update(_empty_stats)
        return df
    if "album_id" not in df.columns:
        raise ValueError(
            "Expected df with an `album_id` column (Discogs release_id)."
        )
    out = df.copy()
    out["album_id"] = out["album_id"].astype(str)
    unique_release_ids = sorted(out["album_id"].unique())
    mapped = {
        r: release_to_aoty[r]
        for r in unique_release_ids
        if r in release_to_aoty
    }
    no_mapping = {r for r in unique_release_ids if r not in release_to_aoty}

    out["album_id_mapped"] = out["album_id"].map(mapped)
    out = out.dropna(subset=["album_id_mapped"]).copy()
    out["album_id"] = out["album_id_mapped"].astype(str)
    result = out.drop(columns=["album_id_mapped"])

    if stats_out is not None:
        in_rows = int(len(df))
        out_rows = int(len(result))
        stats_out.clear()
        stats_out.update(
            {
                "input_rows": in_rows,
                "output_rows": out_rows,
                "rows_dropped": in_rows - out_rows,
                "unique_discogs_releases": len(unique_release_ids),
                "unique_releases_mapped_to_aoty": len(mapped),
                "unique_releases_no_master": 0,
                "unique_releases_master_not_in_aoty_catalog": len(no_mapping),
            }
        )

    return result


def save_release_to_aoty_json(path: Path, mapping: dict[str, str]) -> None:
    """Write ``{discogs_release_id: aoty_album_id}`` JSON (sorted keys)."""
    _atomic_write_json(path, dict(sorted(mapping.items())))


def load_release_to_aoty_json(path: Path) -> dict[str, str]:
    """Load release→AOTY map; non-string values are skipped."""
    raw = _load_json_dict(path)
    out: dict[str, str] = {}
    for k, v in raw.items():
        if isinstance(k, str) and isinstance(v, str):
            out[k] = v
        elif isinstance(k, str) and v is not None:
            out[k] = str(v)
    return out


def _get_discogs_release_master_id(
    *,
    http: DiscogsHttpHelper,
    release_id: str,
) -> str | None:
    data = http.get_release_payload(str(release_id))
    mid = data.get("master_id")
    if mid is None:
        return None
    return str(mid)


def map_discogs_release_ids_to_aoty_album_ids(
    df: pd.DataFrame,
    *,
    discogs_token: str | None = None,
    discogs_master_to_aoty: dict[str, str],
    session: requests.Session | None = None,
    cache_dir: Path | None = None,
    cfg: DiscogsMatchConfig | None = None,
    http: DiscogsHttpHelper | None = None,
    stats_out: dict[str, int] | None = None,
) -> pd.DataFrame:
    """
    Replace Discogs `release_id` values in `df["album_id"]` with canonical
    AOTY `album_id` values.

    If ``stats_out`` is a dict, it is cleared and filled with row/release
    counts (e.g. ``rows_dropped``, ``unique_releases_no_master``).
    """
    _empty_stats: dict[str, int] = {
        "input_rows": 0,
        "output_rows": 0,
        "rows_dropped": 0,
        "unique_discogs_releases": 0,
        "unique_releases_mapped_to_aoty": 0,
        "unique_releases_no_master": 0,
        "unique_releases_master_not_in_aoty_catalog": 0,
    }
    if df.empty:
        if stats_out is not None:
            stats_out.clear()
            stats_out.update(_empty_stats)
        return df
    if "album_id" not in df.columns:
        raise ValueError(
            "Expected df with an `album_id` column (Discogs release_id)."
        )

    own_http = http is None
    if http is None:
        if not discogs_token:
            raise ValueError("discogs_token required unless http is provided")
        http = DiscogsHttpHelper(
            session or requests.Session(),
            discogs_token,
            cfg or DiscogsMatchConfig(),
            cache_dir,
        )

    try:
        out = df.copy()
        out["album_id"] = out["album_id"].astype(str)

        unique_release_ids = sorted(out["album_id"].unique())
        release_to_aoty: dict[str, str] = {}
        no_master: set[str] = set()
        master_not_in_catalog: set[str] = set()

        for rid in unique_release_ids:
            master_id = _get_discogs_release_master_id(
                http=http,
                release_id=rid,
            )
            if master_id is None:
                no_master.add(rid)
                continue
            aoty_id = discogs_master_to_aoty.get(master_id)
            if aoty_id is None:
                master_not_in_catalog.add(rid)
                continue
            release_to_aoty[rid] = aoty_id

        out["album_id_mapped"] = out["album_id"].map(release_to_aoty)
        out = out.dropna(subset=["album_id_mapped"]).copy()
        out["album_id"] = out["album_id_mapped"].astype(str)
        result = out.drop(columns=["album_id_mapped"])

        if stats_out is not None:
            in_rows = int(len(df))
            out_rows = int(len(result))
            stats_out.clear()
            stats_out.update(
                {
                    "input_rows": in_rows,
                    "output_rows": out_rows,
                    "rows_dropped": in_rows - out_rows,
                    "unique_discogs_releases": len(unique_release_ids),
                    "unique_releases_mapped_to_aoty": len(release_to_aoty),
                    "unique_releases_no_master": len(no_master),
                    "unique_releases_master_not_in_aoty_catalog": len(
                        master_not_in_catalog
                    ),
                }
            )

        return result
    finally:
        if own_http:
            http.save_disk()
