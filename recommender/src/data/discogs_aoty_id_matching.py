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
  - Retries once on HTTP 429 using `Retry-After` when present.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import requests

DISCogs_BASE_URL = "https://api.discogs.com"
_RELEASE_CACHE_NAME = "release_master.json"
_SEARCH_CACHE_NAME = "search_database.json"


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
    blob = json.dumps(params, sort_keys=True, separators=(",", ":")).encode("utf-8")
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
    _release_master: dict[str, str | None] = field(default_factory=dict, repr=False)
    _search_payloads: dict[str, dict[str, Any]] = field(
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

    def save_disk(self) -> None:
        if self.cache_dir is None or not self._dirty:
            return
        _atomic_write_json(self._release_path(), self._release_master)
        _atomic_write_json(self._search_path(), self._search_payloads)
        self._dirty = False

    def _throttle(self) -> None:
        if self.cfg.min_request_interval_s <= 0:
            return
        now = time.monotonic()
        wait = self.cfg.min_request_interval_s - (now - self._last_request_mono)
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
        attempt = 0
        while True:
            resp = self.session.get(url, params=params or {}, timeout=timeout_s)
            self._last_request_mono = time.monotonic()
            if resp.status_code == 429 and attempt < self.cfg.max_429_retries:
                ra = resp.headers.get("Retry-After", "60")
                try:
                    delay = float(ra)
                except ValueError:
                    delay = 60.0
                time.sleep(delay)
                attempt += 1
                continue
            self._post_request_headers(resp)
            return resp

    def get_database_search(self, params: dict[str, Any]) -> dict[str, Any]:
        key = _search_cache_key(params)
        if key in self._search_payloads:
            return self._search_payloads[key]

        url = f"{DISCogs_BASE_URL}/database/search"
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

        url = f"{DISCogs_BASE_URL}/releases/{rid}"
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
        raise ValueError(f"AOTY albums missing required columns: {sorted(missing)}")

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

    try:
        for row in aoty_albums.itertuples(index=False):
            aoty_album_id = str(getattr(row, "album_id"))
            artist = str(getattr(row, "artist") or "")
            album_title = str(getattr(row, "album_title") or "")
            aoty_year = _extract_year(getattr(row, "year"))

            if not artist or not album_title:
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
                    1 if aoty_year is not None and cand_year == aoty_year else 0
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

    return master_to_aoty


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
) -> pd.DataFrame:
    """
    Replace Discogs `release_id` values in `df["album_id"]` with canonical
    AOTY `album_id` values.
    """
    if df.empty:
        return df
    if "album_id" not in df.columns:
        raise ValueError("Expected df with an `album_id` column (Discogs release_id).")

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

        for rid in unique_release_ids:
            master_id = _get_discogs_release_master_id(
                http=http,
                release_id=rid,
            )
            if master_id is None:
                continue
            aoty_id = discogs_master_to_aoty.get(master_id)
            if aoty_id is not None:
                release_to_aoty[rid] = aoty_id

        out["album_id_mapped"] = out["album_id"].map(release_to_aoty)
        out = out.dropna(subset=["album_id_mapped"]).copy()
        out["album_id"] = out["album_id_mapped"].astype(str)
        return out.drop(columns=["album_id_mapped"])
    finally:
        if own_http:
            http.save_disk()
