"""
Metadata-based candidate generation for two-stage recommenders.

Stage 1: union of (genre-expanded catalog) + (same-artist albums), then
filters: avg rating, train counts, distinct users, rating rows, priority,
optional user-specific year quantile band (+ optional year slack).

Stage 2: ALS scores only within the candidate index set.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


def _split_genres(genre_val: object) -> set[str]:
    """Normalize genre field to a set of lowercase tokens."""
    if genre_val is None or (isinstance(genre_val, float) and np.isnan(genre_val)):
        return set()
    if isinstance(genre_val, list):
        return {str(g).strip().lower() for g in genre_val if str(g).strip()}
    s = str(genre_val).strip()
    if not s:
        return set()
    return {p.strip().lower() for p in s.split(",") if p.strip()}


@dataclass
class CandidateRetrievalConfig:
    """Tuning knobs for candidate pools."""

    min_avg_rating: float = 0.0
    """Minimum album avg_rating (0–5) after preprocessing."""

    min_train_count: int = 1
    """Minimum train interaction rows per album (post user–album dedupe)."""

    max_candidates: int = 2000
    """Cap pool size after filtering (deterministic sort then trim)."""

    use_genre_expansion: bool = True
    """Include all albums sharing any genre with user's train albums."""

    use_same_artist_expansion: bool = True
    """Include other albums by artists appearing in user's train albums."""

    fallback_to_full_catalog: bool = True
    """If the pool is empty after filters, use all valid items (minus train)."""

    use_year_quantile_filter: bool = True
    """Restrict candidates by release year vs user's train-album year quantiles."""

    year_quantile_low: float = 0.1
    year_quantile_high: float = 0.9
    """Quantiles over positive train years (inclusive band after floor/ceil)."""

    year_window_years: int | None = None
    """If set, widen the quantile band by this many years on each side."""

    min_distinct_users: int = 1
    """Minimum distinct users with any train interaction on the album."""

    min_rating_rows: int = 0
    """Minimum train rows with source == rating (0 = no extra filter)."""

    min_priority_score: float | None = None
    """If set, require album priority_score >= this value."""


def retrieval_config_from_dict(d: dict) -> CandidateRetrievalConfig:
    """Build config from YAML/script dict (same keys as evaluation)."""
    yw = d.get("year_window_years")
    mp = d.get("min_priority_score")
    return CandidateRetrievalConfig(
        min_avg_rating=float(d.get("min_avg_rating", 0.0)),
        min_train_count=int(d.get("min_train_count", 1)),
        max_candidates=int(d.get("max_candidates", 2000)),
        use_genre_expansion=bool(d.get("use_genre_expansion", True)),
        use_same_artist_expansion=bool(d.get("use_same_artist_expansion", True)),
        fallback_to_full_catalog=bool(d.get("fallback_to_full_catalog", True)),
        use_year_quantile_filter=bool(d.get("use_year_quantile_filter", True)),
        year_quantile_low=float(d.get("year_quantile_low", 0.1)),
        year_quantile_high=float(d.get("year_quantile_high", 0.9)),
        year_window_years=int(yw) if yw is not None else None,
        min_distinct_users=int(d.get("min_distinct_users", 1)),
        min_rating_rows=int(d.get("min_rating_rows", 0)),
        min_priority_score=float(mp) if mp is not None else None,
    )


@dataclass
class RetrievalMetadata:
    """Precomputed indexes for fast per-user candidate generation."""

    album_genres: dict[str, set[str]]
    genre_to_albums: dict[str, set[str]]
    artist_to_albums: dict[str, set[str]]
    album_id_to_artist: dict[str, str]
    album_avg_rating: dict[str, float]
    album_train_count: dict[str, int]
    album_year: dict[str, int]
    album_release_date: dict[str, str]
    album_priority: dict[str, float]
    album_distinct_users: dict[str, int]
    album_rating_rows: dict[str, int]
    valid_album_ids: set[str]


def _empty_meta() -> RetrievalMetadata:
    return RetrievalMetadata(
        album_genres={},
        genre_to_albums={},
        artist_to_albums={},
        album_id_to_artist={},
        album_avg_rating={},
        album_train_count={},
        album_year={},
        album_release_date={},
        album_priority={},
        album_distinct_users={},
        album_rating_rows={},
        valid_album_ids=set(),
    )


def build_retrieval_metadata(
    albums: pd.DataFrame,
    train_interactions: pd.DataFrame,
    *,
    genre_col: str = "genre",
    artist_col: str = "artist",
    album_id_col: str = "album_id",
    avg_rating_col: str = "avg_rating",
    year_col: str = "year",
    release_date_col: str = "release_date",
    priority_col: str = "priority_score",
) -> RetrievalMetadata:
    """
    Build inverted indexes from album metadata + train interaction aggregates.

    `albums` must include at least album_id; optional genre, artist, avg_rating,
    year, release_date, priority_score.
    """
    if albums.empty or album_id_col not in albums.columns:
        return _empty_meta()

    a = albums.copy()
    a[album_id_col] = a[album_id_col].astype(str)

    album_genres: dict[str, set[str]] = {}
    genre_to_albums: dict[str, set[str]] = {}
    artist_to_albums: dict[str, set[str]] = {}
    album_id_to_artist: dict[str, str] = {}
    album_avg_rating: dict[str, float] = {}
    album_year: dict[str, int] = {}
    album_release_date: dict[str, str] = {}
    album_priority: dict[str, float] = {}

    for _, row in a.iterrows():
        aid = str(row[album_id_col])
        if genre_col in a.columns:
            gs = _split_genres(row.get(genre_col))
        else:
            gs = set()
        album_genres[aid] = gs
        for g in gs:
            genre_to_albums.setdefault(g, set()).add(aid)

        if artist_col in a.columns:
            art = str(row.get(artist_col) or "").strip().lower()
            album_id_to_artist[aid] = art
            if art:
                artist_to_albums.setdefault(art, set()).add(aid)
        else:
            album_id_to_artist[aid] = ""

        if avg_rating_col in a.columns:
            v = row.get(avg_rating_col)
            try:
                album_avg_rating[aid] = float(v) if pd.notna(v) else 0.0
            except (TypeError, ValueError):
                album_avg_rating[aid] = 0.0
        else:
            album_avg_rating[aid] = 0.0

        if year_col in a.columns:
            try:
                yv = int(row.get(year_col)) if pd.notna(row.get(year_col)) else 0
            except (TypeError, ValueError):
                yv = 0
            album_year[aid] = max(0, yv)
        else:
            album_year[aid] = 0

        if release_date_col in a.columns:
            rd = row.get(release_date_col)
            album_release_date[aid] = (
                str(rd).strip() if rd is not None and str(rd).strip() else ""
            )
        else:
            album_release_date[aid] = ""

        if priority_col in a.columns:
            try:
                pv = float(row.get(priority_col)) if pd.notna(row.get(priority_col)) else 0.0
            except (TypeError, ValueError):
                pv = 0.0
            album_priority[aid] = pv
        else:
            album_priority[aid] = 0.0

    valid_album_ids = set(a[album_id_col].astype(str))

    g_album = train_interactions.groupby("album_id", sort=False)
    album_train_count = {str(k): int(v) for k, v in g_album.size().items()}
    album_distinct_users = {
        str(k): int(v) for k, v in g_album["user_id"].nunique().items()
    }

    if "source" in train_interactions.columns:
        rmask = train_interactions["source"].astype(str) == "rating"
        sub = train_interactions.loc[rmask]
        if not sub.empty:
            album_rating_rows = {
                str(k): int(v) for k, v in sub.groupby("album_id").size().items()
            }
        else:
            album_rating_rows = {}
    else:
        album_rating_rows = {}

    return RetrievalMetadata(
        album_genres=album_genres,
        genre_to_albums=genre_to_albums,
        artist_to_albums=artist_to_albums,
        album_id_to_artist=album_id_to_artist,
        album_avg_rating=album_avg_rating,
        album_train_count=album_train_count,
        album_year=album_year,
        album_release_date=album_release_date,
        album_priority=album_priority,
        album_distinct_users=album_distinct_users,
        album_rating_rows=album_rating_rows,
        valid_album_ids=valid_album_ids,
    )


def _user_year_band(
    train_album_ids: set[str],
    meta: RetrievalMetadata,
    cfg: CandidateRetrievalConfig,
) -> tuple[int, int] | None:
    """Inclusive [low, high] release years from user's train albums, or None."""
    if not cfg.use_year_quantile_filter:
        return None
    years = [
        meta.album_year[aid]
        for aid in train_album_ids
        if aid in meta.album_year and meta.album_year[aid] > 0
    ]
    if not years:
        return None
    lo = float(np.quantile(years, cfg.year_quantile_low))
    hi = float(np.quantile(years, cfg.year_quantile_high))
    # Always span every train year (quantiles alone can exclude extremes).
    lo = min(lo, float(min(years)))
    hi = max(hi, float(max(years)))
    if cfg.year_window_years is not None and cfg.year_window_years > 0:
        lo -= float(cfg.year_window_years)
        hi += float(cfg.year_window_years)
    return (int(math.floor(lo)), int(math.ceil(hi)))


def _year_ok(
    aid: str,
    meta: RetrievalMetadata,
    year_band: tuple[int, int] | None,
) -> bool:
    if year_band is None:
        return True
    y = meta.album_year.get(aid, 0)
    if y <= 0:
        return True
    lo, hi = year_band
    return lo <= y <= hi


def _passes_global_filters(
    aid: str,
    meta: RetrievalMetadata,
    cfg: CandidateRetrievalConfig,
    year_band: tuple[int, int] | None,
) -> bool:
    if aid not in meta.valid_album_ids:
        return False
    avg_r = meta.album_avg_rating.get(aid, 0.0)
    if avg_r < cfg.min_avg_rating:
        return False
    cnt = meta.album_train_count.get(aid, 0)
    if cnt < cfg.min_train_count:
        return False
    du = meta.album_distinct_users.get(aid, 0)
    if du < cfg.min_distinct_users:
        return False
    rr = meta.album_rating_rows.get(aid, 0)
    if rr < cfg.min_rating_rows:
        return False
    if cfg.min_priority_score is not None:
        if meta.album_priority.get(aid, 0.0) < cfg.min_priority_score:
            return False
    if not _year_ok(aid, meta, year_band):
        return False
    return True


def _expand_genres(
    seed_albums: set[str],
    meta: RetrievalMetadata,
) -> set[str]:
    out: set[str] = set()
    genres: set[str] = set()
    for aid in seed_albums:
        genres |= meta.album_genres.get(aid, set())
    for g in genres:
        out |= meta.genre_to_albums.get(g, set())
    return out


def _expand_same_artists(seed_albums: set[str], meta: RetrievalMetadata) -> set[str]:
    artists: set[str] = set()
    for aid in seed_albums:
        ar = meta.album_id_to_artist.get(aid, "")
        if ar:
            artists.add(ar)
    out: set[str] = set()
    for ar in artists:
        out |= meta.artist_to_albums.get(ar, set())
    return out


def _sort_key(
    aid: str,
    meta: RetrievalMetadata,
) -> tuple[float, float, int, int, int, str]:
    """Higher is better for first components (negated for sort)."""
    tc = meta.album_train_count.get(aid, 0)
    pr = meta.album_priority.get(aid, 0.0)
    du = meta.album_distinct_users.get(aid, 0)
    rr = meta.album_rating_rows.get(aid, 0)
    return (-tc, -pr, -du, -rr, -meta.album_year.get(aid, 0), aid)


def _collect_from_pool(
    pool: set[str],
    meta: RetrievalMetadata,
    cfg: CandidateRetrievalConfig,
    year_band: tuple[int, int] | None,
) -> list[str]:
    eligible: list[str] = []
    for aid in pool:
        if _passes_global_filters(aid, meta, cfg, year_band):
            eligible.append(aid)
    eligible.sort(key=lambda x: _sort_key(x, meta))
    return eligible[: cfg.max_candidates]


def candidate_album_ids_for_user(
    train_album_ids: set[str],
    meta: RetrievalMetadata,
    cfg: CandidateRetrievalConfig,
) -> set[str]:
    """
    Return album_id strings eligible for stage-2 ALS ranking for one user.

    Uses train-only history (no test leakage).
    """
    seed = {a for a in train_album_ids if a in meta.valid_album_ids}
    year_band = _user_year_band(seed, meta, cfg)

    pool: set[str] = set(seed)
    if cfg.use_genre_expansion:
        pool |= _expand_genres(seed, meta)
    if cfg.use_same_artist_expansion:
        pool |= _expand_same_artists(seed, meta)

    picked = _collect_from_pool(pool, meta, cfg, year_band)

    if not picked and cfg.fallback_to_full_catalog:
        pool = set(meta.valid_album_ids)
        picked = _collect_from_pool(pool, meta, cfg, year_band)

    return set(picked)


def candidate_item_indices_for_user(
    train_album_ids: set[str],
    meta: RetrievalMetadata,
    item_id2idx: dict[str, int],
    cfg: CandidateRetrievalConfig,
) -> np.ndarray:
    """Map candidate album ids to matrix column indices (unknown ids skipped)."""
    aids = candidate_album_ids_for_user(train_album_ids, meta, cfg)
    idxs = [item_id2idx[a] for a in aids if a in item_id2idx]
    return np.asarray(sorted(set(idxs)), dtype=np.int64)
