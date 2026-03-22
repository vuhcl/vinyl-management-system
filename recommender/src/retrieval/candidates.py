"""
Metadata-based candidate generation for two-stage recommenders.

Stage 1: union of (genre-expanded catalog) + (same-artist albums), then
quality floors (avg rating, min global interaction count).

Stage 2: ALS scores only within the candidate index set.
"""

from __future__ import annotations

from dataclasses import dataclass, field

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
    """Minimum number of train interactions on an album (popularity floor)."""

    max_candidates: int = 2000
    """Cap pool size after filtering (deterministic trim by train_count)."""

    use_genre_expansion: bool = True
    """Include all albums sharing any genre with user's train albums."""

    use_same_artist_expansion: bool = True
    """Include other albums by artists appearing in user's train albums."""

    fallback_to_full_catalog: bool = True
    """If the pool is empty after filters, use all valid items (minus train)."""


def retrieval_config_from_dict(d: dict) -> CandidateRetrievalConfig:
    """Build config from YAML/script dict (same keys as evaluation)."""
    return CandidateRetrievalConfig(
        min_avg_rating=float(d.get("min_avg_rating", 0.0)),
        min_train_count=int(d.get("min_train_count", 1)),
        max_candidates=int(d.get("max_candidates", 2000)),
        use_genre_expansion=bool(d.get("use_genre_expansion", True)),
        use_same_artist_expansion=bool(d.get("use_same_artist_expansion", True)),
        fallback_to_full_catalog=bool(d.get("fallback_to_full_catalog", True)),
    )


@dataclass
class RetrievalMetadata:
    """Precomputed indexes for fast per-user candidate generation."""

    album_genres: dict[str, set[str]]
    genre_to_albums: dict[str, set[str]]
    artist_to_albums: dict[str, set[str]]
    """album_id -> lowercase artist name (for same-artist expansion)."""
    album_id_to_artist: dict[str, str]
    album_avg_rating: dict[str, float]
    album_train_count: dict[str, int]
    valid_album_ids: set[str]


def build_retrieval_metadata(
    albums: pd.DataFrame,
    train_interactions: pd.DataFrame,
    *,
    genre_col: str = "genre",
    artist_col: str = "artist",
    album_id_col: str = "album_id",
    avg_rating_col: str = "avg_rating",
) -> RetrievalMetadata:
    """
    Build inverted indexes from album metadata + train interaction counts.

    `albums` must include at least album_id; optional genre, artist, avg_rating.
    """
    if albums.empty or album_id_col not in albums.columns:
        return RetrievalMetadata(
            album_genres={},
            genre_to_albums={},
            artist_to_albums={},
            album_id_to_artist={},
            album_avg_rating={},
            album_train_count={},
            valid_album_ids=set(),
        )

    a = albums.copy()
    a[album_id_col] = a[album_id_col].astype(str)

    album_genres: dict[str, set[str]] = {}
    genre_to_albums: dict[str, set[str]] = {}
    artist_to_albums: dict[str, set[str]] = {}
    album_id_to_artist: dict[str, str] = {}
    album_avg_rating: dict[str, float] = {}

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

    valid_album_ids = set(a[album_id_col].astype(str))

    tc = train_interactions.groupby("album_id").size()
    album_train_count = {str(k): int(v) for k, v in tc.items()}

    return RetrievalMetadata(
        album_genres=album_genres,
        genre_to_albums=genre_to_albums,
        artist_to_albums=artist_to_albums,
        album_id_to_artist=album_id_to_artist,
        album_avg_rating=album_avg_rating,
        album_train_count=album_train_count,
        valid_album_ids=valid_album_ids,
    )


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
    pool: set[str] = set(seed)

    if cfg.use_genre_expansion:
        pool |= _expand_genres(seed, meta)

    if cfg.use_same_artist_expansion:
        pool |= _expand_same_artists(seed, meta)

    # Quality filters
    filtered: list[tuple[str, float, int]] = []
    for aid in pool:
        if aid not in meta.valid_album_ids:
            continue
        avg_r = meta.album_avg_rating.get(aid, 0.0)
        if avg_r < cfg.min_avg_rating:
            continue
        cnt = meta.album_train_count.get(aid, 0)
        if cnt < cfg.min_train_count:
            continue
        filtered.append((aid, avg_r, cnt))

    if not filtered and cfg.fallback_to_full_catalog:
        for aid in meta.valid_album_ids:
            avg_r = meta.album_avg_rating.get(aid, 0.0)
            if avg_r < cfg.min_avg_rating:
                continue
            cnt = meta.album_train_count.get(aid, 0)
            if cnt < cfg.min_train_count:
                continue
            filtered.append((aid, avg_r, cnt))

    if not filtered:
        return set()

    # Deterministic cap: prefer higher global popularity (train count)
    filtered.sort(key=lambda t: (-t[2], t[0]))
    capped = filtered[: cfg.max_candidates]
    return {t[0] for t in capped}


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
