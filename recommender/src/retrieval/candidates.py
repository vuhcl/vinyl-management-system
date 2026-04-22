"""
Album metadata helpers for reranker features.

Builds per-album inverted indexes (genres, artists, years, ratings) from
`albums.parquet` + train interactions. Used by the reranker to compute
content-based features (genre Jaccard, artist match, year distance, etc.)
on top of ALS scores. Stage-1 recall is always full-catalog ALS.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

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
class RetrievalMetadata:
    """Precomputed indexes for reranker feature computation."""

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
    # Discogs dump + marketplace (optional; empty dicts = Phase-1-only reranker)
    album_release_count: dict[str, int] = field(default_factory=dict)
    album_vinyl_release_count: dict[str, int] = field(default_factory=dict)
    album_unique_country_count: dict[str, int] = field(default_factory=dict)
    album_unique_label_count: dict[str, int] = field(default_factory=dict)
    album_era_span: dict[str, int] = field(default_factory=dict)
    album_has_discogs_master: dict[str, bool] = field(default_factory=dict)
    album_community_want: dict[str, int] = field(default_factory=dict)
    album_community_have: dict[str, int] = field(default_factory=dict)
    album_num_for_sale: dict[str, int] = field(default_factory=dict)
    album_lowest_price: dict[str, float] = field(default_factory=dict)
    album_has_community_stats: dict[str, bool] = field(default_factory=dict)


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
        album_release_count={},
        album_vinyl_release_count={},
        album_unique_country_count={},
        album_unique_label_count={},
        album_era_span={},
        album_has_discogs_master={},
        album_community_want={},
        album_community_have={},
        album_num_for_sale={},
        album_lowest_price={},
        album_has_community_stats={},
    )


def load_discogs_stats_for_reranker_cfg(
    reranker_cfg: dict | None,
) -> pd.DataFrame | None:
    """
    Resolve ``reranker.discogs_master_stats_path`` from YAML and load the
    parquet when present. Relative paths are rooted at the git project root
    (``core.config.get_project_root()``).
    """
    if not reranker_cfg:
        return None
    raw = reranker_cfg.get("discogs_master_stats_path")
    if not raw:
        return None
    from core.config import get_project_root

    p = Path(str(raw)).expanduser()
    if not p.is_absolute():
        p = get_project_root() / p
    return load_discogs_master_stats_parquet(p)


def load_discogs_master_stats_parquet(path: Path | str | None) -> pd.DataFrame | None:
    """
    Load ``discogs_master_stats.parquet`` built by
    ``scripts/build_discogs_master_stats_artifact.py``. Returns ``None`` if
    *path* is falsy or the file is missing / unreadable.
    """
    if not path:
        return None
    p = Path(path).expanduser()
    if not p.is_file():
        return None
    try:
        df = pd.read_parquet(p)
    except Exception:
        return None
    if df.empty or "album_id" not in df.columns:
        return None
    return df


def build_retrieval_metadata(
    albums: pd.DataFrame,
    train_interactions: pd.DataFrame,
    *,
    discogs_master_stats: pd.DataFrame | None = None,
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

    When ``discogs_master_stats`` is a non-empty DataFrame (from
    ``load_discogs_master_stats_parquet``), per-album Discogs aggregates are
    attached for reranker Tier A / Tier B features. Missing albums simply omit
    keys so the reranker zero-imputes with ``has_*`` flags.
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

    album_release_count: dict[str, int] = {}
    album_vinyl_release_count: dict[str, int] = {}
    album_unique_country_count: dict[str, int] = {}
    album_unique_label_count: dict[str, int] = {}
    album_era_span: dict[str, int] = {}
    album_has_discogs_master: dict[str, bool] = {}
    album_community_want: dict[str, int] = {}
    album_community_have: dict[str, int] = {}
    album_num_for_sale: dict[str, int] = {}
    album_lowest_price: dict[str, float] = {}
    album_has_community_stats: dict[str, bool] = {}

    if discogs_master_stats is not None and not discogs_master_stats.empty:
        ds = discogs_master_stats
        aids = ds["album_id"].astype(str)
        for col, target in (
            ("release_count", album_release_count),
            ("vinyl_release_count", album_vinyl_release_count),
            ("unique_country_count", album_unique_country_count),
            ("unique_label_count", album_unique_label_count),
            ("era_span", album_era_span),
            ("community_want", album_community_want),
            ("community_have", album_community_have),
            ("num_for_sale", album_num_for_sale),
        ):
            if col not in ds.columns:
                continue
            for a, v in zip(aids, ds[col].fillna(0).astype(int).tolist()):
                target[str(a)] = int(v)
        if "lowest_price" in ds.columns:
            for a, v in zip(aids, ds["lowest_price"].tolist()):
                try:
                    fv = float(v) if v is not None and not pd.isna(v) else 0.0
                except (TypeError, ValueError):
                    fv = 0.0
                album_lowest_price[str(a)] = fv
        if "has_community_stats" in ds.columns:
            for a, v in zip(aids, ds["has_community_stats"].fillna(0).tolist()):
                album_has_community_stats[str(a)] = bool(int(v))
        if "release_count" in ds.columns:
            for a, rc in zip(aids, ds["release_count"].fillna(0).tolist()):
                album_has_discogs_master[str(a)] = int(rc or 0) > 0

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
        album_release_count=album_release_count,
        album_vinyl_release_count=album_vinyl_release_count,
        album_unique_country_count=album_unique_country_count,
        album_unique_label_count=album_unique_label_count,
        album_era_span=album_era_span,
        album_has_discogs_master=album_has_discogs_master,
        album_community_want=album_community_want,
        album_community_have=album_community_have,
        album_num_for_sale=album_num_for_sale,
        album_lowest_price=album_lowest_price,
        album_has_community_stats=album_has_community_stats,
    )
