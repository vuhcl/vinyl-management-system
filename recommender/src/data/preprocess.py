"""
Data cleaning and normalization. Build unified interactions with weights.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd


def _numeric_star_rating(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def album_ids_with_only_low_star_ratings(
    ratings: pd.DataFrame,
    *,
    at_least_stars: float = 3.0,
) -> set[str]:
    """
    Album IDs where every numeric rating in ``ratings`` is strictly below
    ``at_least_stars`` (default 3.0 → only 1–2★ ever observed for that album).

    Rows with non-numeric ratings are ignored when computing per-album max;
    albums with no valid numeric ratings are not flagged.
    """
    if ratings.empty or "rating" not in ratings.columns or "album_id" not in ratings.columns:
        return set()
    rv = ratings["rating"].map(_numeric_star_rating)
    tmp = ratings.assign(_rv=rv).dropna(subset=["_rv"])
    if tmp.empty:
        return set()
    aid = tmp["album_id"].astype(str)
    mx = tmp.groupby(aid, sort=False)["_rv"].max()
    return set(mx.index[mx < float(at_least_stars)].astype(str).tolist())


def _coerce_bool_flag(value: object, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return default


def _rating_to_strength(value: object, weights: Mapping[str, Any]) -> float:
    """
    Map numeric rating (1–5 scale) to implicit feedback strength.

    Keeps 1–2★ as weak positive signal (tunable via ``rating_low_1`` /
    ``rating_low_2`` in ``weights``).
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    if v >= 4.0:
        return float(weights.get("rating_high", 2.5))
    if v >= 3.0:
        return float(weights.get("rating_mid", 1.5))
    if v >= 2.0:
        return float(weights.get("rating_low_2", 0.8))
    if v >= 1.0:
        return float(weights.get("rating_low_1", 0.4))
    return 0.0


def apply_weights(
    collection: pd.DataFrame,
    wantlist: pd.DataFrame,
    ratings: pd.DataFrame,
    weights: Mapping[str, Any],
) -> pd.DataFrame:
    """
    Merge collection, wantlist, ratings into one interaction table with strength.
    Weights: collection, wantlist, and rating tiers:
    ``rating_high`` (>=4), ``rating_mid`` (>=3), ``rating_low_2`` (>=2),
    ``rating_low_1`` (>=1).

    Optional flags (non-strength keys, consumed here only):

    - ``drop_exclusively_low_rating_albums`` (default True): remove all
      *rating-sourced* rows for any album whose max observed star rating is
      strictly below ``exclusive_low_rating_below`` (default 3.0), i.e. the
      album only ever appears with 1–2★ in the ratings table.
    - ``exclusive_low_rating_below``: threshold in stars (default 3.0).
    """
    rows = []

    if not collection.empty:
        c = collection[["user_id", "album_id"]].copy()
        c["strength"] = weights.get("collection", 1.0)
        c["source"] = "collection"
        rows.append(c)

    if not wantlist.empty:
        w = wantlist[["user_id", "album_id"]].copy()
        w["strength"] = weights.get("wantlist", 2.0)
        w["source"] = "wantlist"
        rows.append(w)

    if not ratings.empty and "rating" in ratings.columns:
        r = ratings.copy()
        if _coerce_bool_flag(
            weights.get("drop_exclusively_low_rating_albums"), default=True
        ):
            floor = float(weights.get("exclusive_low_rating_below", 3.0))
            bad_albums = album_ids_with_only_low_star_ratings(
                r, at_least_stars=floor
            )
            if bad_albums:
                r = r[~r["album_id"].astype(str).isin(bad_albums)]
        if r.empty:
            pass
        else:
            r["strength"] = r["rating"].map(
                lambda x: _rating_to_strength(x, weights)
            )
            r = r[r["strength"] > 0][["user_id", "album_id", "strength"]]
            r["source"] = "rating"
            rows.append(r)

    if not rows:
        return pd.DataFrame(columns=["user_id", "album_id", "strength", "source"])

    out = pd.concat(rows, ignore_index=True)
    # Aggregate same user-album from multiple sources: take max strength
    out = out.groupby(["user_id", "album_id"], as_index=False).agg({"strength": "max", "source": "first"})
    return out


def normalize_ids(interactions: pd.DataFrame, albums: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ensure album_id types match and filter to albums we have metadata for (if provided)."""
    interactions = interactions.copy()
    interactions["user_id"] = interactions["user_id"].astype(str)
    interactions["album_id"] = interactions["album_id"].astype(str)

    if not albums.empty and "album_id" in albums.columns:
        valid_ids = set(albums["album_id"].astype(str))
        interactions = interactions[interactions["album_id"].isin(valid_ids)]

    if not albums.empty:
        albums = albums.copy()
        albums["album_id"] = albums["album_id"].astype(str)
    return interactions, albums


def preprocess(
    raw: dict[str, pd.DataFrame],
    weights: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Clean and normalize. Returns (interactions, albums).
    """
    collection = raw.get("collection", pd.DataFrame())
    wantlist = raw.get("wantlist", pd.DataFrame())
    ratings = raw.get("ratings", pd.DataFrame())
    albums = raw.get("albums", pd.DataFrame())

    interactions = apply_weights(collection, wantlist, ratings, weights)
    interactions, albums = normalize_ids(interactions, albums)
    return interactions, albums


def save_processed(interactions: pd.DataFrame, albums: pd.DataFrame, out_dir: Path) -> None:
    """Write processed interactions and albums to out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    interactions.to_parquet(out_dir / "interactions.parquet", index=False)
    albums.to_parquet(out_dir / "albums.parquet", index=False)


def load_processed(out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load processed interactions and albums from out_dir."""
    out_dir = Path(out_dir)
    interactions = pd.read_parquet(out_dir / "interactions.parquet")
    albums = pd.read_parquet(out_dir / "albums.parquet")
    return interactions, albums
