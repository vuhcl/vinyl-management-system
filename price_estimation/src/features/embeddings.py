"""
Genre and artist features: one-hot or frequency encoding for baseline/advanced models.
Placeholder for learned embeddings in Phase 2 if desired.
"""
from typing import Any

import numpy as np
import pandas as pd


def build_genre_artist_features(
    df: pd.DataFrame,
    *,
    genre_col: str = "genre",
    artist_col: str = "artist",
    year_col: str = "release_year",
    genre_top_k: int = 50,
    artist_top_k: int = 100,
    fill_missing: str = "Unknown",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Build one-hot (or top-k frequency) features for genre and artist.
    Returns (df_with_features, encoder_state) so the same encoding can be applied at predict time.
    """
    out = df.copy()
    state: dict[str, Any] = {}

    if genre_col in out.columns:
        out[genre_col] = out[genre_col].fillna(fill_missing).astype(str).str.strip()
        counts = out[genre_col].value_counts()
        top_genres = counts.head(genre_top_k).index.tolist()
        state["genre_top_k"] = top_genres
        for g in top_genres:
            out[f"genre_{g.replace(' ', '_')}"] = (out[genre_col] == g).astype(int)
        out[f"genre_other"] = (~out[genre_col].isin(top_genres)).astype(int)

    if artist_col in out.columns:
        out[artist_col] = out[artist_col].fillna(fill_missing).astype(str).str.strip()
        counts = out[artist_col].value_counts()
        top_artists = counts.head(artist_top_k).index.tolist()
        state["artist_top_k"] = top_artists
        for a in top_artists[:20]:
            safe = a.replace(" ", "_").replace("/", "_")[:30]
            out[f"artist_{safe}"] = (out[artist_col] == a).astype(int)
        out["artist_other"] = (~out[artist_col].isin(top_artists)).astype(int)

    if year_col in out.columns:
        out[year_col] = pd.to_numeric(out[year_col], errors="coerce")
        out["release_year"] = out[year_col].fillna(0).astype(int)
        state["year_col"] = year_col

    return out, state
