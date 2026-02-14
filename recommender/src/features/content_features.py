"""
Content-based features: genre, artist, year, album rating.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


def prepare_album_features(
    albums: pd.DataFrame,
    genre_col: str = "genre",
    artist_col: str = "artist",
    year_col: str = "year",
    top_k_genres: int = 20,
) -> tuple[pd.DataFrame, dict]:
    """
    One-hot genres (top_k), label-encode artist, bucket year, keep avg_rating.
    Returns (album feature DataFrame with album_id index, encoders dict).
    """
    df = albums.copy()
    df["album_id"] = df["album_id"].astype(str)
    df = df.set_index("album_id")

    encoders = {}
    out = pd.DataFrame(index=df.index)

    if year_col in df.columns:
        y = df[year_col].fillna(0).astype(int).clip(1900, 2030)
        out["year"] = y
        out["year_norm"] = MinMaxScaler().fit_transform(out[["year"]])
        out = out.drop(columns=["year"])

    if "avg_rating" in df.columns:
        out["avg_rating"] = df["avg_rating"].fillna(0)
        out["avg_rating"] = MinMaxScaler().fit_transform(out[["avg_rating"]].fillna(0))

    if genre_col in df.columns:
        # Flatten genres (e.g. "Rock,Pop" -> multiple rows or multi-hot)
        all_genres = []
        for g in df[genre_col].dropna().astype(str):
            all_genres.extend([x.strip() for x in g.split(",")])
        from collections import Counter
        top_genres = [t for t, _ in Counter(all_genres).most_common(top_k_genres)]
        encoders["genre_list"] = top_genres
        for g in top_genres:
            out[f"genre_{g}"] = df[genre_col].fillna("").astype(str).str.contains(g, regex=False).astype(float)
    else:
        encoders["genre_list"] = []

    if artist_col in df.columns:
        le = LabelEncoder()
        out["artist_enc"] = le.fit_transform(df[artist_col].fillna("Unknown").astype(str))
        encoders["artist_enc"] = le
    else:
        out["artist_enc"] = 0
        encoders["artist_enc"] = None

    return out, encoders


def album_feature_matrix(album_features: pd.DataFrame) -> np.ndarray:
    """Dense matrix (n_albums x n_features) for cosine similarity or meta-model."""
    return album_features.values.astype(np.float64)


def cosine_similarity_matrix(X: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity; X is (n_samples, n_features)."""
    from numpy.linalg import norm
    X = np.nan_to_num(X, nan=0.0)
    norms = norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1
    Xn = X / norms
    return Xn @ Xn.T
