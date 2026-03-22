"""
Content-based model: cosine similarity or user-profile scoring.
"""

import numpy as np
import pandas as pd

from ..features.content_features import album_feature_matrix, cosine_similarity_matrix


def build_content_similarity(album_features: pd.DataFrame) -> np.ndarray:
    """Pairwise album-album cosine similarity matrix."""
    X = album_feature_matrix(album_features)
    return cosine_similarity_matrix(X)


def score_content_for_user(
    user_album_ids: np.ndarray,
    album_id_to_idx: dict,
    item_ids: np.ndarray,
    sim_matrix: np.ndarray,
    exclude_item_idxs: np.ndarray,
    top_k: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Score items by average similarity to user's albums. Exclude owned.
    Returns (item_indices, scores) of length up to top_k.
    """
    idx_to_album = {i: a for i, a in enumerate(item_ids)}
    user_idxs = [album_id_to_idx[a] for a in user_album_ids if a in album_id_to_idx]
    if not user_idxs:
        return np.array([], dtype=int), np.array([], dtype=np.float64)
    # sim_matrix: (n_items, n_items)
    user_sim = sim_matrix[user_idxs, :].mean(axis=0)
    user_sim[exclude_item_idxs] = -np.inf
    top = np.argsort(-user_sim)[:top_k]
    valid = user_sim[top] > -np.inf
    top = top[valid]
    return top, user_sim[top].astype(np.float64)


def content_scores_vector(
    user_album_ids: np.ndarray,
    album_id_to_idx: dict,
    n_items: int,
    sim_matrix: np.ndarray,
    exclude_item_idxs: np.ndarray,
) -> np.ndarray:
    """Full score vector (n_items,) for hybrid blending; excluded items = -inf or 0."""
    user_idxs = [album_id_to_idx[a] for a in user_album_ids if a in album_id_to_idx]
    if not user_idxs:
        return np.zeros(n_items, dtype=np.float64)
    user_sim = sim_matrix[user_idxs, :].mean(axis=0)
    user_sim[exclude_item_idxs] = -np.inf
    return user_sim.astype(np.float64)
