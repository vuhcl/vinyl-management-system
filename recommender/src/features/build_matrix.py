"""
Build user-item interaction matrix for ALS (implicit feedback).
"""
import numpy as np
import pandas as pd
from scipy import sparse


def build_user_item_matrix(
    interactions: pd.DataFrame,
    user_col: str = "user_id",
    item_col: str = "album_id",
    weight_col: str = "strength",
    all_user_ids: np.ndarray | None = None,
    all_item_ids: np.ndarray | None = None,
) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    """
    Returns (matrix, user_ids, item_ids) where matrix[i, j] = strength for user_ids[i], item_ids[j].
    If all_user_ids/all_item_ids are provided, use them so matrix shape is fixed (e.g. for eval).
    """
    users = np.unique(interactions[user_col].astype(str).values) if not interactions.empty else np.array([], dtype=str)
    items = np.unique(interactions[item_col].astype(str).values) if not interactions.empty else np.array([], dtype=str)
    if all_user_ids is not None:
        users = np.unique(np.concatenate([users, np.asarray(all_user_ids).astype(str)]))
    if all_item_ids is not None:
        items = np.unique(np.concatenate([items, np.asarray(all_item_ids).astype(str)]))
    user2idx = {u: i for i, u in enumerate(sorted(users))}
    item2idx = {it: j for j, it in enumerate(sorted(items))}

    rows = interactions[user_col].map(user2idx).values
    cols = interactions[item_col].map(item2idx).values
    data = interactions[weight_col].values.astype(np.float64)

    matrix = sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(len(user2idx), len(item2idx)),
    )
    user_ids = np.array(sorted(user2idx.keys()))
    item_ids = np.array(sorted(item2idx.keys()))
    return matrix, user_ids, item_ids


def get_user_item_mappers(
    user_ids: np.ndarray,
    item_ids: np.ndarray,
) -> tuple[dict, dict, dict, dict]:
    """Return user_id2idx, item_id2idx, idx2user_id, idx2item_id."""
    user_id2idx = {u: i for i, u in enumerate(user_ids)}
    item_id2idx = {a: j for j, a in enumerate(item_ids)}
    idx2user_id = {i: u for u, i in user_id2idx.items()}
    idx2item_id = {j: a for a, j in item_id2idx.items()}
    return user_id2idx, item_id2idx, idx2user_id, idx2item_id
