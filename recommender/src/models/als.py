"""
Implicit ALS model for collaborative filtering.
"""
import numpy as np
from scipy import sparse

try:
    from implicit.als import AlternatingLeastSquares
except ImportError:
    AlternatingLeastSquares = None


def train_als(
    user_item: sparse.csr_matrix,
    factors: int = 64,
    regularization: float = 0.01,
    iterations: int = 15,
    alpha: float = 40.0,
    random_state: int = 42,
):
    """
    Train ALS on user-item matrix (items as rows for implicit library).
    Returns fitted model and item_factors, user_factors for scoring.
    """
    if AlternatingLeastSquares is None:
        raise ImportError("Install 'implicit' for ALS: pip install implicit")
    # implicit expects item-user (items rows, users cols)
    item_user = user_item.T.tocsr()
    model = AlternatingLeastSquares(
        factors=factors,
        regularization=regularization,
        iterations=iterations,
        random_state=random_state,
    )
    model.fit(item_user, show_progress=False)
    return model


def predict_als(
    model,
    user_idx: int,
    user_item: sparse.csr_matrix,
    item_ids: np.ndarray,
    exclude_item_idxs: np.ndarray,
    top_k: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get top_k item indices and scores for a user, excluding exclude_item_idxs.
    Returns (item_indices, scores).
    """
    # recommend method returns (item_indices, scores) for the user
    ids, scores = model.recommend(
        user_idx,
        user_item.T.tocsr(),
        N=top_k + len(exclude_item_idxs),
        filter_already_liked_items=True,
    )
    ids = np.asarray(ids)
    scores = np.asarray(scores).astype(np.float64)
    mask = ~np.isin(ids, exclude_item_idxs)
    ids = ids[mask][:top_k]
    scores = scores[mask][:top_k]
    return ids, scores


def als_scores_for_user(
    model,
    user_idx: int,
    user_item: sparse.csr_matrix,
    n_items: int,
    exclude_item_idxs: np.ndarray,
) -> np.ndarray:
    """Return score vector of length n_items for one user (for hybrid blending)."""
    item_user = user_item.T.tocsr()
    # get all item scores
    user_factors = model.user_factors[user_idx]
    scores = item_user.shape[0] * [0.0]
    for i in range(item_user.shape[0]):
        if i in exclude_item_idxs:
            continue
        scores[i] = float(np.dot(user_factors, model.item_factors[i]))
    return np.array(scores, dtype=np.float64)
