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
    # `implicit` expects USER-ITEM confidence matrix for fit:
    # rows=users, cols=items. For implicit feedback, alpha scales confidence.
    user_items = user_item.tocsr().astype(np.float32)
    if alpha != 1.0:
        user_items = user_items * float(alpha)
    model = AlternatingLeastSquares(
        factors=factors,
        regularization=regularization,
        iterations=iterations,
        random_state=random_state,
    )
    model.fit(user_items, show_progress=False)
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
        # `implicit` expects a USER-ITEM matrix for `recommend`:
        # rows=users, cols=items.
        user_item.tocsr(),
        N=top_k + len(exclude_item_idxs),
        # We handle already-seen items ourselves (via exclude_item_idxs),
        # which avoids strict shape requirements inside `implicit` when
        # calling recommend with a scalar userid.
        filter_already_liked_items=False,
    )
    ids = np.asarray(ids)
    scores = np.asarray(scores).astype(np.float64)
    mask = ~np.isin(ids, exclude_item_idxs)
    ids = ids[mask][:top_k]
    scores = scores[mask][:top_k]
    return ids, scores


def predict_als_in_candidates(
    model,
    user_idx: int,
    user_item: sparse.csr_matrix,
    exclude_item_idxs: np.ndarray,
    candidate_item_idxs: np.ndarray,
    top_k: int = 10,
    *,
    score_bonus: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Score only items in ``candidate_item_idxs`` via latent dot products.

    Faster than full-catalog ``recommend`` when |candidates| << n_items.

    If ``score_bonus`` is set, it must have shape ``(len(candidate_item_idxs),)``
    and is added element-wise to ALS dot scores before ranking (e.g. content).
    """
    if candidate_item_idxs.size == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64)
    exclude_set = set(int(x) for x in np.asarray(exclude_item_idxs).ravel())
    uf = np.asarray(model.user_factors[user_idx], dtype=np.float64)
    inds = np.asarray(candidate_item_idxs, dtype=np.int64)
    IF = np.asarray(model.item_factors[inds], dtype=np.float64)
    scores = IF @ uf
    if score_bonus is not None:
        sb = np.asarray(score_bonus, dtype=np.float64).ravel()
        if sb.shape[0] != inds.shape[0]:
            raise ValueError(
                "score_bonus length must match candidate_item_idxs"
            )
        scores = scores + sb
    order = np.argsort(-scores)
    picked: list[int] = []
    picked_scores: list[float] = []
    for j in order:
        idx = int(inds[j])
        if idx in exclude_set:
            continue
        picked.append(idx)
        picked_scores.append(float(scores[j]))
        if len(picked) >= top_k:
            break
    return np.asarray(picked, dtype=np.int64), np.asarray(
        picked_scores, dtype=np.float64
    )


def als_scores_for_user(
    model,
    user_idx: int,
    user_item: sparse.csr_matrix,
    n_items: int,
    exclude_item_idxs: np.ndarray,
) -> np.ndarray:
    """Score vector for one user (for hybrid blending)."""
    item_user = user_item.T.tocsr()
    # get all item scores
    user_factors = model.user_factors[user_idx]
    scores = item_user.shape[0] * [0.0]
    for i in range(item_user.shape[0]):
        if i in exclude_item_idxs:
            continue
        scores[i] = float(np.dot(user_factors, model.item_factors[i]))
    return np.array(scores, dtype=np.float64)
