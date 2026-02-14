"""
Hybrid recommender: blend CF (ALS) and content-based scores.
"""
import numpy as np


def blend_scores(
    cf_scores: np.ndarray,
    content_scores: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """
    final_score = alpha * CF_score + (1 - alpha) * content_score.
    Normalize each score vector to [0,1] before blending if needed.
    """
    def norm(s: np.ndarray) -> np.ndarray:
        s = np.nan_to_num(s, nan=-np.inf, posinf=0, neginf=-np.inf)
        m, M = s.max(), s.min()
        if m <= M:
            return np.zeros_like(s)
        return (s - M) / (m - M)

    cf_n = norm(cf_scores.copy())
    content_n = norm(content_scores.copy())
    return alpha * cf_n + (1 - alpha) * content_n


def rank_hybrid(
    cf_scores: np.ndarray,
    content_scores: np.ndarray,
    alpha: float,
    exclude_mask: np.ndarray,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Blend and return top_k (item_indices, scores). exclude_mask True = exclude.
    """
    blended = blend_scores(cf_scores, content_scores, alpha)
    blended[exclude_mask] = -np.inf
    top = np.argsort(-blended)[:top_k]
    valid = blended[top] > -np.inf
    return top[valid], blended[top[valid]]
