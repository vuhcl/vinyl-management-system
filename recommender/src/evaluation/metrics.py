"""
Ranking metrics: NDCG@K, MAP@K, Recall@K.
"""
import numpy as np


def dcg_at_k(relevances: np.ndarray, k: int) -> float:
    """DCG@k. relevances: 1-d array of 0/1 (or relevance grades)."""
    relevances = np.asarray(relevances)[:k]
    if relevances.size == 0:
        return 0.0
    gains = 2 ** relevances - 1
    positions = np.arange(1, len(gains) + 1, dtype=np.float64)
    return float(np.sum(gains / np.log2(positions + 1)))


def ndcg_at_k(relevances: np.ndarray, k: int) -> float:
    """NDCG@k. relevances: binary or graded; ideal DCG from sorted relevances."""
    relevances = np.asarray(relevances)
    dcg = dcg_at_k(relevances, k)
    ideal = np.sort(relevances)[::-1]
    idcg = dcg_at_k(ideal, k)
    if idcg <= 0:
        return 0.0
    return dcg / idcg


def ap_at_k(predicted: np.ndarray, relevant: set, k: int) -> float:
    """Average precision at K. predicted = ordered list of item ids (or indices)."""
    predicted = np.asarray(predicted)[:k]
    if len(predicted) == 0 or len(relevant) == 0:
        return 0.0
    hits = np.array([1 if p in relevant else 0 for p in predicted])
    precs = np.cumsum(hits) / np.arange(1, len(hits) + 1, dtype=np.float64)
    return float(np.sum(hits * precs) / min(len(relevant), k))


def recall_at_k(predicted: np.ndarray, relevant: set, k: int) -> float:
    """Recall@K."""
    predicted = set(np.asarray(predicted)[:k])
    if len(relevant) == 0:
        return 0.0
    return len(predicted & relevant) / len(relevant)


def evaluate_ranking(
    predicted: np.ndarray,
    relevant: set,
    k: int = 10,
) -> dict[str, float]:
    """Return dict with ndcg@k, map@k, recall@k. predicted = ordered item ids/indices."""
    pred_list = np.asarray(predicted)[:k]
    relevances = np.array([1 if p in relevant else 0 for p in pred_list])
    return {
        f"ndcg@{k}": ndcg_at_k(relevances, k),
        f"map@{k}": ap_at_k(predicted, relevant, k),
        f"recall@{k}": recall_at_k(predicted, relevant, k),
    }
