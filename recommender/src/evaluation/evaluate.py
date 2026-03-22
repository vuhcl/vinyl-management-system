"""
Offline evaluation: train/test split, run NDCG@K, MAP@K, Recall@K.
"""

import warnings

import numpy as np
import pandas as pd
from scipy import sparse

from ..features.build_matrix import build_user_item_matrix, get_user_item_mappers
from ..models.als import predict_als, predict_als_in_candidates, train_als
from ..retrieval.candidates import (
    CandidateRetrievalConfig,
    build_retrieval_metadata,
    candidate_item_indices_for_user,
    retrieval_config_from_dict,
)
from .metrics import ap_at_k, ndcg_at_k, recall_at_k


def _maybe_warn_candidate_hit_rate(
    retrieval: dict | None,
    metrics: dict[str, float],
) -> None:
    """Warn if candidate_relevant_hit_rate is below configured minimum."""
    if not retrieval or retrieval.get("min_candidate_relevant_hit_rate") is None:
        return
    key = "candidate_relevant_hit_rate"
    if key not in metrics:
        return
    min_rate = float(retrieval["min_candidate_relevant_hit_rate"])
    actual = float(metrics[key])
    if actual < min_rate:
        metrics["candidate_retrieval_hit_rate_below_min"] = 1.0
        msg = (
            f"{key}={actual:.4f} is below "
            f"min_candidate_relevant_hit_rate={min_rate:.4f}"
        )
        if retrieval.get("fail_on_low_candidate_hit_rate"):
            raise ValueError(msg)
        warnings.warn(msg, UserWarning, stacklevel=2)


def leave_one_out_split(
    interactions: pd.DataFrame,
    user_col: str = "user_id",
    item_col: str = "album_id",
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """For each user, hold out one random interaction for test; rest for train."""
    rng = np.random.default_rng(random_state)
    train_list, test_list = [], []
    for uid, g in interactions.groupby(user_col):
        g = g.reset_index(drop=True)
        if len(g) < 2:
            train_list.append(g)
            continue
        idx = rng.integers(0, len(g))
        test_list.append(g.iloc[[idx]])
        train_list.append(g.drop(index=g.index[idx]))
    train = pd.concat(train_list, ignore_index=True)
    test = pd.concat(test_list, ignore_index=True)
    return train, test


def run_evaluation(
    train_interactions: pd.DataFrame,
    test_interactions: pd.DataFrame,
    als_config: dict,
    k: int = 10,
    random_state: int = 42,
    albums: pd.DataFrame | None = None,
    retrieval: dict | None = None,
) -> dict[str, float]:
    """
    Build matrix from train, fit ALS, for each test user predict top-k and compute metrics.
    Returns aggregate NDCG@k, MAP@k, Recall@k (averaged over users with test items).

    If ``retrieval`` is provided with ``{"enabled": True}`` and ``albums`` is non-empty,
    stage-1 builds a candidate pool (genres ∪ same-artist, quality floors) and stage-2
    scores only those items with ALS (faster when the pool is small).
    """
    all_item_ids = np.unique(
        np.concatenate(
            [
                train_interactions["album_id"].astype(str).values,
                test_interactions["album_id"].astype(str).values,
            ]
        )
    )
    matrix, user_ids, item_ids = build_user_item_matrix(
        train_interactions, weight_col="strength", all_item_ids=all_item_ids
    )
    user_id2idx, item_id2idx, _, idx2item_id = get_user_item_mappers(user_ids, item_ids)
    model = train_als(
        matrix,
        factors=als_config.get("factors", 64),
        regularization=als_config.get("regularization", 0.01),
        iterations=als_config.get("iterations", 15),
        alpha=als_config.get("alpha", 40.0),
        random_state=random_state,
    )
    use_two_stage = bool(
        retrieval
        and retrieval.get("enabled")
        and albums is not None
        and not albums.empty
    )
    meta = None
    rcfg = CandidateRetrievalConfig()
    if use_two_stage:
        rcfg = retrieval_config_from_dict(retrieval or {})
        meta = build_retrieval_metadata(albums, train_interactions)
        if not meta.valid_album_ids:
            use_two_stage = False

    test_by_user = test_interactions.groupby("user_id")["album_id"].apply(set).to_dict()
    ndcgs, maps, recalls = [], [], []
    rel_hits: list[int] = []
    two_stage_used: list[int] = []
    for uid, relevant in test_by_user.items():
        if uid not in user_id2idx:
            continue
        user_idx = user_id2idx[uid]
        train_items = set(
            train_interactions[train_interactions["user_id"] == uid]["album_id"].astype(
                str
            )
        )
        exclude_idxs = np.array(
            [item_id2idx[a] for a in train_items if a in item_id2idx], dtype=int
        )
        relevant_idx = {item_id2idx[a] for a in relevant if a in item_id2idx}
        if not relevant_idx:
            continue
        pred_idxs: np.ndarray
        if use_two_stage and meta is not None and albums is not None:
            train_albums = {
                str(x) for x in train_items if str(x) in meta.valid_album_ids
            }
            cand = candidate_item_indices_for_user(
                train_albums,
                meta,
                item_id2idx,
                rcfg,
            )
            cand_set = set(int(x) for x in cand.tolist())
            rel_hits.append(1 if relevant_idx & cand_set else 0)
            if cand.size > 0:
                two_stage_used.append(1)
                pred_idxs, _ = predict_als_in_candidates(
                    model,
                    user_idx,
                    matrix,
                    exclude_idxs,
                    cand,
                    top_k=k,
                )
            else:
                two_stage_used.append(0)
                pred_idxs, _ = predict_als(
                    model, user_idx, matrix, item_ids, exclude_idxs, top_k=k
                )
        else:
            pred_idxs, _ = predict_als(
                model, user_idx, matrix, item_ids, exclude_idxs, top_k=k
            )
        rel_arr = np.array([1 if i in relevant_idx else 0 for i in pred_idxs])
        ndcgs.append(ndcg_at_k(rel_arr, k))
        maps.append(ap_at_k(pred_idxs, relevant_idx, k))
        recalls.append(recall_at_k(pred_idxs, relevant_idx, k))
    out: dict[str, float] = {
        f"ndcg@{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
        f"map@{k}": float(np.mean(maps)) if maps else 0.0,
        f"recall@{k}": float(np.mean(recalls)) if recalls else 0.0,
    }
    if use_two_stage and rel_hits:
        out["candidate_relevant_hit_rate"] = float(np.mean(rel_hits))
        out["two_stage_candidate_nonempty_rate"] = float(np.mean(two_stage_used))
    _maybe_warn_candidate_hit_rate(retrieval, out)
    return out
