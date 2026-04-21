"""
Offline evaluation: train/test split, run NDCG@K, MAP@K, Recall@K.

Stage-1 is always full-catalog ALS (predict_als over all matrix items).
The optional reranker takes ALS top-N candidates and reorders them using
content-based features derived from RetrievalMetadata.
"""

import numpy as np
import pandas as pd
from scipy import sparse

from ..features.build_matrix import build_user_item_matrix, get_user_item_mappers
from ..models.als import predict_als, train_als
from ..models.reranker import (
    ReRankerBundle,
    build_reranker_training_frame,
    rerank_candidates_for_user,
    reranker_config_from_dict,
    train_reranker,
)
from ..retrieval.candidates import (
    RetrievalMetadata,
    build_retrieval_metadata,
)
from .metrics import ap_at_k, ndcg_at_k, recall_at_k


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


def evaluate_pretrained_als(
    model,
    matrix: sparse.csr_matrix,
    user_id2idx: dict[str, int],
    item_id2idx: dict[str, int],
    item_ids: np.ndarray,
    train_interactions: pd.DataFrame,
    test_interactions: pd.DataFrame,
    k: int,
    *,
    meta: RetrievalMetadata | None = None,
    reranker_bundle: ReRankerBundle | None = None,
) -> dict[str, float]:
    """
    Rank with a fitted ALS model without re-training.

    Stage-1 recall is always full-catalog ALS (``predict_als``).
    When ``reranker_bundle`` and ``meta`` are both present, re-ranks
    ALS top-N candidates using the trained reranker.
    """
    test_by_user = test_interactions.groupby("user_id")["album_id"].apply(set).to_dict()
    _ti = train_interactions.assign(
        user_id=train_interactions["user_id"].astype(str),
        album_id=train_interactions["album_id"].astype(str),
    )
    train_by_user: dict[str, set[str]] = (
        _ti.groupby("user_id")["album_id"].apply(set).to_dict()
    )
    item_train_counts: dict[str, int] = {
        str(a): int(v)
        for a, v in train_interactions.groupby("album_id", sort=False)
        .size()
        .items()
    }
    n_items = len(item_ids)
    all_item_idxs = np.arange(n_items, dtype=np.int64)

    ndcgs, maps, recalls = [], [], []
    strat_pop_head: list[float] = []
    strat_pop_tail: list[float] = []
    pops_for_median: list[int] = []

    for uid, relevant in test_by_user.items():
        if uid not in user_id2idx:
            continue
        user_idx = user_id2idx[uid]
        train_items = train_by_user.get(uid, set())
        exclude_idxs = np.array(
            [item_id2idx[a] for a in train_items if a in item_id2idx], dtype=int
        )
        relevant_idx = {item_id2idx[a] for a in relevant if a in item_id2idx}
        if not relevant_idx:
            continue
        rel_album = str(next(iter(relevant)))
        pop_obs = item_train_counts.get(rel_album, 0)
        pops_for_median.append(int(pop_obs))

        pred_idxs: np.ndarray
        if reranker_bundle is not None and meta is not None:
            train_albums = {str(x) for x in train_items}
            pred_idxs, _ = rerank_candidates_for_user(
                bundle=reranker_bundle,
                model=model,
                user_idx=user_idx,
                train_albums=train_albums,
                candidate_item_idxs=all_item_idxs,
                exclude_idxs=exclude_idxs,
                item_ids=item_ids,
                meta=meta,
                item_train_counts=item_train_counts,
                top_k=k,
            )
        else:
            pred_idxs, _ = predict_als(
                model, user_idx, matrix, item_ids, exclude_idxs, top_k=k
            )

        rel_arr = np.array([1 if i in relevant_idx else 0 for i in pred_idxs])
        n = ndcg_at_k(rel_arr, k)
        ndcgs.append(n)
        maps.append(ap_at_k(pred_idxs, relevant_idx, k))
        recalls.append(recall_at_k(pred_idxs, relevant_idx, k))

    out: dict[str, float] = {
        f"ndcg@{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
        f"map@{k}": float(np.mean(maps)) if maps else 0.0,
        f"recall@{k}": float(np.mean(recalls)) if recalls else 0.0,
    }
    if pops_for_median and ndcgs:
        med = float(np.median(pops_for_median))
        for n_dcg, p in zip(ndcgs, pops_for_median):
            if float(p) >= med:
                strat_pop_head.append(n_dcg)
            else:
                strat_pop_tail.append(n_dcg)
        if strat_pop_head:
            out[f"ndcg@{k}_pop_head"] = float(np.mean(strat_pop_head))
            out[f"n_users_pop_head"] = float(len(strat_pop_head))
        if strat_pop_tail:
            out[f"ndcg@{k}_pop_tail"] = float(np.mean(strat_pop_tail))
            out[f"n_users_pop_tail"] = float(len(strat_pop_tail))
        out["heldout_item_train_count_median"] = med
    return out


def run_evaluation(
    train_interactions: pd.DataFrame,
    test_interactions: pd.DataFrame,
    als_config: dict,
    k: int = 10,
    random_state: int = 42,
    albums: pd.DataFrame | None = None,
    reranker: dict | None = None,
) -> dict[str, float]:
    """
    Build matrix from train, fit ALS, evaluate top-k predictions.

    Returns aggregate NDCG@k, MAP@k, Recall@k averaged over users with test items.
    Stage-1 recall is always full-catalog ALS.

    When ``reranker`` is ``{"enabled": True, ...}`` and ``albums`` is non-empty,
    trains a second-stage reranker over ALS top-N and reports reranked metrics
    alongside ALS-only metrics.
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
    user_id2idx, item_id2idx, _, _ = get_user_item_mappers(user_ids, item_ids)
    model = train_als(
        matrix,
        factors=als_config.get("factors", 64),
        regularization=als_config.get("regularization", 0.01),
        iterations=als_config.get("iterations", 15),
        alpha=als_config.get("alpha", 40.0),
        random_state=random_state,
    )

    base_out = evaluate_pretrained_als(
        model, matrix, user_id2idx, item_id2idx, item_ids,
        train_interactions, test_interactions, k,
    )
    out = dict(base_out)

    rr_cfg = reranker_config_from_dict(reranker)
    meta: RetrievalMetadata | None = None
    if rr_cfg.enabled and albums is not None and not albums.empty:
        meta = build_retrieval_metadata(albums, train_interactions)
        if not meta.valid_album_ids:
            meta = None

    can_rerank = bool(rr_cfg.enabled and meta is not None)
    if can_rerank:
        rr_df, rr_stats = build_reranker_training_frame(
            model=model,
            item_ids=item_ids,
            user_id2idx=user_id2idx,
            item_id2idx=item_id2idx,
            train_interactions=train_interactions,
            test_interactions=test_interactions,
            meta=meta,
            rr_cfg=rr_cfg,
        )
        rr_bundle = train_reranker(rr_df, rr_cfg)
        out.update(rr_stats)
        if rr_bundle is not None:
            reranked = evaluate_pretrained_als(
                model, matrix, user_id2idx, item_id2idx, item_ids,
                train_interactions, test_interactions, k,
                meta=meta, reranker_bundle=rr_bundle,
            )
            for key in (f"ndcg@{k}", f"map@{k}", f"recall@{k}"):
                out[f"als_only_{key}"] = float(base_out.get(key, 0.0))
                out[key] = float(reranked.get(key, 0.0))
            out["reranker_enabled"] = 1.0
            out["reranker_model_type"] = 1.0 if rr_bundle.model_type == "linear" else 2.0
            out["reranker_train_rows"] = float(rr_bundle.train_rows)
            out["reranker_positive_rate"] = float(rr_bundle.positive_rate)
            for k2, v2 in reranked.items():
                if k2 in (f"ndcg@{k}", f"map@{k}", f"recall@{k}"):
                    continue
                out[k2] = float(v2)
    return out
