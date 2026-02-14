"""
Offline evaluation: train/test split, run NDCG@K, MAP@K, Recall@K.
"""
import numpy as np
import pandas as pd
from scipy import sparse

from ...features.build_matrix import build_user_item_matrix, get_user_item_mappers
from ...models.als import train_als, predict_als
from .metrics import ndcg_at_k, ap_at_k, recall_at_k


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
) -> dict[str, float]:
    """
    Build matrix from train, fit ALS, for each test user predict top-k and compute metrics.
    Returns aggregate NDCG@k, MAP@k, Recall@k (averaged over users with test items).
    """
    all_item_ids = np.unique(np.concatenate([
        train_interactions["album_id"].astype(str).values,
        test_interactions["album_id"].astype(str).values,
    ]))
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
    test_by_user = test_interactions.groupby("user_id")["album_id"].apply(set).to_dict()
    ndcgs, maps, recalls = [], [], []
    for uid, relevant in test_by_user.items():
        if uid not in user_id2idx:
            continue
        user_idx = user_id2idx[uid]
        train_items = set(
            train_interactions[train_interactions["user_id"] == uid]["album_id"].astype(str)
        )
        exclude_idxs = np.array([item_id2idx[a] for a in train_items if a in item_id2idx], dtype=int)
        relevant_idx = {item_id2idx[a] for a in relevant if a in item_id2idx}
        if not relevant_idx:
            continue
        pred_idxs, _ = predict_als(model, user_idx, matrix, item_ids, exclude_idxs, top_k=k)
        pred_set = set(pred_idxs)
        pred_ids = [idx2item_id[i] for i in pred_idxs]
        rel_arr = np.array([1 if i in relevant_idx else 0 for i in pred_idxs])
        ndcgs.append(ndcg_at_k(rel_arr, k))
        maps.append(ap_at_k(pred_idxs, relevant_idx, k))
        recalls.append(recall_at_k(pred_idxs, relevant_idx, k))
    return {
        f"ndcg@{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
        f"map@{k}": float(np.mean(maps)) if maps else 0.0,
        f"recall@{k}": float(np.mean(recalls)) if recalls else 0.0,
    }
