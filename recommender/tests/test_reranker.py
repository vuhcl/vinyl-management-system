from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from recommender.src.evaluation.evaluate import leave_one_out_split
from recommender.src.features.build_matrix import (
    build_user_item_matrix,
    get_user_item_mappers,
)
from recommender.src.models.als import train_als
from recommender.src.models.reranker import (
    ReRankerConfig,
    build_reranker_training_frame,
    rerank_candidates_for_user,
    train_reranker,
)
from recommender.src.retrieval.candidates import build_retrieval_metadata


def _toy_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    interactions = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1", "u2", "u2", "u3", "u3"],
            "album_id": ["a1", "a2", "a3", "a2", "a4", "a1", "a5"],
            "strength": [1.0, 1.2, 1.5, 1.1, 1.0, 0.9, 1.3],
        }
    )
    albums = pd.DataFrame(
        [
            {
                "album_id": "a1",
                "artist": "x",
                "genre": "rock",
                "avg_rating": 4.0,
                "year": 1999,
            },
            {
                "album_id": "a2",
                "artist": "x",
                "genre": "rock",
                "avg_rating": 4.1,
                "year": 2001,
            },
            {
                "album_id": "a3",
                "artist": "y",
                "genre": "electronic",
                "avg_rating": 3.9,
                "year": 2005,
            },
            {
                "album_id": "a4",
                "artist": "z",
                "genre": "jazz",
                "avg_rating": 3.6,
                "year": 2010,
            },
            {
                "album_id": "a5",
                "artist": "x",
                "genre": "rock",
                "avg_rating": 4.2,
                "year": 2003,
            },
        ]
    )
    return interactions, albums


def test_build_train_and_rerank_smoke() -> None:
    interactions, albums = _toy_data()
    train_i, test_i = leave_one_out_split(interactions, random_state=42)
    all_item_ids = np.unique(
        np.concatenate(
            [
                train_i["album_id"].astype(str).values,
                test_i["album_id"].astype(str).values,
            ]
        )
    )
    matrix, user_ids, item_ids = build_user_item_matrix(
        train_i, weight_col="strength", all_item_ids=all_item_ids
    )
    user_id2idx, item_id2idx, _, _ = get_user_item_mappers(user_ids, item_ids)
    model = train_als(
        matrix,
        factors=8,
        regularization=0.05,
        iterations=2,
        alpha=10.0,
        random_state=42,
    )
    meta = build_retrieval_metadata(albums, train_i)
    rr_cfg = ReRankerConfig(
        enabled=True,
        model_type="linear",
        candidate_top_n=20,
        train_sample_n=1000,
        negative_sampling=10,
        random_state=42,
    )
    rr_df, rr_stats = build_reranker_training_frame(
        model=model,
        item_ids=item_ids,
        user_id2idx=user_id2idx,
        item_id2idx=item_id2idx,
        train_interactions=train_i,
        test_interactions=test_i,
        meta=meta,
        rr_cfg=rr_cfg,
    )
    assert len(rr_df) > 0
    assert rr_stats["rr_train_rows"] > 0
    bundle = train_reranker(rr_df, rr_cfg)
    if bundle is None:
        pytest.skip("Toy split produced single-class reranker frame.")

    uid = str(next(iter(user_id2idx.keys())))
    user_idx = user_id2idx[uid]
    train_items = set(
        train_i[train_i["user_id"] == uid]["album_id"].astype(str).tolist()
    )
    cand = np.arange(len(item_ids), dtype=np.int64)
    exclude = np.array(
        [item_id2idx[a] for a in train_items if a in item_id2idx],
        dtype=np.int64,
    )
    item_train_counts = {
        str(a): int(v)
        for a, v in train_i.groupby("album_id", sort=False).size().items()
    }
    rank_idx, scores = rerank_candidates_for_user(
        bundle=bundle,
        model=model,
        user_idx=user_idx,
        train_albums=train_items,
        candidate_item_idxs=cand,
        exclude_idxs=exclude,
        item_ids=item_ids,
        meta=meta,
        item_train_counts=item_train_counts,
        top_k=5,
    )
    assert rank_idx.ndim == 1
    assert scores.ndim == 1
    assert len(rank_idx) == len(scores)
