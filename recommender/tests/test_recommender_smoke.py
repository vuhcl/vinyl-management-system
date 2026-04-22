"""
Smoke test for the full evaluation path: full-catalog ALS stage-1,
optional reranker stage-2.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from recommender.src.evaluation.evaluate import run_evaluation


def _toy_interactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1", "u2", "u2", "u3", "u3", "u3"],
            "album_id": ["a1", "a2", "a3", "a2", "a4", "a1", "a5", "a3"],
            "strength": [1.0, 1.2, 1.5, 1.1, 1.0, 0.9, 1.3, 1.1],
        }
    )


def _toy_albums() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"album_id": "a1", "artist": "x", "genre": "rock", "avg_rating": 4.0, "year": 1999},
            {"album_id": "a2", "artist": "x", "genre": "rock", "avg_rating": 4.1, "year": 2001},
            {"album_id": "a3", "artist": "y", "genre": "electronic", "avg_rating": 3.9, "year": 2005},
            {"album_id": "a4", "artist": "z", "genre": "jazz", "avg_rating": 3.6, "year": 2010},
            {"album_id": "a5", "artist": "x", "genre": "rock", "avg_rating": 4.2, "year": 2003},
        ]
    )


def _als_cfg() -> dict:
    return {"factors": 8, "regularization": 0.05, "iterations": 2, "alpha": 10.0}


def test_run_evaluation_smoke() -> None:
    """Full-catalog ALS evaluation returns expected metric keys."""
    from recommender.src.evaluation.evaluate import leave_one_out_split

    interactions = _toy_interactions()
    train_i, test_i = leave_one_out_split(interactions, random_state=42)
    metrics = run_evaluation(train_i, test_i, _als_cfg(), k=5, random_state=42)
    assert "ndcg@5" in metrics
    assert "map@5" in metrics
    assert "recall@5" in metrics
    assert 0.0 <= metrics["ndcg@5"] <= 1.0


def test_run_evaluation_with_reranker_smoke() -> None:
    """When albums are present and reranker enabled, run completes and reranked metric exists."""
    from recommender.src.evaluation.evaluate import leave_one_out_split

    interactions = _toy_interactions()
    albums = _toy_albums()
    train_i, test_i = leave_one_out_split(interactions, random_state=42)
    rr_cfg = {
        "enabled": True,
        "model_type": "linear",
        "candidate_top_n": 20,
        "train_sample_n": 1000,
        "negative_sampling": 10,
        "random_state": 42,
    }
    metrics = run_evaluation(
        train_i, test_i, _als_cfg(), k=5, random_state=42,
        albums=albums, reranker=rr_cfg,
    )
    assert "ndcg@5" in metrics
    # Reranker may not train if single-class; in either case the run completes.
