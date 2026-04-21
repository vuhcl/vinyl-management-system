"""
Smoke test: run_evaluation on a tiny synthetic dataset.
Verifies that the full-catalog ALS path (and optional reranker path) completes
and produces expected metric keys.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from recommender.src.evaluation.evaluate import run_evaluation


def _toy_interactions() -> tuple[pd.DataFrame, pd.DataFrame]:
    interactions = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1", "u2", "u2", "u2", "u3", "u3", "u3"],
            "album_id": ["a1", "a2", "a3", "a2", "a4", "a5", "a1", "a5", "a6"],
            "strength": [1.0, 1.2, 1.5, 1.1, 1.0, 0.8, 0.9, 1.3, 1.1],
        }
    )
    albums = pd.DataFrame(
        [
            {"album_id": "a1", "artist": "x", "genre": "rock", "avg_rating": 4.0, "year": 1999},
            {"album_id": "a2", "artist": "x", "genre": "rock", "avg_rating": 4.1, "year": 2001},
            {"album_id": "a3", "artist": "y", "genre": "electronic", "avg_rating": 3.9, "year": 2005},
            {"album_id": "a4", "artist": "z", "genre": "jazz", "avg_rating": 3.6, "year": 2010},
            {"album_id": "a5", "artist": "x", "genre": "rock", "avg_rating": 4.2, "year": 2003},
            {"album_id": "a6", "artist": "w", "genre": "pop", "avg_rating": 3.5, "year": 2015},
        ]
    )
    return interactions, albums


_ALS_TINY = {
    "factors": 8,
    "regularization": 0.05,
    "iterations": 2,
    "alpha": 10.0,
}


def test_run_evaluation_smoke() -> None:
    """Full-catalog ALS evaluation completes and returns ndcg@5."""
    interactions, _ = _toy_interactions()
    metrics = run_evaluation(
        interactions,
        interactions,
        _ALS_TINY,
        k=5,
        random_state=42,
    )
    assert f"ndcg@5" in metrics
    assert isinstance(metrics[f"ndcg@5"], float)


def test_run_evaluation_with_albums_smoke() -> None:
    """Evaluation with album metadata provided still completes (reranker disabled)."""
    interactions, albums = _toy_interactions()
    metrics = run_evaluation(
        interactions,
        interactions,
        _ALS_TINY,
        k=5,
        random_state=42,
        albums=albums,
        reranker={"enabled": False},
    )
    assert "ndcg@5" in metrics


def test_run_evaluation_reranker_smoke() -> None:
    """Evaluation with reranker enabled over a tiny dataset completes without error."""
    interactions, albums = _toy_interactions()
    rr_cfg = {
        "enabled": True,
        "model_type": "linear",
        "candidate_top_n": 10,
        "train_sample_n": 1000,
        "negative_sampling": 5,
        "random_state": 42,
    }
    metrics = run_evaluation(
        interactions,
        interactions,
        _ALS_TINY,
        k=5,
        random_state=42,
        albums=albums,
        reranker=rr_cfg,
    )
    assert "ndcg@5" in metrics
