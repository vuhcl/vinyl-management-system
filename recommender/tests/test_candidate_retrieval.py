"""Tests for two-stage metadata candidate retrieval."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from recommender.src.evaluation.evaluate import (
    leave_one_out_split,
    run_evaluation,
)
from recommender.src.retrieval.candidates import (
    CandidateRetrievalConfig,
    build_retrieval_metadata,
    candidate_album_ids_for_user,
)


def test_candidate_pool_includes_genre_and_artist_neighbors() -> None:
    albums = pd.DataFrame(
        [
            {
                "album_id": "1",
                "artist": "A",
                "genre": "rock",
                "avg_rating": 4.0,
            },
            {
                "album_id": "2",
                "artist": "B",
                "genre": "rock",
                "avg_rating": 4.0,
            },
            {
                "album_id": "3",
                "artist": "A",
                "genre": "jazz",
                "avg_rating": 4.0,
            },
            {
                "album_id": "4",
                "artist": "Z",
                "genre": "classical",
                "avg_rating": 4.0,
            },
        ]
    )
    train = pd.DataFrame(
        {
            "user_id": ["u"] * 3,
            "album_id": ["1", "1", "2"],
            "strength": [1.0, 1.0, 1.0],
        }
    )
    meta = build_retrieval_metadata(albums, train)
    cfg = CandidateRetrievalConfig(
        min_avg_rating=0.0,
        min_train_count=1,
        max_candidates=100,
    )
    pool = candidate_album_ids_for_user({"1"}, meta, cfg)
    assert "1" in pool
    assert "2" in pool  # same genre as 1
    assert "3" in pool  # same artist as 1
    assert "4" not in pool


def test_run_evaluation_two_stage_smoke() -> None:
    """Tiny end-to-end: two-stage path runs without error."""
    rng = np.random.default_rng(0)
    users = [f"u{i}" for i in range(8)]
    items = [f"a{j}" for j in range(20)]
    rows = []
    for u in users:
        picks = rng.choice(items, size=5, replace=False)
        for it in picks:
            rows.append((u, it, 1.0))
    interactions = pd.DataFrame(
        rows, columns=["user_id", "album_id", "strength"]
    )
    albums = pd.DataFrame(
        [
            {
                "album_id": f"a{j}",
                "artist": f"artist{j % 3}",
                "genre": "rock" if j % 2 == 0 else "pop",
                "avg_rating": 3.5,
            }
            for j in range(20)
        ]
    )
    train_i, test_i = leave_one_out_split(interactions, random_state=42)
    metrics = run_evaluation(
        train_i,
        test_i,
        {
            "factors": 8,
            "regularization": 0.1,
            "iterations": 2,
            "alpha": 10.0,
        },
        k=5,
        random_state=42,
        albums=albums,
        retrieval={"enabled": True, "max_candidates": 50},
    )
    assert "ndcg@5" in metrics
    assert "candidate_relevant_hit_rate" in metrics


def test_min_candidate_hit_rate_warns() -> None:
    """Very tight pool triggers below-min flag when threshold is high."""
    interactions = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1"],
            "album_id": ["a", "b", "c"],
            "strength": [1.0, 1.0, 1.0],
        }
    )
    albums = pd.DataFrame(
        [
            {"album_id": "a", "artist": "x", "genre": "g1", "avg_rating": 4.0},
            {"album_id": "b", "artist": "y", "genre": "g2", "avg_rating": 4.0},
            {"album_id": "c", "artist": "z", "genre": "g3", "avg_rating": 4.0},
        ]
    )
    train_i, test_i = leave_one_out_split(interactions, random_state=0)
    retrieval = {
        "enabled": True,
        "max_candidates": 500,
        "min_candidate_relevant_hit_rate": 0.99,
    }
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        metrics = run_evaluation(
            train_i,
            test_i,
            {"factors": 8, "regularization": 0.1, "iterations": 2, "alpha": 10.0},
            k=3,
            random_state=0,
            albums=albums,
            retrieval=retrieval,
        )
    assert metrics.get("candidate_retrieval_hit_rate_below_min") == 1.0
    assert w


def test_min_candidate_hit_rate_fail() -> None:
    interactions = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1"],
            "album_id": ["a", "b", "c"],
            "strength": [1.0, 1.0, 1.0],
        }
    )
    albums = pd.DataFrame(
        [
            {"album_id": "a", "artist": "x", "genre": "g1", "avg_rating": 4.0},
            {"album_id": "b", "artist": "y", "genre": "g2", "avg_rating": 4.0},
            {"album_id": "c", "artist": "z", "genre": "g3", "avg_rating": 4.0},
        ]
    )
    train_i, test_i = leave_one_out_split(interactions, random_state=0)
    retrieval = {
        "enabled": True,
        "min_candidate_relevant_hit_rate": 0.99,
        "fail_on_low_candidate_hit_rate": True,
    }
    with pytest.raises(ValueError, match="candidate_relevant_hit_rate"):
        run_evaluation(
            train_i,
            test_i,
            {"factors": 8, "regularization": 0.1, "iterations": 2, "alpha": 10.0},
            k=3,
            random_state=0,
            albums=albums,
            retrieval=retrieval,
        )
