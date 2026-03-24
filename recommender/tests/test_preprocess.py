"""Tests for recommender preprocess / interaction weights."""

from __future__ import annotations

import pandas as pd

from recommender.src.data.preprocess import (
    album_ids_with_only_low_star_ratings,
    apply_weights,
)


def test_album_ids_only_low_stars():
    ratings = pd.DataFrame(
        {
            "user_id": ["a", "b", "c", "d"],
            "album_id": ["x", "x", "y", "z"],
            "rating": [2.0, 1.0, 4.0, 2.5],
        }
    )
    # x: max 2 — only low; y has 4; z only 2.5
    bad = album_ids_with_only_low_star_ratings(ratings, at_least_stars=3.0)
    assert bad == {"x", "z"}


def test_apply_weights_drops_exclusively_low_album_ratings():
    collection = pd.DataFrame({"user_id": ["u1"], "album_id": ["keep_coll"]})
    ratings = pd.DataFrame(
        {
            "user_id": ["a", "b"],
            "album_id": ["only_low", "has_three"],
            "rating": [2.0, 3.0],
        }
    )
    weights = {
        "collection": 1.0,
        "rating_high": 2.5,
        "rating_mid": 1.5,
        "rating_low_2": 0.8,
        "rating_low_1": 0.4,
        "drop_exclusively_low_rating_albums": True,
        "exclusive_low_rating_below": 3.0,
    }
    out = apply_weights(collection, pd.DataFrame(), ratings, weights)
    rating_rows = out[out["source"] == "rating"]
    assert set(rating_rows["album_id"].astype(str)) == {"has_three"}
    assert "only_low" not in set(rating_rows["album_id"].astype(str))


def test_apply_weights_can_disable_low_album_drop():
    ratings = pd.DataFrame(
        {
            "user_id": ["a"],
            "album_id": ["only_low"],
            "rating": [2.0],
        }
    )
    weights = {
        "rating_high": 2.5,
        "rating_mid": 1.5,
        "rating_low_2": 0.8,
        "rating_low_1": 0.4,
        "drop_exclusively_low_rating_albums": False,
    }
    out = apply_weights(pd.DataFrame(), pd.DataFrame(), ratings, weights)
    assert len(out) == 1
    assert out.iloc[0]["album_id"] == "only_low"
