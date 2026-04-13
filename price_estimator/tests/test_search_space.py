"""Tests for hyperparameter space sampling."""
from __future__ import annotations

import numpy as np

from price_estimator.src.training.search_space import sample_from_space


def test_sample_choice_list():
    rng = np.random.default_rng(0)
    space = {"a": [1, 2, 3], "b": ["x", "y"]}
    for _ in range(20):
        p = sample_from_space(space, rng)
        assert p["a"] in (1, 2, 3)
        assert p["b"] in ("x", "y")


def test_sample_numeric_range_pair_int():
    rng = np.random.default_rng(1)
    space = {"k": [1, 5]}
    for _ in range(30):
        p = sample_from_space(space, rng)
        assert 1 <= p["k"] <= 5
        assert isinstance(p["k"], int)


def test_sample_numeric_range_pair_float():
    rng = np.random.default_rng(2)
    space = {"lr": [0.01, 0.1]}
    p = sample_from_space(space, rng)
    assert 0.01 <= p["lr"] <= 0.1
