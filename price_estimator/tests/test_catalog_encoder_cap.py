"""Tests for frequency-capped primary_artist_id / primary_label_id encoders."""
from __future__ import annotations

from price_estimator.src.training.train_vinyliq import (
    _auto_top_k_id_encoder,
    _fit_frequency_capped_id_encoder,
)


def test_auto_top_k_small_vocab_unchanged():
    assert _auto_top_k_id_encoder(10_000, 100) == 100
    assert _auto_top_k_id_encoder(10_000, 500) == 500


def test_auto_top_k_large_vocab_capped():
    k = _auto_top_k_id_encoder(100_000, 50_000)
    assert k == 3000


def test_auto_top_k_mid_size():
    # n//25 = 400 < 500 → k = 500 before min with n_unique
    k = _auto_top_k_id_encoder(10_000, 10_000)
    assert k == 500


def test_fit_frequency_capped_assigns_one_based_ranks():
    ids = ["a", "b", "a", "a", "c", "b"]
    enc = _fit_frequency_capped_id_encoder(ids, max_k=2)
    assert enc["a"] == 1.0
    assert enc["b"] == 2.0
    assert "c" not in enc


def test_fit_frequency_capped_empty():
    assert _fit_frequency_capped_id_encoder([], 10) == {}
    assert _fit_frequency_capped_id_encoder(["", ""], 10) == {}
