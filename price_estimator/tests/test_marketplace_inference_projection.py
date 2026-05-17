"""Marketplace projections (Redis/serve) + residual-anchor parity helpers."""
from __future__ import annotations

import pytest

from price_estimator.src.storage.marketplace_db import (
    decode_redis_marketplace_cached_payload,
    marketplace_inference_stats_from_row,
    merge_marketplace_client_overlay,
    redis_marketplace_cache_blob_from_row,
)
from price_estimator.src.training.sale_floor_targets import inference_residual_anchor_usd


def test_residual_anchor_prefers_price_suggestions_max_when_ladder_present() -> None:
    row = marketplace_inference_stats_from_row(
        {
            "release_lowest_price": 10.0,
            "price_suggestions_json": (
                '{"Near Mint (NM or M-)": {"value": 55.0}, '
                '"Good (G)": {"value": 22.0}}'
            ),
        },
    )
    m = inference_residual_anchor_usd(
        row,
        nm_grade_key="Near Mint (NM or M-)",
    )
    assert m == pytest.approx(55.0)


def test_inference_anchor_without_ps_ladder_follows_no_sale_history_floor() -> None:
    row = marketplace_inference_stats_from_row({"release_lowest_price": 33.0})
    m = inference_residual_anchor_usd(
        row,
        nm_grade_key="Near Mint (NM or M-)",
    )
    assert m == pytest.approx(33.0)


def test_merge_client_overlay_replaces_floor() -> None:
    base = marketplace_inference_stats_from_row(
        {
            "release_lowest_price": 10.0,
            "num_for_sale": 5,
            "community_want": 1,
        },
    )
    merged = merge_marketplace_client_overlay(
        base,
        {"release_lowest_price": 27.39},
    )
    assert merged["release_lowest_price"] == pytest.approx(27.39)
    assert merged["num_for_sale"] == 5


def test_merge_client_overlay_sale_stats_quartet() -> None:
    base = marketplace_inference_stats_from_row({"release_lowest_price": 44.0})
    merged = merge_marketplace_client_overlay(
        base,
        {
            "sale_stats_average_usd": 108.62,
            "sale_stats_median_usd": 101.37,
            "sale_stats_high_usd": 235.29,
            "sale_stats_low_usd": 30.59,
        },
    )
    assert merged["release_lowest_price"] == pytest.approx(44.0)
    assert merged["sale_stats_average_usd"] == pytest.approx(108.62)
    assert merged["sale_stats_median_usd"] == pytest.approx(101.37)
    assert merged["sale_stats_high_usd"] == pytest.approx(235.29)
    assert merged["sale_stats_low_usd"] == pytest.approx(30.59)


def test_redis_blob_roundtrip() -> None:
    row = marketplace_inference_stats_from_row(
        {
            "release_lowest_price": 12.34,
            "num_for_sale": 60,
            "release_num_for_sale": 55,
            "community_want": 100,
            "community_have": 40,
            "blocked_from_sale": 0,
            "price_suggestions_json": "{}",
            "sale_stats_median_usd": 101.37,
            "sale_stats_low_usd": 30.59,
        },
    )
    blob = redis_marketplace_cache_blob_from_row(row)
    back = decode_redis_marketplace_cached_payload(blob)
    assert back["release_lowest_price"] == pytest.approx(12.34)
    assert back["community_want"] == 100
    assert back["price_suggestions_json"] == "{}"
    assert back["sale_stats_median_usd"] == pytest.approx(101.37)
    assert back["sale_stats_low_usd"] == pytest.approx(30.59)


def test_redis_legacy_two_field_payload() -> None:
    decoded = decode_redis_marketplace_cached_payload(
        {"release_lowest_price": 27.39, "num_for_sale": 60},
    )
    assert decoded["release_lowest_price"] == pytest.approx(27.39)
    assert decoded["num_for_sale"] == 60
    assert decoded.get("community_want") is None
