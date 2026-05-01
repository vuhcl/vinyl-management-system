"""Unit tests for ``compute_marketplace_upsert_values`` merge/coalesce logic."""

from __future__ import annotations

import json

from price_estimator.src.storage.marketplace_db import compute_marketplace_upsert_values


def test_insert_empty_prev_basic_stats():
    comp = compute_marketplace_upsert_values(
        "123",
        {"num_for_sale": 7, "blocked_from_sale": 0},
        {},
        fetched_at="2025-01-01T00:00:00+00:00",
    )
    assert comp["release_id"] == "123"
    assert comp["fetched_at"] == "2025-01-01T00:00:00+00:00"
    assert comp["num_for_sale"] == 7
    assert comp["blocked_from_sale"] == 0
    assert '"num_for_sale": 7' in comp["raw_json"] or '"num_for_sale":7' in comp["raw_json"]
    assert comp["norm"]["num_for_sale"] == 7


def test_empty_price_suggestions_preserves_prev_ladder():
    prev_ps = json.dumps({"Near Mint (NM or M-)": {"value": 10.0, "currency": "USD"}})
    prev = {"price_suggestions_json": prev_ps}
    comp = compute_marketplace_upsert_values(
        "456",
        {},
        prev,
        price_suggestions_payload={},
        fetched_at="fixed",
    )
    assert comp["price_suggestions_json"] == prev_ps


def test_release_payload_none_coalesces_listing_fields():
    prev = {
        "release_lowest_price": 12.5,
        "release_num_for_sale": 3,
        "community_want": 100,
        "community_have": 50,
    }
    comp = compute_marketplace_upsert_values(
        "789",
        {"num_for_sale": 1},
        prev,
        release_payload=None,
        fetched_at="fixed",
    )
    assert comp["release_lowest_price"] == 12.5
    assert comp["release_num_for_sale"] == 3
    assert comp["community_want"] == 100
    assert comp["community_have"] == 50


def test_release_payload_extracts_and_fills_missing_coalesce():
    release_pl = {
        "lowest_price": {"value": 20.0},
        "num_for_sale": 9,
        "community": {"want": 5, "have": 8},
    }
    comp = compute_marketplace_upsert_values(
        "999",
        {},
        {},
        release_payload=release_pl,
        fetched_at="fixed",
    )
    assert comp["release_lowest_price"] == 20.0
    assert comp["release_num_for_sale"] == 9
    assert comp["community_want"] == 5
    assert comp["community_have"] == 8

