"""``InferenceService.fetch_stats`` live Discogs hydration."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from price_estimator.src.inference.service import InferenceService
from price_estimator.src.storage.marketplace_db import (
    compute_marketplace_upsert_values,
)


def test_fetch_stats_live_upserts_price_suggestions_payload(
    tmp_path: Path,
) -> None:
    mp = MagicMock()
    mp_row: dict[str, object] = {
        "release_id": "1",
        "fetched_at": "2026-01-01T00:00:00+00:00",
        "num_for_sale": 5,
        "raw_json": "{}",
        "release_lowest_price": 12.0,
        "blocked_from_sale": 0,
        "price_suggestions_json": '{"Near Mint (NM or M-)": {"value": 42.0}}',
        "community_want": 1,
        "community_have": 2,
    }
    mp.get.return_value = dict(mp_row)

    captured: dict[str, object] = {}

    def _upsert(*_: object, **kwargs: object) -> dict[str, object]:
        captured["kwargs"] = kwargs
        comp = compute_marketplace_upsert_values(
            "1",
            {},
            mp.get.return_value or {},
            release_payload=kwargs.get("release_payload"),
            price_suggestions_payload=kwargs.get("price_suggestions_payload"),
        )
        merged = {**mp.get.return_value}
        merged["price_suggestions_json"] = comp["price_suggestions_json"]
        merged["release_lowest_price"] = comp["release_lowest_price"]
        merged["num_for_sale"] = comp["num_for_sale"]
        mp.get.return_value = merged
        return comp["norm"]

    mp.upsert.side_effect = _upsert

    fs = MagicMock()
    fs.get.return_value = {"genre": "Rock", "country": "US", "year": 2000}

    (tmp_path / "condition_params.json").write_text(
        json.dumps({"alpha": 0.06, "beta": 0.04, "ref_grade": 8.0})
    )

    client = MagicMock()
    client.get_release_with_retries.return_value = {
        "lowest_price": 12.0,
        "num_for_sale": 5,
        "community": {"want": 1, "have": 2},
    }
    client.get_price_suggestions_with_retries.return_value = {
        "Near Mint (NM or M-)": {"value": 99.0, "currency": "USD"},
    }

    svc = InferenceService(
        model_dir=tmp_path,
        marketplace_store=mp,
        feature_store=fs,
    )
    svc.redis_cache = MagicMock()
    svc.redis_cache.get.return_value = None
    svc._get_discogs_client = lambda: client

    out = svc.fetch_stats("1", use_cache=True, refresh=True)

    assert out["source"] == "live"
    kw = captured["kwargs"]
    assert isinstance(kw, dict)
    ps = kw.get("price_suggestions_payload")
    assert isinstance(ps, dict)
    assert ps.get("Near Mint (NM or M-)", {}).get("value") == 99.0
    client.get_release_with_retries.assert_called_once()
    client.get_price_suggestions_with_retries.assert_called_once()

    set_calls = svc.redis_cache.set.call_args_list
    assert set_calls, "expected redis SET after live fetch"
    blob = set_calls[-1][0][1]
    assert blob.get("price_suggestions_json")
    assert "99" in str(blob.get("price_suggestions_json"))


def test_invalidate_marketplace_redis_cache_strips_release_id(tmp_path: Path) -> None:
    mp = MagicMock()
    fs = MagicMock()
    (tmp_path / "condition_params.json").write_text(
        json.dumps({"alpha": 0.06, "beta": 0.04, "ref_grade": 8.0})
    )

    svc = InferenceService(
        model_dir=tmp_path,
        marketplace_store=mp,
        feature_store=fs,
    )
    svc.redis_cache = MagicMock()
    svc.redis_cache.enabled.return_value = True

    out = svc.invalidate_marketplace_redis_cache("  99  ")
    assert out == {"release_id": "99", "redis_cache_enabled": True}
    svc.redis_cache.invalidate.assert_called_once_with("99")
