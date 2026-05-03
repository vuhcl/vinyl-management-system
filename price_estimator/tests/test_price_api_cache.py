"""Redis marketplace cache HTTP surface."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from starlette.testclient import TestClient


def test_delete_cache_marketplace_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    import price_estimator.src.api.main as api_main

    mock_svc = MagicMock()
    mock_svc.invalidate_marketplace_redis_cache.return_value = {
        "release_id": "42",
        "redis_cache_enabled": True,
    }
    monkeypatch.setattr(api_main, "_svc", mock_svc)

    tc = TestClient(api_main.app)
    r = tc.delete("/cache/marketplace/42")
    assert r.status_code == 200
    body = r.json()
    assert body["release_id"] == "42"
    assert body["redis_cache_enabled"] is True
    mock_svc.invalidate_marketplace_redis_cache.assert_called_once_with("42")
