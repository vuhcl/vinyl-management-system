"""Behavior tests for the optional Redis layer.

These tests focus on graceful degradation paths because the production
boot sequence on local dev machines (no REDIS_HOST) and during a
Memorystore outage must both stay healthy. Live Redis integration is
exercised by the GKE smoke run, not here.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from price_estimator.src.storage.redis_stats_cache import RedisStatsCache


def test_disabled_when_host_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("REDIS_HOST", raising=False)
    cache = RedisStatsCache()
    assert cache.enabled() is False
    assert cache.get("12345") is None
    cache.set("12345", {"release_lowest_price": 9.99, "num_for_sale": 3})
    cache.invalidate("12345")


def test_disabled_when_redis_package_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REDIS_HOST", "fake-host")
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args, **kwargs):
        if name == "redis":
            raise ImportError("simulated missing redis package")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=fake_import):
        cache = RedisStatsCache()
    assert cache.enabled() is False
    assert cache.get("12345") is None


def test_disabled_when_ping_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REDIS_HOST", "127.0.0.1")
    monkeypatch.setenv("REDIS_PORT", "1")
    cache = RedisStatsCache()
    assert cache.enabled() is False


def test_get_returns_none_for_nondict_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("REDIS_HOST", raising=False)
    cache = RedisStatsCache()

    class StubClient:
        def __init__(self, value: str | None) -> None:
            self._value = value

        def get(self, key: str) -> str | None:
            return self._value

    cache._client = StubClient('"not a dict"')
    assert cache.get("12345") is None

    cache._client = StubClient("{not valid json")
    assert cache.get("12345") is None

    cache._client = StubClient(None)
    assert cache.get("12345") is None


def test_get_round_trip_with_stub_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("REDIS_HOST", raising=False)
    cache = RedisStatsCache()

    class StubClient:
        def __init__(self) -> None:
            self.store: dict[str, str] = {}

        def get(self, key: str) -> str | None:
            return self.store.get(key)

        def setex(self, key: str, ttl: int, value: str) -> None:
            self.store[key] = value

        def delete(self, key: str) -> None:
            self.store.pop(key, None)

    cache._client = StubClient()
    payload = {"release_lowest_price": 12.34, "num_for_sale": 7}
    cache.set("rid-1", payload)
    assert cache.get("rid-1") == payload
    cache.invalidate("rid-1")
    assert cache.get("rid-1") is None


def test_ttl_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("REDIS_HOST", raising=False)
    monkeypatch.setenv("REDIS_TTL_SECONDS", "60")
    cache = RedisStatsCache(ttl_seconds=999)
    assert cache.ttl_seconds == 60


def test_invalid_env_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("REDIS_HOST", raising=False)
    monkeypatch.setenv("REDIS_PORT", "not-an-int")
    cache = RedisStatsCache(port=6379)
    assert cache.port == 6379
