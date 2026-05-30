"""Grader serving /health and HEAD probe behavior."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def grader_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    import grader.serving.main as main

    monkeypatch.setattr(main, "_model", MagicMock())
    monkeypatch.setattr(
        main,
        "_health_snapshot",
        {
            "status": "ok",
            "model_loaded": True,
            "guidelines_version": "test-v1",
            "model_guidelines_version_tag": "test-v1",
        },
    )
    return TestClient(main.app)


def test_health_get_returns_snapshot(grader_client: TestClient) -> None:
    r = grader_client.get("/health")
    assert r.status_code == 200
    assert r.json()["guidelines_version"] == "test-v1"


def test_health_head_returns_empty_200(grader_client: TestClient) -> None:
    r = grader_client.head("/health")
    assert r.status_code == 200
    assert r.content == b""
