"""FastAPI /estimate contract — extension-consumed JSON fields."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from starlette.testclient import TestClient

from price_estimator.src.api.schemas import EstimateResponse


@pytest.fixture
def estimate_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    import price_estimator.src.api.main as api_main

    monkeypatch.delenv("VINYLIQ_API_KEY", raising=False)
    mock_svc = MagicMock()
    mock_svc.estimate.return_value = {
        "release_id": "456663",
        "estimated_price": 42.5,
        "confidence_interval": [38.0, 47.0],
        "baseline_median": 40.0,
        "model_version": "test-v1",
        "status": "ok",
        "num_for_sale": 3,
        "warnings": [],
        "residual_anchor_usd": 40.0,
    }
    monkeypatch.setattr(api_main, "_svc", mock_svc)
    return TestClient(api_main.app)


def test_estimate_response_matches_extension_contract(
    estimate_client: TestClient,
) -> None:
    r = estimate_client.post(
        "/estimate",
        json={
            "release_id": "456663",
            "media_condition": "Very Good (VG)",
            "sleeve_condition": "Good (G)",
        },
    )
    assert r.status_code == 200
    parsed = EstimateResponse.model_validate(r.json())
    assert parsed.estimated_price == 42.5
    assert len(parsed.confidence_interval) >= 2
    assert parsed.confidence_interval[0] == 38.0
    assert parsed.confidence_interval[1] == 47.0
    assert parsed.model_version == "test-v1"
    assert parsed.status == "ok"


def test_estimate_forwards_marketplace_client_overlay(
    estimate_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import price_estimator.src.api.main as api_main

    mock_svc = api_main._svc
    assert mock_svc is not None
    r = estimate_client.post(
        "/estimate",
        json={
            "release_id": "99",
            "media_condition": "Near Mint (NM or M-)",
            "sleeve_condition": "Near Mint (NM or M-)",
            "marketplace_client": {
                "release_lowest_price": 25.0,
                "num_for_sale": 5,
            },
        },
    )
    assert r.status_code == 200
    mock_svc.estimate.assert_called_with(
        "99",
        "Near Mint (NM or M-)",
        "Near Mint (NM or M-)",
        refresh_stats=False,
        marketplace_client={
            "release_lowest_price": 25.0,
            "num_for_sale": 5,
        },
    )


def test_estimate_requires_api_key_when_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import price_estimator.src.api.main as api_main

    monkeypatch.setenv("VINYLIQ_API_KEY", "secret-key")
    mock_svc = MagicMock()
    mock_svc.estimate.return_value = {
        "release_id": "1",
        "estimated_price": 10.0,
        "confidence_interval": [9.0, 11.0],
        "model_version": "test",
        "status": "ok",
        "num_for_sale": 0,
        "warnings": [],
    }
    monkeypatch.setattr(api_main, "_svc", mock_svc)
    tc = TestClient(api_main.app)

    r = tc.post(
        "/estimate",
        json={"release_id": "1"},
    )
    assert r.status_code == 401

    r = tc.post(
        "/estimate",
        json={"release_id": "1"},
        headers={"X-API-Key": "secret-key"},
    )
    assert r.status_code == 200
