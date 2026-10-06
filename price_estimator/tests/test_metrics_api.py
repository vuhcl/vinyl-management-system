# Run on its own; Instrumentator metrics are shared across apps in one process.
# Second app silently serves the first app's series (no exception).
# Assert a service-unique handler (/predict, /estimate, /auth/login) so a wrong
# registry fails; do not assert exact counter == 1 (process-global).
"""Price API Prometheus /metrics (HTTP request latency)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from starlette.testclient import TestClient


@pytest.fixture
def price_metrics_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
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
    mock_svc.features.ping.return_value = None
    mock_svc.model_dir = MagicMock()
    mock_svc.model_dir.__truediv__ = lambda self, other: MagicMock(
        is_file=lambda: False
    )
    mock_svc.model_source = "local"
    # Set before TestClient so lifespan get_service() returns the mock.
    monkeypatch.setattr(api_main, "_svc", mock_svc)
    return TestClient(api_main.app)


def test_metrics_exposes_estimate_handler(price_metrics_client: TestClient) -> None:
    before = price_metrics_client.get("/metrics")
    assert before.status_code == 200
    assert "http_request_duration_seconds" in before.text
    assert "http_requests_total" in before.text

    r = price_metrics_client.post(
        "/estimate",
        json={
            "release_id": "456663",
            "media_condition": "Very Good (VG)",
            "sleeve_condition": "Good (G)",
        },
    )
    assert r.status_code in (200, 422, 500)

    after = price_metrics_client.get("/metrics")
    assert after.status_code == 200
    assert 'handler="/estimate"' in after.text
