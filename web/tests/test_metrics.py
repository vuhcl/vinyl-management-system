# Run on its own; Instrumentator metrics are shared across apps in one process.
# Second app silently serves the first app's series (no exception).
# Assert a service-unique handler (/predict, /estimate, /auth/login) so a wrong
# registry fails; do not assert exact counter == 1 (process-global).
"""Web app Prometheus /metrics (HTTP request latency)."""

from __future__ import annotations

from starlette.testclient import TestClient

from web.app.main import app


def test_metrics_exposes_auth_login_handler() -> None:
    client = TestClient(app)

    home = client.get("/")
    assert home.status_code == 200

    before = client.get("/metrics")
    assert before.status_code == 200
    assert "http_request_duration_seconds" in before.text
    assert "http_requests_total" in before.text

    login = client.get("/auth/login")
    assert login.status_code == 200

    after = client.get("/metrics")
    assert after.status_code == 200
    assert 'handler="/auth/login"' in after.text
