# Run on its own; Instrumentator metrics are shared across apps in one process.
# Second app silently serves the first app's series (no exception).
# Assert a service-unique handler (/predict, /estimate, /auth/login) so a wrong
# registry fails; do not assert exact counter == 1 (process-global).
"""Grader serving Prometheus /metrics (HTTP request latency)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def grader_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    import grader.serving.main as main

    mock_model = MagicMock()
    mock_model.predict.return_value = pd.DataFrame(
        {
            "predicted_sleeve_condition": ["VG+"],
            "predicted_media_condition": ["VG"],
            "sleeve_confidence": [0.9],
            "media_confidence": [0.8],
        }
    )
    monkeypatch.setattr(main, "_model", mock_model)
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
    monkeypatch.setattr(
        main,
        "apply_rules_to_pyfunc_batch",
        lambda out, texts, ids, metas: [
            {
                "item_id": ids[0],
                "predicted_sleeve_condition": "VG+",
                "predicted_media_condition": "VG",
                "metadata": {
                    "contradiction_detected": False,
                    "rule_override_applied": False,
                    "rule_override_target": None,
                    "guidelines_version": "test-v1",
                },
            }
        ],
    )
    monkeypatch.setattr(main, "get_model_guidelines_version_tag", lambda: "test-v1")
    return TestClient(main.app)


def test_metrics_exposes_predict_handler(grader_client: TestClient) -> None:
    before = grader_client.get("/metrics")
    assert before.status_code == 200
    assert "http_request_duration_seconds" in before.text
    assert "http_requests_total" in before.text

    r = grader_client.post(
        "/predict",
        json={"text": "VG+ vinyl, sleeve VG"},
    )
    # Uniqueness is the handler label; 200 preferred but not required.
    assert r.status_code in (200, 422, 500)

    after = grader_client.get("/metrics")
    assert after.status_code == 200
    assert 'handler="/predict"' in after.text
    # Optional probe path (may also appear from other tests in this process).
    assert "http_requests_total" in after.text
