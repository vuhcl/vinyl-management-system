"""FastAPI /predict contract — extension-consumed JSON fields."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from grader.demo.golden_loader import golden_predict_demo_path
from grader.serving.rule_postprocess import init_rule_stack
from grader.serving.schemas import PredictResponse


def _golden_example_text() -> str:
    data = json.loads(golden_predict_demo_path().read_text(encoding="utf-8"))
    examples = data.get("examples") or []
    assert examples, "golden_predict_demo.json needs examples"
    return str(examples[0]["text"])


@pytest.fixture
def predict_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    import grader.serving.main as main

    monkeypatch.delenv("MLFLOW_MODEL_URI", raising=False)
    init_rule_stack()

    text = _golden_example_text()

    def fake_predict(df: pd.DataFrame) -> pd.DataFrame:
        row = df.iloc[0]
        return pd.DataFrame(
            [
                {
                    "item_id": row["item_id"],
                    "text": row["text"],
                    "predicted_sleeve_condition": "Good",
                    "predicted_media_condition": "Good",
                    "sleeve_confidence": 0.88,
                    "media_confidence": 0.85,
                }
            ]
        )

    mock_model = MagicMock()
    mock_model.predict.side_effect = fake_predict
    monkeypatch.setattr(main, "load_grader_pyfunc", lambda: mock_model)
    monkeypatch.setattr(
        main,
        "verify_serving_guidelines_pairing",
        lambda model_uri=None: None,
    )

    with TestClient(main.app) as client:
        yield client


def test_predict_response_matches_extension_contract(
    predict_client: TestClient,
) -> None:
    text = _golden_example_text()
    r = predict_client.post("/predict", json={"text": text})
    assert r.status_code == 200
    parsed = PredictResponse.model_validate(r.json())
    assert len(parsed.predictions) == 1
    row = parsed.predictions[0]
    assert row.predicted_media_condition == "Good"
    assert row.predicted_sleeve_condition == "Good"
    assert row.media_confidence == pytest.approx(0.85)
    assert row.sleeve_confidence == pytest.approx(0.88)


def test_predict_rejects_empty_batch_modes(predict_client: TestClient) -> None:
    r = predict_client.post("/predict", json={})
    assert r.status_code == 422
