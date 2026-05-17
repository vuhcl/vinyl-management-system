"""Regression tests for serving rule post-process ↔ RuleEngine contract."""

from __future__ import annotations

import pandas as pd

from grader.serving.rule_postprocess import _pyfunc_df_to_prediction_dicts
from grader.src.rules.rule_engine import RuleEngine


def test_pyfunc_singleton_confidence_scores_rule_engine_apply(guidelines_path):
    """
    MLflow pyfunc exposes only top-1 confidence per target; RuleEngine uses
    ``scores.get(predicted_grade)`` — singleton maps must remain valid.
    """
    engine = RuleEngine(guidelines_path=guidelines_path)
    out_df = pd.DataFrame(
        [
            {
                "item_id": "row1",
                "predicted_sleeve_condition": "Very Good Plus",
                "predicted_media_condition": "Very Good Plus",
                "sleeve_confidence": 0.71,
                "media_confidence": 0.82,
            }
        ]
    )
    records = [
        {
            "source": "test",
            "media_verifiable": True,
            "text_clean": "plays perfectly with light sleeve wear",
        }
    ]
    preds = _pyfunc_df_to_prediction_dicts(out_df, records)
    assert len(preds) == 1
    pred = preds[0]
    assert pred["confidence_scores"]["sleeve"]["Very Good Plus"] == 0.71
    assert pred["confidence_scores"]["media"]["Very Good Plus"] == 0.82

    text = str(records[0]["text_clean"])
    result = engine.apply(pred, text)
    assert result["predicted_sleeve_condition"] in (
        "Very Good Plus",
        "Very Good",
        "Near Mint",
    )
    assert "metadata" in result
    assert "contradiction_detected" in result["metadata"]
