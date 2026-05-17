"""Monitoring smoke tests for grader (schema + prediction drift)."""
from __future__ import annotations

import great_expectations as gx
import numpy as np
import pandas as pd
import pytest
from great_expectations.expectations import ExpectColumnValuesToBeBetween

from price_estimator.src.monitoring.drift_stats import categorical_psi_topk

pytestmark = pytest.mark.monitoring


def _batch(df: pd.DataFrame):
    ctx = gx.get_context(mode="ephemeral")
    src = ctx.data_sources.add_pandas(name="nlp_logs")
    return src.read_dataframe(df, asset_name="batch")


def test_monitoring_ge_inference_log_schema() -> None:
    df = pd.DataFrame(
        {
            "item_id": ["a", "b"],
            "predicted_sleeve": [
                "Near Mint (NM or M-)",
                "Very Good Plus (VG+)",
            ],
            "predicted_media": [
                "Near Mint (NM or M-)",
                "Near Mint (NM or M-)",
            ],
            "sleeve_confidence": [0.92, 0.71],
            "media_confidence": [0.88, 0.65],
        }
    )
    b = _batch(df)
    assert b.validate(
        ExpectColumnValuesToBeBetween(
            column="sleeve_confidence", min_value=0.0, max_value=1.0
        )
    ).success
    assert b.validate(
        ExpectColumnValuesToBeBetween(
            column="media_confidence", min_value=0.0, max_value=1.0
        )
    ).success


def test_monitoring_prediction_grade_shift_detected() -> None:
    rng = np.random.default_rng(7)
    grades = ["NM", "VG+", "G+", "M"]
    ref = pd.Series(rng.choice(grades, size=400, p=[0.4, 0.35, 0.2, 0.05]))
    cur = pd.Series(["NM"] * 400)
    psi = categorical_psi_topk(ref, cur, k=10)
    assert psi > 0.10
