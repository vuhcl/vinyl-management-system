"""Unit tests for hull math, manifest merge, and calibration helpers."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from price_estimator.src.inference.confidence_interval import hull_float_usd_then_round
from price_estimator.src.models.confidence_calibration import dollars_abs_errors_from_log1p_pair
from price_estimator.src.models.manifest_merge import merge_model_manifest
from price_estimator.src.models.regressor_constants import MANIFEST_FILE


def test_hull_float_usd_then_round_widens_to_include_point() -> None:
    lo, hi = hull_float_usd_then_round(30.0, 32.0, 100.0)
    assert lo <= 100.0 <= hi
    assert lo <= 32.0 and hi >= 32.0


def test_dollars_abs_errors_log1p_pair() -> None:
    y = np.log1p(np.array([10.0, 20.0]))
    p = np.log1p(np.array([12.0, 18.0]))
    err = dollars_abs_errors_from_log1p_pair(y, p)
    assert err.shape == (2,)
    assert err[0] == pytest.approx(abs(12.0 - 10.0))
    assert err[1] == pytest.approx(abs(18.0 - 20.0))


def test_merge_model_manifest_shallow_update(tmp_path: Path) -> None:
    p = tmp_path / MANIFEST_FILE
    p.write_text(json.dumps({"schema_version": 2, "backend": "xgboost"}))
    merge_model_manifest(
        tmp_path,
        {
            "quantile_intervals": {
                "enabled": True,
                "lower": "x.joblib",
                "upper": "y.joblib",
            }
        },
    )
    data = json.loads(p.read_text())
    assert data["schema_version"] == 2
    assert data["quantile_intervals"]["enabled"] is True
