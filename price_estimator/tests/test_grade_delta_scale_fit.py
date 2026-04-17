"""Tests for pooled cross-grade ``grade_delta_scale`` fit."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from price_estimator.src.models.grade_delta_scale_fit import (
    add_anchor_decade_bins,
    compute_bin_deltas,
    fit_grade_delta_scale_from_frame,
    grid_score_bin_medians,
    predicted_vgp_to_nm_log_lift,
)
from price_estimator.src.models.grade_delta_scale_schema import (
    validate_grade_delta_scale_fit_json,
)


def _synthetic_sale_frame(*, n_nm: int = 25, n_vgp: int = 25) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for _ in range(n_nm):
        rows.append(
            {
                "order_date": "2020-01-15",
                "usd": 100.0,
                "log_price": math.log1p(100.0),
                "media_ord": 7.0,
                "sleeve_ord": 7.0,
                "eff_ord": 7.0,
                "anchor_usd": 50.0,
                "year": 2000.0,
            }
        )
    for _ in range(n_vgp):
        rows.append(
            {
                "order_date": "2020-02-01",
                "usd": 80.0,
                "log_price": math.log1p(80.0),
                "media_ord": 6.0,
                "sleeve_ord": 6.0,
                "eff_ord": 6.0,
                "anchor_usd": 50.0,
                "year": 2000.0,
            }
        )
    return pd.DataFrame(rows)


def test_compute_bin_deltas_median_gap() -> None:
    df = add_anchor_decade_bins(_synthetic_sale_frame())
    bins = compute_bin_deltas(df, min_bin_rows=10, min_grade_rows=5)
    assert len(bins) == 1
    b = bins[0]
    assert b.n_nm == 25 and b.n_vgp == 25
    want = float(np.median(np.log1p(np.full(25, 100.0)))) - float(
        np.median(np.log1p(np.full(25, 80.0)))
    )
    assert abs(b.emp_delta_log - want) < 1e-9


def test_fit_grade_delta_scale_from_frame_validates() -> None:
    df = _synthetic_sale_frame()
    blob = fit_grade_delta_scale_from_frame(
        df,
        nm_grade_key="Near Mint (NM or M-)",
        min_bin_rows=10,
        min_grade_rows=5,
        gammas=(0.0, 0.2),
        age_ks=(0.0, 0.05),
    )
    validate_grade_delta_scale_fit_json(blob)
    assert blob["fit_metadata"]["fit_kind"] == "cross_grade_bin_median_v1"
    assert blob["fit_metadata"]["row_count_sales"] == 50
    assert blob["fit_metadata"]["bins_used"] == 1
    assert "sale_date_min" in blob["fit_metadata"]


def test_predicted_vgp_to_nm_log_lift_matches_zero_scale() -> None:
    from price_estimator.src.features.vinyliq_features import GradeDeltaScaleParams

    sp = GradeDeltaScaleParams(
        price_ref_usd=50.0,
        price_gamma=0.0,
        price_scale_min=0.25,
        price_scale_max=4.0,
        age_k=0.0,
        age_center_year=2000.0,
    )
    lift = predicted_vgp_to_nm_log_lift(
        80.0,
        anchor_usd=50.0,
        release_year=2000.0,
        base_alpha=-0.06,
        base_beta=-0.04,
        scale=sp,
    )
    assert abs(lift - (-0.06 - 0.04)) < 1e-9


def test_grid_score_bin_medians_finite() -> None:
    df = add_anchor_decade_bins(_synthetic_sale_frame())
    bins = compute_bin_deltas(df, min_bin_rows=10, min_grade_rows=5)
    sc = grid_score_bin_medians(
        bins,
        base_alpha=-0.06,
        base_beta=-0.04,
        price_ref_usd=50.0,
        age_center_year=2000.0,
        price_gamma=0.0,
        age_k=0.0,
        price_scale_min=0.25,
        price_scale_max=4.0,
    )
    assert math.isfinite(sc)


def test_fit_empty_frame_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        fit_grade_delta_scale_from_frame(
            pd.DataFrame(),
            nm_grade_key="Near Mint (NM or M-)",
        )
