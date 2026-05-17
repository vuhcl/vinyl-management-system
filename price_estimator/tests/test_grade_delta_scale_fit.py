"""Tests for pooled cross-grade ``grade_delta_scale`` fit."""
from __future__ import annotations

import json
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
    assert blob["fit_metadata"]["fit_kind"] == "cross_grade_bin_median_v2_alpha_beta"
    assert blob["fit_metadata"]["row_count_sales"] == 50
    assert blob["fit_metadata"]["bins_used"] == 1
    assert "alpha" in blob and "beta" in blob
    assert "sale_date_min" in blob["fit_metadata"]


def test_fit_grade_delta_scale_fixed_alpha_beta_legacy_kind() -> None:
    df = _synthetic_sale_frame()
    blob = fit_grade_delta_scale_from_frame(
        df,
        nm_grade_key="Near Mint (NM or M-)",
        fit_alpha_beta=False,
        min_bin_rows=10,
        min_grade_rows=5,
        gammas=(0.0,),
        age_ks=(0.0,),
    )
    validate_grade_delta_scale_fit_json(blob)
    assert blob["fit_metadata"]["fit_kind"] == "cross_grade_bin_median_v1"
    assert "alpha" not in blob and "beta" not in blob


def test_fit_alpha_beta_recovers_asymmetric_synthetic() -> None:
    """Triplet strata identify α and β separately when sale log-prices match uplift geometry."""
    rows: list[dict[str, float | str]] = []
    anchor = 50.0
    year = 2000.0
    L_nm = math.log1p(100.0)
    L_sym_vgp = L_nm - (0.06 + 0.04)  # (6,6)→(7,7) with g=1
    L_media_sl = L_nm - 0.06  # (6,7)→(7,7)
    L_sleeve_sl = L_nm - 0.04  # (7,6)→(7,7)
    for _ in range(30):
        rows.append(
            {
                "order_date": "2020-01-15",
                "usd": 100.0,
                "log_price": L_nm,
                "media_ord": 7.0,
                "sleeve_ord": 7.0,
                "eff_ord": 7.0,
                "anchor_usd": anchor,
                "year": year,
            }
        )
    for _ in range(30):
        rows.append(
            {
                "order_date": "2020-01-15",
                "usd": float(np.expm1(L_sym_vgp)),
                "log_price": L_sym_vgp,
                "media_ord": 6.0,
                "sleeve_ord": 6.0,
                "eff_ord": 6.0,
                "anchor_usd": anchor,
                "year": year,
            }
        )
    for _ in range(30):
        rows.append(
            {
                "order_date": "2020-01-15",
                "usd": float(np.expm1(L_media_sl)),
                "log_price": L_media_sl,
                "media_ord": 6.0,
                "sleeve_ord": 7.0,
                "eff_ord": 6.0,
                "anchor_usd": anchor,
                "year": year,
            }
        )
    for _ in range(30):
        rows.append(
            {
                "order_date": "2020-01-15",
                "usd": float(np.expm1(L_sleeve_sl)),
                "log_price": L_sleeve_sl,
                "media_ord": 7.0,
                "sleeve_ord": 6.0,
                "eff_ord": 6.0,
                "anchor_usd": anchor,
                "year": year,
            }
        )
    df = pd.DataFrame(rows)
    blob = fit_grade_delta_scale_from_frame(
        df,
        nm_grade_key="Near Mint (NM or M-)",
        min_bin_rows=40,
        min_grade_rows=10,
        gammas=(0.0,),
        age_ks=(0.0,),
    )
    assert abs(blob["alpha"] - 0.06) < 1e-6
    assert abs(blob["beta"] - 0.04) < 1e-6


def test_grade_delta_scale_json_merges_alpha_beta_into_condition_params(tmp_path) -> None:
    from price_estimator.src.models.condition_adjustment import (
        load_params_with_grade_delta_overlays,
        save_params,
    )

    save_params(tmp_path / "condition_params.json", {"alpha": 0.06, "beta": 0.04})
    fit_blob = {
        "schema_version": 1,
        "fit_metadata": {"fitted_at": "2020-01-01T00:00:00+00:00"},
        "price_ref_usd": 40.0,
        "price_gamma": 0.0,
        "price_scale_min": 0.25,
        "price_scale_max": 4.0,
        "age_k": 0.0,
        "age_center_year": 1999.0,
        "alpha": 0.055,
        "beta": 0.038,
    }
    (tmp_path / "grade_delta_scale.json").write_text(json.dumps(fit_blob))
    merged = load_params_with_grade_delta_overlays(tmp_path)
    assert merged["alpha"] == pytest.approx(0.055)
    assert merged["beta"] == pytest.approx(0.038)
    assert merged["grade_delta_scale"]["price_ref_usd"] == 40.0


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
        base_alpha=0.06,
        base_beta=0.04,
        scale=sp,
    )
    assert abs(lift - (0.06 + 0.04)) < 1e-9


def test_grid_score_bin_medians_finite() -> None:
    df = add_anchor_decade_bins(_synthetic_sale_frame())
    bins = compute_bin_deltas(df, min_bin_rows=10, min_grade_rows=5)
    sc = grid_score_bin_medians(
        bins,
        base_alpha=0.06,
        base_beta=0.04,
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
