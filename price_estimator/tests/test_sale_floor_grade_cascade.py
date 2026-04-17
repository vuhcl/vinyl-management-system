"""Sale-floor ordinal cascade, uplift, and condition scale parity."""
from __future__ import annotations

import json
from datetime import datetime

import pytest

from price_estimator.src.features.vinyliq_features import (
    GradeDeltaScaleParams,
    apply_condition_log_adjustment,
    grade_delta_scale_params_from_cond,
    scaled_condition_log_adjustment,
)
from price_estimator.src.models.condition_adjustment import (
    load_params_with_grade_delta_overlays,
    merge_grade_delta_scale_dict,
)
from price_estimator.src.models.grade_delta_scale_schema import (
    build_placeholder_grade_delta_fit,
    validate_grade_delta_scale_fit_json,
)
from price_estimator.src.training.sale_floor_targets import (
    eligible_ordinal_cascade_sale_rows,
    sale_floor_blend_bundle,
    sale_floor_blend_config_from_raw,
    sale_floor_blend_sf_cfg_for_policy,
)


def _row(
    *,
    order_date: str,
    price: float,
    media: str,
    sleeve: str,
) -> dict:
    return {
        "order_date": order_date,
        "price_user_usd_approx": price,
        "media_condition": media,
        "sleeve_condition": sleeve,
    }


def test_sale_floor_blend_sf_cfg_for_policy_overrides() -> None:
    raw = {"sale_condition_policy": "ordinal_cascade", "w_base": 0.5}
    nm = sale_floor_blend_sf_cfg_for_policy(raw, "nm_substrings_only")
    oc = sale_floor_blend_sf_cfg_for_policy(raw, "ordinal_cascade")
    assert nm["sale_condition_policy"] == "nm_substrings_only"
    assert oc["sale_condition_policy"] == "ordinal_cascade"
    assert nm["w_base"] == 0.5


def test_sale_floor_blend_config_merges_ordinal_cascade_block() -> None:
    raw = {
        "n_min_trend": 8,
        "ordinal_cascade": {
            "sale_condition_policy": "ordinal_cascade",
            "min_rows_strict": 99,
            "relax_steps": [6.0, 5.0],
        },
    }
    cfg = sale_floor_blend_config_from_raw(raw, nm_grade_key="Near Mint (NM or M-)")
    assert cfg.sale_condition_policy == "ordinal_cascade"
    assert cfg.min_rows_strict == 99
    assert cfg.relax_steps == (6.0, 5.0)


def test_cascade_relaxes_to_vg_plus_when_nm_sparse() -> None:
    cfg = sale_floor_blend_config_from_raw(
        {
            "sale_condition_policy": "ordinal_cascade",
            "min_rows_strict": 5,
            "min_rows_relax_1": 3,
            "min_rows_relax_2": 3,
            "n_min_trend": 3,
        },
        nm_grade_key="Near Mint (NM or M-)",
    )
    rows = [
        _row(order_date="2020-01-01", price=40.0, media="Very Good Plus (VG+)", sleeve="Very Good Plus (VG+)"),
        _row(order_date="2020-02-01", price=42.0, media="Very Good Plus (VG+)", sleeve="Very Good Plus (VG+)"),
        _row(order_date="2020-03-01", price=41.0, media="Very Good Plus (VG+)", sleeve="Very Good Plus (VG+)"),
        _row(order_date="2020-04-01", price=100.0, media="Near Mint (NM or M-)", sleeve="Near Mint (NM or M-)"),
    ]
    t_ref = datetime(2021, 1, 1)
    mp = {"median_price": 50.0, "price_suggestions_json": "{}"}
    elig, tag = eligible_ordinal_cascade_sale_rows(
        rows,
        t_ref,
        cfg=cfg,
        mp_row=mp,
        nm_grade_key="Near Mint (NM or M-)",
        release_year=2010.0,
    )
    assert tag == "relax_1"
    assert len(elig) == 4


def test_cascade_strict_when_enough_nm() -> None:
    cfg = sale_floor_blend_config_from_raw(
        {
            "sale_condition_policy": "ordinal_cascade",
            "min_rows_strict": 3,
            "n_min_trend": 3,
        },
        nm_grade_key="Near Mint (NM or M-)",
    )
    rows = [
        _row(order_date="2020-01-01", price=100.0, media="Near Mint (NM or M-)", sleeve="Near Mint (NM or M-)"),
        _row(order_date="2020-02-01", price=102.0, media="Near Mint (NM or M-)", sleeve="Near Mint (NM or M-)"),
        _row(order_date="2020-03-01", price=101.0, media="Near Mint (NM or M-)", sleeve="Near Mint (NM or M-)"),
    ]
    t_ref = datetime(2021, 1, 1)
    mp = {"median_price": 100.0}
    elig, tag = eligible_ordinal_cascade_sale_rows(
        rows,
        t_ref,
        cfg=cfg,
        mp_row=mp,
        nm_grade_key="Near Mint (NM or M-)",
        release_year=None,
    )
    assert tag == "strict"
    assert len(elig) == 3


def test_label_cap_clamps_absurd_listing_vs_modest_sales() -> None:
    """Toxic listing floors must not produce six-figure y when comps are ~tens of USD."""
    sf = {"sale_condition_policy": "nm_substrings_only", "nm_substrings": ["near mint"]}
    mp = {
        "fetched_at": "2021-01-01T00:00:00",
        "median_price": 90.0,
        "release_lowest_price": 117_820.87,
    }
    sales = [
        _row(
            order_date="2020-06-01",
            price=88.0,
            media="Near Mint (NM or M-)",
            sleeve="Near Mint (NM or M-)",
        ),
    ]
    fetch = {"status": "ok", "fetched_at": "2021-01-01T00:00:00"}
    y, _m, flags = sale_floor_blend_bundle(
        mp,
        sales,
        fetch,
        sf_cfg=sf,
        nm_grade_key="Near Mint (NM or M-)",
        release_year=1999.0,
    )
    assert y is not None
    assert flags.get("has_sale_history") == 1.0
    assert y < 2000.0
    assert y > 40.0


def test_sale_floor_blend_bundle_legacy_policy_flag() -> None:
    sf = {"sale_condition_policy": "nm_substrings_only", "nm_substrings": ["near mint"]}
    mp = {
        "fetched_at": "2021-01-01T00:00:00",
        "median_price": 80.0,
        "release_lowest_price": 75.0,
    }
    sales = [
        _row(order_date="2020-06-01", price=70.0, media="Near Mint (NM or M-)", sleeve="Near Mint (NM or M-)"),
    ]
    fetch = {"status": "ok", "fetched_at": "2021-01-01T00:00:00"}
    y, m, flags = sale_floor_blend_bundle(
        mp,
        sales,
        fetch,
        sf_cfg=sf,
        nm_grade_key="Near Mint (NM or M-)",
        release_year=1999.0,
    )
    assert y is not None and y > 0
    assert flags.get("sale_relax_tier_code") == -1.0


def test_scaled_condition_matches_legacy_when_scale_off() -> None:
    logp = 4.0
    m, s = 7.0, 7.0
    a, b, ref = -0.06, -0.04, 8.0
    leg = apply_condition_log_adjustment(logp, m, s, alpha=a, beta=b, ref_grade=ref)
    p = GradeDeltaScaleParams(price_gamma=0.0, age_k=0.0)
    adj = scaled_condition_log_adjustment(
        logp,
        m,
        s,
        base_alpha=a,
        base_beta=b,
        ref_grade=ref,
        anchor_usd=100.0,
        release_year=2000.0,
        scale_params=p,
    )
    assert leg == pytest.approx(adj)


def test_scaled_condition_with_price_gamma_differs() -> None:
    logp = 4.0
    m, s = 6.0, 6.0
    a, b, ref = -0.06, -0.04, 8.0
    p = GradeDeltaScaleParams(price_ref_usd=50.0, price_gamma=0.5, price_scale_min=0.1, price_scale_max=10.0)
    adj = scaled_condition_log_adjustment(
        logp,
        m,
        s,
        base_alpha=a,
        base_beta=b,
        ref_grade=ref,
        anchor_usd=200.0,
        release_year=None,
        scale_params=p,
    )
    leg = apply_condition_log_adjustment(logp, m, s, alpha=a, beta=b, ref_grade=ref)
    assert adj != pytest.approx(leg)


def test_grade_delta_scale_params_from_cond_nested() -> None:
    cond = {"alpha": -0.06, "grade_delta_scale": {"price_gamma": 0.0, "age_k": 0.0}}
    g = grade_delta_scale_params_from_cond(cond)
    assert g is not None
    assert g.price_gamma == 0.0


def test_load_params_with_grade_delta_overlays(tmp_path) -> None:
    (tmp_path / "condition_params.json").write_text(
        json.dumps({"alpha": -0.06, "beta": -0.04, "ref_grade": 8.0})
    )
    (tmp_path / "grade_delta_scale.json").write_text(
        json.dumps({"price_ref_usd": 40.0, "price_gamma": 0.2, "age_k": 0.0})
    )
    merged = load_params_with_grade_delta_overlays(tmp_path)
    inner = merged.get("grade_delta_scale")
    assert isinstance(inner, dict)
    assert inner.get("price_ref_usd") == 40.0
    assert inner.get("price_gamma") == 0.2


def test_merge_grade_delta_scale_dict() -> None:
    base = {"alpha": -0.06, "grade_delta_scale": {"price_gamma": 0.1}}
    out = merge_grade_delta_scale_dict(base, {"price_ref_usd": 30.0})
    assert out["grade_delta_scale"]["price_gamma"] == 0.1
    assert out["grade_delta_scale"]["price_ref_usd"] == 30.0


def test_validate_grade_delta_scale_fit_json_accepts_placeholder() -> None:
    blob = build_placeholder_grade_delta_fit()
    validate_grade_delta_scale_fit_json(blob)


def test_validate_grade_delta_scale_rejects_bad_schema() -> None:
    with pytest.raises(ValueError):
        validate_grade_delta_scale_fit_json({"schema_version": 2})
