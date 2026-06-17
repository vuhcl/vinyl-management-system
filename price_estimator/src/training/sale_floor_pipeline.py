"""Log-blend, diagnostics bundle, and public training helpers."""
from __future__ import annotations

import copy
import math
from datetime import datetime
from typing import Any

import numpy as np

from price_estimator.src.inference.anchor_guardrails_config import AnchorGuardrailsConfig
from price_estimator.src.sale_floor.anchor_guardrails import (
    blend_path_anchor_usd,
    prepare_stats_for_serving,
)
from price_estimator.src.sale_floor.reference_floor import reference_floor_training_usd
from price_estimator.src.storage.marketplace_db import marketplace_inference_stats_from_row
from price_estimator.src.training.label_synthesis import parse_price_suggestion_value

from .sale_floor_blend_config import (
    SaleFloorBlendConfig,
    _cap_final_y_label,
    _cap_listing_floor_against_sales,
    sale_floor_blend_config_from_raw,
)
from .sale_floor_enums import SaleConditionPolicy
from .sale_floor_eligibility import (
    eligible_nm_sale_rows,
    eligible_ordinal_cascade_sale_rows,
)
from .sale_floor_inference import (
    effective_listing_floor_lo,
    residual_anchor_m_full_data,
    residual_anchor_m_no_sale_history,
)
from .sale_floor_nowcast import sold_nowcast_s
from price_estimator.src.sale_floor.blend import blend_weight_w_eff, sale_floor_blend_y

from .sale_floor_row_parsing import _positive, reference_time_t_ref

__all__ = [
    "blend_weight_w_eff",
    "sale_floor_blend_bundle",
    "sale_floor_blend_sf_cfg_for_policy",
    "sale_floor_blend_y",
    "sale_floor_label_diagnostics",
]


def _sale_floor_blend_compute(
    mp_row: dict[str, Any],
    sale_rows: list[dict[str, Any]],
    fetch_status: dict[str, Any] | None,
    *,
    sf_cfg: dict[str, Any],
    nm_grade_key: str,
    release_year: float | None = None,
    ag_cfg: AnchorGuardrailsConfig | None = None,
) -> tuple[float | None, float | None, dict[str, float], dict[str, Any]]:
    """
    Core sale-floor blend; returns ``(y_label, m_anchor, x_flags, diagnostics)``.

    ``diagnostics`` is for QA (listing vs sold nowcast vs caps); safe to log, not used in training X.
    """
    raw_cfg = sf_cfg if isinstance(sf_cfg, dict) else {}
    cfg = sale_floor_blend_config_from_raw(raw_cfg, nm_grade_key=nm_grade_key)

    lo = effective_listing_floor_lo(mp_row)
    lo_raw_usd = float(lo) if lo is not None and lo > 0 else None
    has_listing_floor = 1.0 if lo is not None and lo > 0 else 0.0

    sh_ok = (
        fetch_status is not None
        and str(fetch_status.get("status") or "").strip().lower() == "ok"
    )
    sh_fetched = str(fetch_status.get("fetched_at") or "") if fetch_status else None
    t_ref = reference_time_t_ref(str(mp_row.get("fetched_at") or ""), sh_fetched)

    s: float | None = None
    tier = "none"
    n_elig = 0
    s_imputed = 0.0
    has_sale_history = 0.0
    relax_tag = "n/a"
    elig: list[tuple[datetime, float]] = []

    if t_ref is not None and sh_ok:
        if cfg.sale_condition_policy == SaleConditionPolicy.ORDINAL_CASCADE:
            elig, relax_tag = eligible_ordinal_cascade_sale_rows(
                sale_rows,
                t_ref,
                cfg=cfg,
                mp_row=mp_row,
                nm_grade_key=nm_grade_key,
                release_year=release_year,
            )
        else:
            elig = eligible_nm_sale_rows(sale_rows, t_ref, cfg=cfg)
            relax_tag = "legacy_nm_substrings"

        s, tier, n_elig = sold_nowcast_s(elig, t_ref, cfg=cfg)
        if s is not None and s > 0:
            has_sale_history = 1.0

    p_max_obs: float | None = None
    if elig:
        _prices = np.array([float(p) for _, p in elig], dtype=np.float64)
        if _prices.size > 0:
            p_max_obs = float(np.max(_prices))

    s_pre_cap = float(s) if s is not None and s > 0 else None
    if (
        cfg.label_cap_enabled
        and s is not None
        and s > 0
        and p_max_obs is not None
        and p_max_obs > 0
    ):
        hi_now = p_max_obs * float(cfg.label_cap_sale_nowcast_max_multiple_of_max_price)
        if math.isfinite(hi_now) and hi_now > 0:
            s = min(float(s), hi_now)

    lo_for_blend: float | None = lo
    listing_lo_clip_applied = False
    if cfg.label_cap_enabled and lo_for_blend is not None and lo_for_blend > 0:
        k_clip = float(cfg.label_cap_listing_lo_clip_multiple_of_sale_peak)
        if (
            p_max_obs is not None
            and p_max_obs > 0
            and k_clip > 0
            and math.isfinite(k_clip)
            and math.isfinite(float(lo_for_blend))
        ):
            hi_clip = float(p_max_obs) * k_clip
            if hi_clip > 0 and math.isfinite(hi_clip) and float(lo_for_blend) > hi_clip:
                lo_for_blend = hi_clip
                listing_lo_clip_applied = True
        lo_for_blend = _cap_listing_floor_against_sales(
            float(lo_for_blend),
            s,
            p_max_obs,
            cfg=cfg,
        )
        if p_max_obs is None and (s is None or s <= 0):
            med_only = _positive(mp_row.get("release_lowest_price"))
            if med_only is not None and med_only > 0:
                lo_for_blend = min(
                    lo_for_blend,
                    float(med_only)
                    * float(cfg.label_cap_listing_max_multiple_of_median_only),
                )

    y_blend = sale_floor_blend_y(s, lo_for_blend, tier, cfg=cfg)
    diag: dict[str, Any] = {
        "listing_floor_raw_usd": lo_raw_usd,
        "listing_floor_for_blend_usd": float(lo_for_blend)
        if lo_for_blend is not None and lo_for_blend > 0
        else None,
        "sold_nowcast_usd": float(s) if s is not None and s > 0 else None,
        "sold_nowcast_usd_pre_cap": s_pre_cap,
        "sold_tier": tier,
        "n_eligible_sales": int(n_elig),
        "p_max_sale_observed_usd": p_max_obs,
        "sale_history_fetch_ok": bool(sh_ok),
        "sale_relax_tag": relax_tag,
        "release_lowest_price_mp_usd": _positive(mp_row.get("release_lowest_price")),
        "y_blend_usd": float(y_blend) if y_blend is not None and y_blend > 0 else None,
        "listing_lo_clip_applied": listing_lo_clip_applied,
    }

    flags = {
        "has_sale_history": has_sale_history,
        "s_imputed": s_imputed,
        "has_listing_floor": has_listing_floor,
        "sale_relax_tier_code": _relax_tier_code(relax_tag),
    }

    if y_blend is None:
        return None, None, flags, diag

    y_f = float(y_blend)
    if has_sale_history:
        m = residual_anchor_m_full_data(mp_row, nm_grade_key=nm_grade_key)
    else:
        m = residual_anchor_m_no_sale_history(mp_row, nm_grade_key=nm_grade_key)
    if m is None:
        m = y_f
    m_f = float(m)
    m_anchor_raw_usd = float(m_f)

    if ag_cfg is not None and ag_cfg.enabled:
        ref_train, ref_diag = reference_floor_training_usd(
            mp_row,
            sale_rows,
            fetch_status,
            nm_grade_key=nm_grade_key,
            ag_cfg=ag_cfg,
        )
        stats = marketplace_inference_stats_from_row(mp_row)
        stats_work = copy.deepcopy(dict(stats))
        for k in (
            "sale_stats_low_usd",
            "sale_stats_median_usd",
            "sale_stats_average_usd",
            "sale_stats_high_usd",
            "n_sales",
        ):
            if ref_diag.get(k) is not None:
                stats_work[k] = ref_diag[k]
        prepare_stats_for_serving(stats_work, ag_cfg, nm_grade_key=nm_grade_key)
        ps_rung = parse_price_suggestion_value(
            stats_work.get("price_suggestions_json"),
            nm_grade_key,
        )
        if ps_rung is not None and ps_rung > 0:
            m_anchor_raw_usd = float(ps_rung)
        if ref_train is not None:
            m_f = blend_path_anchor_usd(
                m_anchor_raw_usd,
                stats_work,
                ag_cfg,
                cfg,
                nm_grade_key=nm_grade_key,
                reference_floor_usd_override=ref_train,
            )
        diag.update(ref_diag)
        diag["m_anchor_raw_usd"] = float(m_anchor_raw_usd)
        diag["m_anchor_blended_usd"] = float(m_f)
        diag["anchor_guardrail_applied"] = bool(
            abs(float(m_f) - float(m_anchor_raw_usd)) > 1e-6
        )

    if cfg.label_cap_enabled:
        y_final = _cap_final_y_label(y_f, m_f, p_max_obs, cfg=cfg)
    else:
        y_final = y_f
    diag["m_anchor_usd"] = float(m_f)
    diag["y_label_final_usd"] = float(y_final)
    diag["label_cap_enabled"] = bool(cfg.label_cap_enabled)
    return (
        float(y_final),
        float(m_f),
        flags,
        diag,
    )


def sale_floor_label_diagnostics(
    mp_row: dict[str, Any],
    sale_rows: list[dict[str, Any]],
    fetch_status: dict[str, Any] | None,
    *,
    sf_cfg: dict[str, Any],
    nm_grade_key: str,
    release_year: float | None = None,
    ag_cfg: AnchorGuardrailsConfig | None = None,
) -> tuple[float | None, float | None, dict[str, float], dict[str, Any]]:
    """
    Training-time sale-floor bundle plus a **diagnostic** dict for label QA.

    The diagnostic keys include ``listing_floor_raw_usd``, ``sold_nowcast_usd``,
    ``p_max_sale_observed_usd``, ``listing_lo_clip_applied``, ``y_label_final_usd``,
    ``m_anchor_usd``, etc.
    """
    return _sale_floor_blend_compute(
        mp_row,
        sale_rows,
        fetch_status,
        sf_cfg=sf_cfg,
        nm_grade_key=nm_grade_key,
        release_year=release_year,
        ag_cfg=ag_cfg,
    )


def sale_floor_blend_bundle(
    mp_row: dict[str, Any],
    sale_rows: list[dict[str, Any]],
    fetch_status: dict[str, Any] | None,
    *,
    sf_cfg: dict[str, Any],
    nm_grade_key: str,
    release_year: float | None = None,
    ag_cfg: AnchorGuardrailsConfig | None = None,
) -> tuple[float | None, float | None, dict[str, float]]:
    """
    Returns ``(y_label, m_anchor, x_flags)``.

    ``x_flags`` includes ``has_sale_history``, ``s_imputed``, ``has_listing_floor`` (0/1 floats).
    """
    y, m, flags, _diag = _sale_floor_blend_compute(
        mp_row,
        sale_rows,
        fetch_status,
        sf_cfg=sf_cfg,
        nm_grade_key=nm_grade_key,
        release_year=release_year,
        ag_cfg=ag_cfg,
    )
    return y, m, flags


def _normalize_sale_condition_policy(policy: str | SaleConditionPolicy) -> str:
    if isinstance(policy, SaleConditionPolicy):
        return policy.value
    try:
        return SaleConditionPolicy(str(policy).strip().lower()).value
    except ValueError:
        return SaleConditionPolicy.NM_SUBSTRINGS_ONLY.value


def sale_floor_blend_sf_cfg_for_policy(
    sf_cfg: dict[str, Any] | None,
    policy: str | SaleConditionPolicy,
) -> dict[str, Any]:
    """Shallow copy of ``sale_floor_blend`` YAML dict with ``sale_condition_policy`` set."""
    raw = dict(sf_cfg) if isinstance(sf_cfg, dict) else {}
    pol = _normalize_sale_condition_policy(policy)
    out = {**raw, "sale_condition_policy": pol}
    return out


def _relax_tier_code(tag: str) -> float:
    return {
        "strict": 0.0,
        "relax_1": 1.0,
        "relax_2": 2.0,
        "none": 3.0,
        "legacy_nm_substrings": -1.0,
        "n/a": -2.0,
    }.get(tag, -3.0)
