"""Reference-floor helpers for anchor guardrails (training vs inference sources)."""
from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np

from price_estimator.src.inference.anchor_guardrails_config import AnchorGuardrailsConfig
from price_estimator.src.sale_floor.anchor_guardrails import (
    is_inflated_ladder,
    listing_credible_for_caps,
    mint_rung_usd,
    nm_rung_usd,
    reference_floor_usd,
    sale_stats_blend_apply,
)
from price_estimator.src.sale_floor.ratio_guardrails import (
    compute_ratio_sale_ladder,
    diagnostics_to_dict,
    listing_credible_vs_sale_low,
)
from price_estimator.src.storage.marketplace_db import marketplace_inference_stats_from_row
from price_estimator.src.training.sale_floor_inference import max_price_suggestion_ladder_usd
from price_estimator.src.training.sale_floor_row_parsing import (
    _positive,
    parse_iso_datetime,
    reference_time_t_ref,
    sale_row_usd,
)


def sale_stats_from_history(
    sale_rows: list[dict[str, Any]],
    t_ref: datetime,
) -> dict[str, float | None]:
    """Low/median/high/mean USD from all parseable sales with ``order_date <= t_ref``."""
    prices: list[float] = []
    for row in sale_rows:
        od = parse_iso_datetime(str(row.get("order_date") or ""))
        if od is None or od > t_ref:
            continue
        p = sale_row_usd(row)
        if p is not None and p > 0:
            prices.append(float(p))
    if not prices:
        return {
            "sale_stats_low_usd": None,
            "sale_stats_median_usd": None,
            "sale_stats_high_usd": None,
            "sale_stats_average_usd": None,
            "n_sales": 0,
        }
    arr = np.array(prices, dtype=np.float64)
    return {
        "sale_stats_low_usd": float(np.min(arr)),
        "sale_stats_median_usd": float(np.median(arr)),
        "sale_stats_high_usd": float(np.max(arr)),
        "sale_stats_average_usd": float(np.mean(arr)),
        "n_sales": int(arr.size),
    }


def reference_floor_inference_usd(
    stats: dict[str, Any],
    cfg: AnchorGuardrailsConfig,
    *,
    nm_grade_key: str,
) -> float | None:
    """``max(Discogs sale_stats median/avg, credible listing)`` — inference overlay path."""
    return reference_floor_usd(stats, cfg, nm_grade_key=nm_grade_key)


def _credible_listing_usd(
    stats: dict[str, Any],
    cfg: AnchorGuardrailsConfig,
    *,
    nm_grade_key: str,
    hist_stats: dict[str, float | None] | None = None,
) -> float | None:
    listing = _positive(stats.get("release_lowest_price"))
    if listing is None:
        return None
    check_stats = dict(stats)
    if hist_stats is not None:
        for key in (
            "sale_stats_low_usd",
            "sale_stats_high_usd",
            "sale_stats_median_usd",
            "sale_stats_average_usd",
        ):
            val = hist_stats.get(key)
            if val is not None:
                check_stats[key] = val
    if not listing_credible_for_caps(check_stats, cfg, nm_grade_key=nm_grade_key):
        return None
    return float(listing)


def reference_floor_training_usd(
    mp_row: dict[str, Any],
    sale_rows: list[dict[str, Any]],
    fetch_status: dict[str, Any] | None,
    *,
    nm_grade_key: str,
    ag_cfg: AnchorGuardrailsConfig,
) -> tuple[float | None, dict[str, Any]]:
    """
    Training reference: ``max(median, mean, credible listing)`` from sale_history quartiles.

    Aggregates all USD sales before ``t_ref`` (not NM-filtered). Sold nowcast ``s`` is excluded.
    """
    stats = marketplace_inference_stats_from_row(mp_row)
    listing = _credible_listing_usd(
        stats, ag_cfg, nm_grade_key=nm_grade_key, hist_stats=None
    )

    sh_ok = (
        fetch_status is not None
        and str(fetch_status.get("status") or "").strip().lower() == "ok"
    )
    sh_fetched = str(fetch_status.get("fetched_at") or "") if fetch_status else None
    t_ref = reference_time_t_ref(str(mp_row.get("fetched_at") or ""), sh_fetched)

    hist_stats: dict[str, float | None] = {
        "sale_stats_low_usd": None,
        "sale_stats_median_usd": None,
        "sale_stats_high_usd": None,
        "sale_stats_average_usd": None,
        "n_sales": 0,
    }
    if t_ref is not None and sh_ok and sale_rows:
        hist_stats = sale_stats_from_history(sale_rows, t_ref)

    if listing is None:
        listing = _credible_listing_usd(
            stats, ag_cfg, nm_grade_key=nm_grade_key, hist_stats=hist_stats
        )

    candidates: list[float] = []
    med = hist_stats.get("sale_stats_median_usd")
    avg = hist_stats.get("sale_stats_average_usd")
    if med is not None and med > 0:
        candidates.append(float(med))
    if avg is not None and avg > 0:
        candidates.append(float(avg))
    if listing is not None and listing > 0:
        candidates.append(float(listing))

    ref = max(candidates) if candidates else None
    diag: dict[str, Any] = {
        **hist_stats,
        "credible_listing_usd": listing,
        "sale_history_fetch_ok": bool(sh_ok),
        "reference_floor_training_usd": ref,
    }
    return ref, diag


def gate_outcomes_for_ref(
    ref: float | None,
    stats: dict[str, Any],
    cfg: AnchorGuardrailsConfig,
    *,
    nm_grade_key: str,
) -> dict[str, Any]:
    """Evaluate inflation / blend gates for a precomputed ``reference_floor``."""
    mx = max_price_suggestion_ladder_usd(stats)
    nm = nm_rung_usd(stats, nm_grade_key)
    mint = mint_rung_usd(stats, nm_grade_key=nm_grade_key)

    guardrails_active = ref is not None and ref > 0
    ratio_mx_ref: float | None = None
    if guardrails_active and mx is not None and mx > 0:
        ratio_mx_ref = float(mx) / float(ref)

    inflated = is_inflated_ladder(
        stats,
        cfg,
        nm_grade_key=nm_grade_key,
        reference_floor_usd_override=ref,
    )
    nm_for_ratio = nm if nm is not None and nm > 0 else mx
    ratio_diag = None
    blend_strength = 0.0
    if guardrails_active and nm_for_ratio is not None and nm_for_ratio > 0:
        ratio_diag = compute_ratio_sale_ladder(
            stats,
            cfg,
            grade_rung_usd=float(nm_for_ratio),
            reference_floor_usd_override=ref,
        )
        blend_strength = float(ratio_diag.blend_strength)
    blend_apply = blend_strength > 0.0

    out: dict[str, Any] = {
        "reference_floor_usd": ref,
        "mx_usd": mx,
        "nm_rung_usd": nm,
        "mint_rung_usd": mint,
        "ratio_mx_ref": ratio_mx_ref,
        "guardrails_active": guardrails_active,
        "is_inflated_ladder": inflated,
        "sale_stats_blend_apply": blend_apply,
        "blend_strength": blend_strength,
        "listing_credible": listing_credible_for_caps(
            stats, cfg, nm_grade_key=nm_grade_key
        ),
        "listing_credible_vs_sale_low": listing_credible_vs_sale_low(stats, cfg),
    }
    if ratio_diag is not None:
        out.update(diagnostics_to_dict(ratio_diag))
    return out
