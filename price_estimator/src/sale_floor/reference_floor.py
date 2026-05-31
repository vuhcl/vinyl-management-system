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
) -> float | None:
    listing = _positive(stats.get("release_lowest_price"))
    if listing is None:
        return None
    if not listing_credible_for_caps(stats, cfg, nm_grade_key=nm_grade_key):
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
    listing = _credible_listing_usd(stats, ag_cfg, nm_grade_key=nm_grade_key)

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
    blend_apply = sale_stats_blend_apply(
        stats,
        cfg,
        nm_grade_key=nm_grade_key,
        reference_floor_usd_override=ref,
    )

    return {
        "reference_floor_usd": ref,
        "mx_usd": mx,
        "nm_rung_usd": nm,
        "mint_rung_usd": mint,
        "ratio_mx_ref": ratio_mx_ref,
        "guardrails_active": guardrails_active,
        "is_inflated_ladder": inflated,
        "sale_stats_blend_apply": blend_apply,
        "listing_credible": listing_credible_for_caps(
            stats, cfg, nm_grade_key=nm_grade_key
        ),
    }
