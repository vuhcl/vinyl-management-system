"""Ratio-based PS ladder blend strength (continuous guardrails)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

from price_estimator.src.inference.anchor_guardrails_config import AnchorGuardrailsConfig
from price_estimator.src.training.sale_floor_row_parsing import _positive

BlendDirection = Literal["down", "up", "none"]


@dataclass(frozen=True)
class RatioBlendDiagnostics:
    R_sale: float | None
    R_ladder: float | None
    spread_floor: float | None
    skew: float | None
    left_skew: bool
    skip_down_blend: bool
    blend_direction: BlendDirection
    reference_ref_usd: float | None
    ratio_gap: float | None
    blend_strength: float
    listing_credible_vs_sale_low: bool
    ratio_blend_fallback: bool
    sale_stats_average_missing: bool


def listing_credible_vs_sale_low(
    stats: dict[str, Any],
    cfg: AnchorGuardrailsConfig,
) -> bool:
    """Listing credible for reference / spread_floor when sale quartiles exist."""
    listing = _positive(stats.get("release_lowest_price"))
    if listing is None or listing < float(cfg.min_listing_usd):
        return False
    sale_low = _positive(stats.get("sale_stats_low_usd"))
    if sale_low is None:
        return True
    if listing < sale_low:
        return False
    sale_high = _positive(stats.get("sale_stats_high_usd"))
    if sale_high is not None and listing > sale_high:
        return False
    return True


def _sale_low_credible_for_denom(
    median: float,
    sale_low: float,
    *,
    left_skew: bool,
    outlier_threshold: float,
) -> bool:
    if sale_low <= 0 or median <= 0:
        return False
    if left_skew and median / sale_low > outlier_threshold:
        return False
    return True


def spread_floor_denom_usd(
    stats: dict[str, Any],
    cfg: AnchorGuardrailsConfig,
    *,
    left_skew: bool,
) -> float | None:
    median = _positive(stats.get("sale_stats_median_usd"))
    if median is None or median <= 0:
        return None
    listing_cred = listing_credible_vs_sale_low(stats, cfg)
    if listing_cred:
        listing = _positive(stats.get("release_lowest_price"))
        if listing is not None and listing > 0:
            return float(listing)
    sale_low = _positive(stats.get("sale_stats_low_usd"))
    if sale_low is not None and _sale_low_credible_for_denom(
        float(median),
        float(sale_low),
        left_skew=left_skew,
        outlier_threshold=float(cfg.ratio_outlier_low_threshold),
    ):
        return float(sale_low)
    return float(median)


def reference_ref_usd(
    stats: dict[str, Any],
    cfg: AnchorGuardrailsConfig,
    *,
    reference_floor_usd_override: float | None = None,
) -> float | None:
    if reference_floor_usd_override is not None and reference_floor_usd_override > 0:
        return float(reference_floor_usd_override)
    median = _positive(stats.get("sale_stats_median_usd"))
    avg = _positive(stats.get("sale_stats_average_usd"))
    listing = _positive(stats.get("release_lowest_price"))
    if listing is not None and not listing_credible_vs_sale_low(stats, cfg):
        listing = None
    candidates: list[float] = []
    if median is not None and median > 0:
        candidates.append(float(median))
    if avg is not None and avg > 0:
        candidates.append(float(avg))
    if listing is not None and listing > 0:
        candidates.append(float(listing))
    return max(candidates) if candidates else None


def ratio_mode_inputs_sufficient(
    stats: dict[str, Any],
    cfg: AnchorGuardrailsConfig,
) -> bool:
    median = _positive(stats.get("sale_stats_median_usd"))
    avg = _positive(stats.get("sale_stats_average_usd"))
    n_sales = stats.get("n_sales")
    if median is None or median <= 0 or avg is None or avg <= 0:
        return False
    if spread_floor_denom_usd(stats, cfg, left_skew=avg < median) is None:
        return False
    if n_sales is not None:
        try:
            if int(n_sales) < int(cfg.min_n_sales_for_ratio):
                return False
        except (TypeError, ValueError):
            pass
    return True


def _strength_from_gap(
    ratio_gap: float,
    *,
    denom: float,
    full_strength: float,
    tolerance: float,
    direction: BlendDirection,
) -> float:
    if direction == "none" or denom <= 0 or full_strength <= 0:
        return 0.0
    if direction == "down":
        if ratio_gap <= float(tolerance) * max(0.0, denom):
            return 0.0
        excess = ratio_gap / denom
    else:
        if abs(ratio_gap) <= float(tolerance) * max(0.0, denom):
            return 0.0
        excess = abs(ratio_gap) / denom
    return float(min(1.0, max(0.0, excess / full_strength)))


def compute_ratio_sale_ladder(
    stats: dict[str, Any],
    cfg: AnchorGuardrailsConfig,
    *,
    grade_rung_usd: float,
    reference_floor_usd_override: float | None = None,
    use_legacy_fallback: bool = False,
) -> RatioBlendDiagnostics:
    """Compute ``R_sale``, ``R_ladder``, and continuous ``blend_strength``."""
    avg = _positive(stats.get("sale_stats_average_usd"))
    median = _positive(stats.get("sale_stats_median_usd"))
    sale_low = _positive(stats.get("sale_stats_low_usd"))
    listing_cred = listing_credible_vs_sale_low(stats, cfg)
    ref = reference_ref_usd(
        stats,
        cfg,
        reference_floor_usd_override=reference_floor_usd_override,
    )
    rung = float(grade_rung_usd)
    avg_missing = avg is None or avg <= 0

    if ref is None or ref <= 0 or rung <= 0 or not math.isfinite(rung):
        return RatioBlendDiagnostics(
            R_sale=None,
            R_ladder=None,
            spread_floor=None,
            skew=None,
            left_skew=False,
            skip_down_blend=False,
            blend_direction="none",
            reference_ref_usd=ref,
            ratio_gap=None,
            blend_strength=0.0,
            listing_credible_vs_sale_low=listing_cred,
            ratio_blend_fallback=use_legacy_fallback or avg_missing,
            sale_stats_average_missing=avg_missing,
        )

    fallback = use_legacy_fallback or avg_missing or not ratio_mode_inputs_sufficient(
        stats, cfg
    )
    R_ladder = rung / float(ref)

    if fallback:
        R_sale = float(cfg.inflated_max_rung_to_reference)
        spread_floor = None
        skew = None
        left_skew = False
        skip_down = False
        ratio_gap = R_ladder - R_sale
        direction: BlendDirection = "none"
        tol = float(cfg.ratio_blend_tolerance)
        full = float(cfg.ratio_blend_full_strength)
        if ratio_gap > 0 and rung > ref:
            direction = "down"
            strength = _strength_from_gap(
                ratio_gap,
                denom=R_ladder,
                full_strength=full,
                tolerance=tol,
                direction="down",
            )
        elif ratio_gap < 0 and rung < ref:
            direction = "up"
            strength = _strength_from_gap(
                ratio_gap,
                denom=R_sale,
                full_strength=full,
                tolerance=tol,
                direction="up",
            )
        else:
            strength = 0.0
        return RatioBlendDiagnostics(
            R_sale=R_sale,
            R_ladder=R_ladder,
            spread_floor=spread_floor,
            skew=skew,
            left_skew=left_skew,
            skip_down_blend=skip_down,
            blend_direction=direction,
            reference_ref_usd=ref,
            ratio_gap=ratio_gap,
            blend_strength=strength,
            listing_credible_vs_sale_low=listing_cred,
            ratio_blend_fallback=True,
            sale_stats_average_missing=avg_missing,
        )

    assert avg is not None and median is not None
    left_skew = float(avg) < float(median)
    sale_low_outlier = (
        sale_low is not None
        and median > 0
        and float(sale_low) > 0
        and float(median) / float(sale_low)
        > float(cfg.ratio_outlier_low_threshold)
    )
    denom = spread_floor_denom_usd(stats, cfg, left_skew=left_skew)
    if denom is None or denom <= 0:
        return compute_ratio_sale_ladder(
            stats,
            cfg,
            grade_rung_usd=grade_rung_usd,
            reference_floor_usd_override=reference_floor_usd_override,
            use_legacy_fallback=True,
        )

    spread_floor = float(median) / float(denom)
    skew_for_rs = 1.0 if left_skew else float(avg) / float(median)
    R_sale = spread_floor * skew_for_rs
    skip_down = (left_skew and spread_floor < 1.0) or (
        sale_low_outlier and spread_floor < 1.0 and listing_cred
    )
    ratio_gap = R_ladder - R_sale
    tol = float(cfg.ratio_blend_tolerance)
    full = float(cfg.ratio_blend_full_strength)
    direction = "none"
    strength = 0.0
    if ratio_gap > 0 and rung > ref and not skip_down:
        direction = "down"
        strength = _strength_from_gap(
            ratio_gap,
            denom=R_ladder,
            full_strength=full,
            tolerance=tol,
            direction="down",
        )
    elif ratio_gap < 0 and rung < ref:
        direction = "up"
        strength = _strength_from_gap(
            ratio_gap,
            denom=R_sale,
            full_strength=full,
            tolerance=tol,
            direction="up",
        )

    return RatioBlendDiagnostics(
        R_sale=R_sale,
        R_ladder=R_ladder,
        spread_floor=spread_floor,
        skew=skew_for_rs,
        left_skew=left_skew,
        skip_down_blend=skip_down,
        blend_direction=direction,
        reference_ref_usd=ref,
        ratio_gap=ratio_gap,
        blend_strength=strength,
        listing_credible_vs_sale_low=listing_cred,
        ratio_blend_fallback=False,
        sale_stats_average_missing=False,
    )


def blend_strength_for_grade(
    stats: dict[str, Any],
    cfg: AnchorGuardrailsConfig,
    *,
    grade_rung_usd: float,
    reference_floor_usd_override: float | None = None,
) -> tuple[float, RatioBlendDiagnostics]:
    """Return ``(strength, diagnostics)`` for a grade PS rung."""
    if not cfg.enabled:
        diag = compute_ratio_sale_ladder(
            stats,
            cfg,
            grade_rung_usd=grade_rung_usd,
            reference_floor_usd_override=reference_floor_usd_override,
            use_legacy_fallback=True,
        )
        return 0.0, diag

    if not cfg.ratio_blend_enabled:
        from price_estimator.src.sale_floor.anchor_guardrails import is_inflated_ladder

        inflated = is_inflated_ladder(
            stats,
            cfg,
            nm_grade_key="Near Mint (NM or M-)",
            reference_floor_usd_override=reference_floor_usd_override,
        )
        strength = 1.0 if inflated else 0.0
        diag = compute_ratio_sale_ladder(
            stats,
            cfg,
            grade_rung_usd=grade_rung_usd,
            reference_floor_usd_override=reference_floor_usd_override,
            use_legacy_fallback=True,
        )
        return strength, RatioBlendDiagnostics(
            **{**diag.__dict__, "blend_strength": strength}
        )

    diag = compute_ratio_sale_ladder(
        stats,
        cfg,
        grade_rung_usd=grade_rung_usd,
        reference_floor_usd_override=reference_floor_usd_override,
    )
    return float(diag.blend_strength), diag


def diagnostics_to_dict(diag: RatioBlendDiagnostics) -> dict[str, Any]:
    return {
        "R_sale": diag.R_sale,
        "R_ladder": diag.R_ladder,
        "spread_floor": diag.spread_floor,
        "skew": diag.skew,
        "left_skew": diag.left_skew,
        "skip_down_blend": diag.skip_down_blend,
        "blend_direction": diag.blend_direction,
        "reference_ref_usd": diag.reference_ref_usd,
        "ratio_gap": diag.ratio_gap,
        "blend_strength": diag.blend_strength,
        "listing_credible_vs_sale_low": diag.listing_credible_vs_sale_low,
        "ratio_blend_fallback": diag.ratio_blend_fallback,
        "sale_stats_average_missing": diag.sale_stats_average_missing,
    }
