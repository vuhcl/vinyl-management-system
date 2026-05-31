"""PS ladder guardrails (trim + log-blend anchor nudge) — shared by training and inference."""
from __future__ import annotations

import json
import math
from typing import Any

from price_estimator.src.inference.anchor_guardrails_config import AnchorGuardrailsConfig
from price_estimator.src.sale_floor.blend import sale_floor_blend_y_for_anchor
from price_estimator.src.storage.marketplace_db import (
    price_suggestions_ladder_from_json,
)
from price_estimator.src.training.sale_floor_blend_config import SaleFloorBlendConfig
from price_estimator.src.training.sale_floor_inference import (
    max_price_suggestion_ladder_usd,
)
from price_estimator.src.training.sale_floor_row_parsing import _parse_ps_grade, _positive


def _append_warning_once(warnings: list[str], code: str) -> None:
    if code not in warnings:
        warnings.append(code)


def listing_credible_for_caps(
    stats: dict[str, Any],
    cfg: AnchorGuardrailsConfig,
    *,
    nm_grade_key: str,
) -> bool:
    lo = _positive(stats.get("release_lowest_price"))
    if lo is None or lo < float(cfg.min_listing_usd):
        return False
    mx = max_price_suggestion_ladder_usd(stats)
    nm = _parse_ps_grade(stats.get("price_suggestions_json"), nm_grade_key)
    if mx is not None and mx > 0:
        if float(lo) < float(mx) * float(cfg.min_listing_to_max_rung_ratio):
            return False
    if nm is not None and nm > 0:
        if float(lo) < float(nm) * float(cfg.min_listing_to_nm_rung_ratio):
            return False
    return True


def reference_floor_usd(
    stats: dict[str, Any],
    cfg: AnchorGuardrailsConfig,
    *,
    nm_grade_key: str,
    reference_floor_usd_override: float | None = None,
) -> float | None:
    """``max(median, average, credible listing)`` from overlay or explicit override."""
    if reference_floor_usd_override is not None and reference_floor_usd_override > 0:
        return float(reference_floor_usd_override)

    listing = _positive(stats.get("release_lowest_price"))
    if listing is not None and not listing_credible_for_caps(
        stats, cfg, nm_grade_key=nm_grade_key
    ):
        listing = None
    candidates = [
        _positive(stats.get("sale_stats_median_usd")),
        _positive(stats.get("sale_stats_average_usd")),
        listing,
    ]
    vals = [float(x) for x in candidates if x is not None and x > 0]
    return max(vals) if vals else None


def nm_rung_usd(stats: dict[str, Any], nm_grade_key: str) -> float | None:
    v = _parse_ps_grade(stats.get("price_suggestions_json"), nm_grade_key)
    if v is not None and v > 0 and math.isfinite(float(v)):
        return float(v)
    return None


def mint_rung_usd(stats: dict[str, Any], *, nm_grade_key: str) -> float | None:
    ladder = price_suggestions_ladder_from_json(
        stats.get("price_suggestions_json")
        if isinstance(stats.get("price_suggestions_json"), str)
        else None
    )
    if not ladder:
        raw = stats.get("price_suggestions_json")
        if isinstance(raw, dict):
            ladder = {
                str(k): v if isinstance(v, dict) else {"value": v}
                for k, v in raw.items()
            }
    best: float | None = None
    for grade, entry in ladder.items():
        g = str(grade).strip().lower()
        if "near mint" in g or "(nm" in g:
            continue
        if "mint" not in g:
            continue
        val = _positive(entry.get("value") if isinstance(entry, dict) else entry)
        if val is not None and val > 0:
            if best is None or val > best:
                best = float(val)
    return best


def is_inflated_ladder(
    stats: dict[str, Any],
    cfg: AnchorGuardrailsConfig,
    *,
    nm_grade_key: str,
    reference_floor_usd_override: float | None = None,
) -> bool:
    ref = reference_floor_usd(
        stats,
        cfg,
        nm_grade_key=nm_grade_key,
        reference_floor_usd_override=reference_floor_usd_override,
    )
    mx = max_price_suggestion_ladder_usd(stats)
    if ref is None or mx is None or ref <= 0 or mx <= 0:
        return False
    if mx > ref * float(cfg.inflated_max_rung_to_reference):
        return True
    nm = nm_rung_usd(stats, nm_grade_key)
    if nm is not None and float(cfg.mint_outlier_multiple_of_nm) > 0:
        mint = mint_rung_usd(stats, nm_grade_key=nm_grade_key)
        if mint is not None and mint > nm * float(cfg.mint_outlier_multiple_of_nm):
            return True
    return False


def guardrails_active(
    stats: dict[str, Any],
    cfg: AnchorGuardrailsConfig,
    *,
    nm_grade_key: str,
    reference_floor_usd_override: float | None = None,
) -> bool:
    if not cfg.enabled:
        return False
    return (
        reference_floor_usd(
            stats,
            cfg,
            nm_grade_key=nm_grade_key,
            reference_floor_usd_override=reference_floor_usd_override,
        )
        is not None
    )


def sale_stats_blend_apply(
    stats: dict[str, Any],
    cfg: AnchorGuardrailsConfig,
    *,
    nm_grade_key: str,
    reference_floor_usd_override: float | None = None,
) -> bool:
    if not guardrails_active(
        stats,
        cfg,
        nm_grade_key=nm_grade_key,
        reference_floor_usd_override=reference_floor_usd_override,
    ):
        return False
    return is_inflated_ladder(
        stats,
        cfg,
        nm_grade_key=nm_grade_key,
        reference_floor_usd_override=reference_floor_usd_override,
    )


def trim_price_suggestions_json(
    stats: dict[str, Any],
    cfg: AnchorGuardrailsConfig,
    *,
    nm_grade_key: str,
) -> bool:
    nm = nm_rung_usd(stats, nm_grade_key)
    if nm is None or nm <= 0:
        return False
    raw = stats.get("price_suggestions_json")
    ladder = price_suggestions_ladder_from_json(
        raw if isinstance(raw, str) else None
    )
    if not ladder and isinstance(raw, dict):
        ladder = {
            str(k): dict(v) if isinstance(v, dict) else {"value": v, "currency": "USD"}
            for k, v in raw.items()
        }
    if not ladder:
        return False

    changed = False
    mint_mult = float(cfg.mint_outlier_multiple_of_nm)
    mint_cap = nm * mint_mult if mint_mult > 0 else None
    winsor_mult = float(cfg.ladder_rung_winsorize_multiple_of_nm)
    winsor_cap = nm * winsor_mult if winsor_mult > 0 else None

    for grade, entry in ladder.items():
        if not isinstance(entry, dict):
            continue
        val = _positive(entry.get("value"))
        if val is None or val <= 0:
            continue
        g_lower = str(grade).strip().lower()
        new_val: float | None = None
        if (
            mint_cap is not None
            and "mint" in g_lower
            and "near mint" not in g_lower
            and "(nm" not in g_lower
        ):
            if val > mint_cap:
                new_val = float(mint_cap)
        if winsor_cap is not None and val > winsor_cap:
            cap_v = float(winsor_cap)
            new_val = cap_v if new_val is None else min(new_val, cap_v)
        if new_val is not None and abs(new_val - val) > 1e-6:
            entry["value"] = new_val
            changed = True

    if changed:
        stats["price_suggestions_json"] = json.dumps(
            ladder, ensure_ascii=False, separators=(",", ":")
        )
    return changed


def prepare_stats_for_serving(
    stats: dict[str, Any],
    cfg: AnchorGuardrailsConfig,
    *,
    nm_grade_key: str,
) -> list[str]:
    warnings: list[str] = []
    if not cfg.enabled:
        return warnings
    ref = reference_floor_usd(stats, cfg, nm_grade_key=nm_grade_key)
    if ref is None:
        warnings.append("anchor_guardrails_skipped_no_reference")
        return warnings
    if is_inflated_ladder(stats, cfg, nm_grade_key=nm_grade_key):
        if trim_price_suggestions_json(stats, cfg, nm_grade_key=nm_grade_key):
            _append_warning_once(warnings, "anchor_guardrail_applied")
    return warnings


# Backward-compatible alias
prepare_stats_for_inference = prepare_stats_for_serving


def blend_path_anchor_usd(
    grade_rung_usd: float,
    stats: dict[str, Any],
    cfg: AnchorGuardrailsConfig,
    blend_cfg: SaleFloorBlendConfig,
    *,
    nm_grade_key: str,
    warnings: list[str] | None = None,
    reference_floor_usd_override: float | None = None,
) -> float:
    rung = float(grade_rung_usd)
    if rung <= 0 or not math.isfinite(rung):
        return rung
    if not sale_stats_blend_apply(
        stats,
        cfg,
        nm_grade_key=nm_grade_key,
        reference_floor_usd_override=reference_floor_usd_override,
    ):
        return rung
    ref = reference_floor_usd(
        stats,
        cfg,
        nm_grade_key=nm_grade_key,
        reference_floor_usd_override=reference_floor_usd_override,
    )
    if ref is None:
        return rung
    blended = sale_floor_blend_y_for_anchor(ref, rung, "A", cfg=blend_cfg)
    if warnings is not None:
        _append_warning_once(warnings, "anchor_guardrail_applied")
    if blended is not None and blended > 0 and math.isfinite(float(blended)):
        return float(blended)
    return rung
