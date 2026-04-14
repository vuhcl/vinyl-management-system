"""§7.1d sold nowcast ``s`` + listing floor blend for ``training_label.mode: sale_floor_blend``."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable

import numpy as np
from scipy.stats import theilslopes

from price_estimator.src.storage.marketplace_db import price_suggestion_values_by_grade


def _parse_ps_grade(raw_json: str | None, grade_key: str) -> float | None:
    from price_estimator.src.training.label_synthesis import parse_price_suggestion_value

    return parse_price_suggestion_value(raw_json, grade_key)


def _positive(v: Any) -> float | None:
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if x > 0 else None


def parse_iso_datetime(s: str | None) -> datetime | None:
    if s is None or not str(s).strip():
        return None
    t = str(s).strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(t)
    except ValueError:
        return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def reference_time_t_ref(
    mp_fetched_at: str | None,
    sh_fetch_fetched_at: str | None,
) -> datetime | None:
    """§7.1d: ``min(MP.fetched_at, SH_fetch.fetched_at)`` when both exist; else whichever is set."""
    mp = parse_iso_datetime(mp_fetched_at)
    sh = parse_iso_datetime(sh_fetch_fetched_at)
    if mp is not None and sh is not None:
        return min(mp, sh)
    return mp or sh


def sale_row_usd(row: dict[str, Any]) -> float | None:
    v = row.get("price_user_usd_approx")
    if v is not None:
        p = _positive(v)
        if p is not None:
            return p
    for key in ("price_user_currency_text", "price_original_text"):
        raw = row.get(key)
        if raw is None or not str(raw).strip():
            continue
        m = re.search(r"[\d,]+\.?\d*", str(raw).replace(",", ""))
        if not m:
            continue
        try:
            x = float(m.group(0))
        except ValueError:
            continue
        if x > 0:
            return x
    return None


def _nm_allowed(
    media: str | None,
    sleeve: str | None,
    *,
    nm_substrings: tuple[str, ...],
) -> bool:
    blob = f"{media or ''} {sleeve or ''}".lower()
    return any(s.lower() in blob for s in nm_substrings)


@dataclass(frozen=True)
class SaleFloorBlendConfig:
    n_min_trend: int = 8
    recency_half_life_days: float = 365.0
    w_base: float = 0.55
    w_min: float = 0.2
    w_max: float = 0.9
    tier_b_delta: float = 0.05
    tier_c_delta: float = 0.1
    gap_epsilon_log: float = 0.02
    gap_k_down: float = 0.15
    gap_k_up: float = 0.12
    gap_delta_cap: float = 0.5
    nm_substrings: tuple[str, ...] = (
        "near mint",
        "(nm",
        "mint (m",
    )


def eligible_nm_sale_rows(
    rows: Iterable[dict[str, Any]],
    t_ref: datetime,
    *,
    cfg: SaleFloorBlendConfig,
) -> list[tuple[datetime, float]]:
    out: list[tuple[datetime, float]] = []
    for r in rows:
        if not _nm_allowed(r.get("media_condition"), r.get("sleeve_condition"), nm_substrings=cfg.nm_substrings):
            continue
        price = sale_row_usd(r)
        if price is None:
            continue
        od = parse_iso_datetime(str(r.get("order_date") or ""))
        if od is None or od > t_ref:
            continue
        out.append((od, float(price)))
    out.sort(key=lambda x: (x[0], x[1]))
    return out


def sold_nowcast_s(
    eligible: list[tuple[datetime, float]],
    t_ref: datetime,
    *,
    cfg: SaleFloorBlendConfig,
) -> tuple[float | None, str, int]:
    """
    Tier A/B/C sold-dollar nowcast ``s`` at ``t_ref``.

    Returns ``(s_or_none, tier_name, n_eligible)``.
    """
    n = len(eligible)
    if n == 0:
        return None, "none", 0
    prices = np.array([p for _, p in eligible], dtype=np.float64)
    dates = [d for d, _ in eligible]
    t0 = min(dates)
    xs = np.array([(d - t0).total_seconds() / 86400.0 for d in dates], dtype=np.float64)
    t_x = (t_ref - t0).total_seconds() / 86400.0
    ys = np.log(prices)

    if n >= cfg.n_min_trend and float(np.std(xs)) > 1e-9:
        try:
            res = theilslopes(ys, xs)
            intercept = float(res.intercept)
            slope = float(res.slope)
            log_s = intercept + slope * float(t_x)
            s = float(math.exp(log_s))
            if s > 0 and math.isfinite(s):
                return s, "A", n
        except (ValueError, RuntimeError):
            pass

    if n >= 3:
        ages_days = np.array([(t_ref - d).total_seconds() / 86400.0 for d in dates], dtype=np.float64)
        H = max(1.0, float(cfg.recency_half_life_days))
        w = np.exp(-np.maximum(ages_days, 0.0) / H)
        sw = float(np.sum(w))
        if sw <= 0:
            med = float(np.median(prices))
            return med if med > 0 else None, "B", n
        yb = float(np.sum(prices * w) / sw)
        return yb if yb > 0 else None, "B", n

    last_p = float(prices[-1])
    return last_p if last_p > 0 else None, "C", n


def effective_listing_floor_lo(row: dict[str, Any]) -> float | None:
    return _positive(row.get("release_lowest_price")) or _positive(row.get("lowest_price"))


def max_price_suggestion_ladder_usd(row: dict[str, Any]) -> float | None:
    """§7.1d residual anchor: max positive grade value from ``price_suggestions_json``."""
    vals = price_suggestion_values_by_grade(row.get("price_suggestions_json"))
    if not vals:
        return None
    return max(vals.values())


def residual_anchor_m_full_data(
    row: dict[str, Any],
    *,
    nm_grade_key: str,
) -> float | None:
    """
    ``m`` when sale history exists (§7.1d): max PS ladder → NM grade → listing floor.
    """
    mx = max_price_suggestion_ladder_usd(row)
    if mx is not None:
        return mx
    y_nm = _parse_ps_grade(row.get("price_suggestions_json"), nm_grade_key)
    if y_nm is not None:
        return float(y_nm)
    lo = effective_listing_floor_lo(row)
    if lo is not None:
        return float(lo)
    return None


def residual_anchor_m_no_sale_history(row: dict[str, Any], *, nm_grade_key: str) -> float | None:
    """§7.1b-style: ``lo``-first; then NM suggestion; avoid PS max ladder when no SH."""
    lo = effective_listing_floor_lo(row)
    if lo is not None:
        return float(lo)
    y_nm = _parse_ps_grade(row.get("price_suggestions_json"), nm_grade_key)
    if y_nm is not None:
        return float(y_nm)
    return None


def blend_weight_w_eff(
    *,
    s: float,
    lo: float,
    tier: str,
    cfg: SaleFloorBlendConfig,
) -> float:
    w = float(cfg.w_base)
    if tier == "B":
        w -= float(cfg.tier_b_delta)
    elif tier == "C":
        w -= float(cfg.tier_c_delta)
    w = max(float(cfg.w_min), min(float(cfg.w_max), w))

    eps = float(cfg.gap_epsilon_log)
    d_log = math.log(lo) - math.log(s)
    if d_log < -eps:
        w += float(cfg.gap_k_down) * min(abs(d_log), float(cfg.gap_delta_cap))
    elif d_log > eps:
        w -= float(cfg.gap_k_up) * min(d_log, float(cfg.gap_delta_cap))

    return max(float(cfg.w_min), min(float(cfg.w_max), w))


def sale_floor_blend_y(
    s: float | None,
    lo: float | None,
    tier: str,
    *,
    cfg: SaleFloorBlendConfig,
) -> float | None:
    """§7.1d log-blend; no ``lo`` → ``y = s``; no ``s`` → None (caller excludes or uses other mode)."""
    if s is not None and s > 0 and lo is not None and lo > 0:
        w_eff = blend_weight_w_eff(s=s, lo=lo, tier=tier, cfg=cfg)
        return float(math.exp(w_eff * math.log(s) + (1.0 - w_eff) * math.log(lo)))
    if s is not None and s > 0:
        return float(s)
    if lo is not None and lo > 0:
        return float(lo)
    return None


def sale_floor_blend_bundle(
    mp_row: dict[str, Any],
    sale_rows: list[dict[str, Any]],
    fetch_status: dict[str, Any] | None,
    *,
    sf_cfg: dict[str, Any],
    nm_grade_key: str,
) -> tuple[float | None, float | None, dict[str, float]]:
    """
    Returns ``(y_label, m_anchor, x_flags)``.

    ``x_flags`` includes ``has_sale_history``, ``s_imputed``, ``has_listing_floor`` (0/1 floats).
    """
    raw_cfg = sf_cfg if isinstance(sf_cfg, dict) else {}
    nm_tup = raw_cfg.get("nm_substrings")
    nm_sub: tuple[str, ...] = SaleFloorBlendConfig.nm_substrings
    if isinstance(nm_tup, (list, tuple)) and nm_tup:
        nm_sub = tuple(str(x) for x in nm_tup)
    cfg = SaleFloorBlendConfig(
        n_min_trend=int(raw_cfg.get("n_min_trend", 8)),
        recency_half_life_days=float(raw_cfg.get("recency_half_life_days", 365.0)),
        w_base=float(raw_cfg.get("w_base", 0.55)),
        w_min=float(raw_cfg.get("w_min", 0.2)),
        w_max=float(raw_cfg.get("w_max", 0.9)),
        tier_b_delta=float(raw_cfg.get("tier_b_delta", 0.05)),
        tier_c_delta=float(raw_cfg.get("tier_c_delta", 0.1)),
        gap_epsilon_log=float(raw_cfg.get("gap_epsilon_log", 0.02)),
        gap_k_down=float(raw_cfg.get("gap_k_down", 0.15)),
        gap_k_up=float(raw_cfg.get("gap_k_up", 0.12)),
        gap_delta_cap=float(raw_cfg.get("gap_delta_cap", 0.5)),
        nm_substrings=nm_sub,
    )

    lo = effective_listing_floor_lo(mp_row)
    has_listing_floor = 1.0 if lo is not None and lo > 0 else 0.0

    sh_ok = fetch_status is not None and str(fetch_status.get("status") or "").strip().lower() == "ok"
    sh_fetched = str(fetch_status.get("fetched_at") or "") if fetch_status else None
    t_ref = reference_time_t_ref(str(mp_row.get("fetched_at") or ""), sh_fetched)

    s: float | None = None
    tier = "none"
    n_elig = 0
    s_imputed = 0.0
    has_sale_history = 0.0

    if t_ref is not None and sh_ok:
        elig = eligible_nm_sale_rows(sale_rows, t_ref, cfg=cfg)
        s, tier, n_elig = sold_nowcast_s(elig, t_ref, cfg=cfg)
        if s is not None and s > 0:
            has_sale_history = 1.0

    y = sale_floor_blend_y(s, lo, tier, cfg=cfg)
    if y is None:
        return None, None, {
            "has_sale_history": has_sale_history,
            "s_imputed": s_imputed,
            "has_listing_floor": has_listing_floor,
        }

    if has_sale_history:
        m = residual_anchor_m_full_data(mp_row, nm_grade_key=nm_grade_key)
    else:
        m = residual_anchor_m_no_sale_history(mp_row, nm_grade_key=nm_grade_key)
    if m is None:
        m = y

    return float(y), float(m), {
        "has_sale_history": has_sale_history,
        "s_imputed": s_imputed,
        "has_listing_floor": has_listing_floor,
    }
