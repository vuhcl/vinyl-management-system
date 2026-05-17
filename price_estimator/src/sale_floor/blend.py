"""Log-blend of sold signal and listing floor (shared by training and inference)."""
from __future__ import annotations

import math
from typing import Protocol


class SaleFloorLogBlendConfig(Protocol):
    w_base: float
    w_min: float
    w_max: float
    tier_b_delta: float
    tier_c_delta: float
    gap_epsilon_log: float
    gap_k_down: float
    gap_k_up: float
    gap_delta_cap: float


def blend_weight_w_eff(
    *,
    s: float,
    lo: float,
    tier: str,
    cfg: SaleFloorLogBlendConfig,
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
    cfg: SaleFloorLogBlendConfig,
) -> float | None:
    """Log-blend; no ``lo`` → ``y = s``; no ``s`` → None."""
    if s is not None and s > 0 and lo is not None and lo > 0:
        w_eff = blend_weight_w_eff(s=s, lo=lo, tier=tier, cfg=cfg)
        return float(math.exp(w_eff * math.log(s) + (1.0 - w_eff) * math.log(lo)))
    if s is not None and s > 0:
        return float(s)
    if lo is not None and lo > 0:
        return float(lo)
    return None
