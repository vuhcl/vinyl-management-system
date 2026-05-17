"""Apply anchor guardrails to residual reconstruction anchors."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from price_estimator.src.training.sale_floor_blend_config import SaleFloorBlendConfig

from .anchor_guardrails import blend_path_anchor_usd, guardrails_active
from .anchor_guardrails_config import AnchorGuardrailsConfig


@dataclass
class AnchorGuardrailsContext:
    cfg: AnchorGuardrailsConfig
    blend_cfg: SaleFloorBlendConfig
    warnings: list[str]


def resolve_residual_anchor_usd(
    *,
    stats: dict[str, Any],
    raw_anchor_usd: float,
    guardrails_ctx: AnchorGuardrailsContext | None,
    nm_grade_key: str,
) -> float:
    """Blend per-path or fallback anchor when guardrails apply."""
    anchor = float(raw_anchor_usd)
    if guardrails_ctx is None or not guardrails_ctx.cfg.enabled:
        return anchor
    if not guardrails_active(
        stats, guardrails_ctx.cfg, nm_grade_key=nm_grade_key
    ):
        if "anchor_guardrails_skipped_no_reference" not in guardrails_ctx.warnings:
            guardrails_ctx.warnings.append("anchor_guardrails_skipped_no_reference")
        return anchor
    return blend_path_anchor_usd(
        anchor,
        stats,
        guardrails_ctx.cfg,
        guardrails_ctx.blend_cfg,
        nm_grade_key=nm_grade_key,
        warnings=guardrails_ctx.warnings,
    )
