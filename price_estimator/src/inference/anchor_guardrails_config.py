"""Anchor guardrails config parsed from VinylIQ YAML."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AnchorGuardrailsConfig:
    enabled: bool = False
    min_listing_usd: float = 50.0
    min_listing_to_nm_rung_ratio: float = 0.12
    min_listing_to_max_rung_ratio: float = 0.08
    inflated_max_rung_to_reference: float = 2.5
    mint_outlier_multiple_of_nm: float = 1.15
    ladder_rung_winsorize_multiple_of_nm: float = 0.0
    ratio_blend_enabled: bool = True
    ratio_blend_tolerance: float = 0.0
    ratio_blend_full_strength: float = 0.35
    ratio_outlier_low_threshold: float = 3.0
    min_n_sales_for_ratio: int = 2


def anchor_guardrails_config_from_raw(raw: dict[str, Any] | None) -> AnchorGuardrailsConfig:
    if not isinstance(raw, dict):
        return AnchorGuardrailsConfig()
    return AnchorGuardrailsConfig(
        enabled=bool(raw.get("enabled", False)),
        min_listing_usd=float(raw.get("min_listing_usd", 50.0)),
        min_listing_to_nm_rung_ratio=float(
            raw.get("min_listing_to_nm_rung_ratio", 0.12)
        ),
        min_listing_to_max_rung_ratio=float(
            raw.get("min_listing_to_max_rung_ratio", 0.08)
        ),
        inflated_max_rung_to_reference=float(
            raw.get("inflated_max_rung_to_reference", 2.5)
        ),
        mint_outlier_multiple_of_nm=float(
            raw.get("mint_outlier_multiple_of_nm", 1.15)
        ),
        ladder_rung_winsorize_multiple_of_nm=float(
            raw.get("ladder_rung_winsorize_multiple_of_nm", 0.0)
        ),
        ratio_blend_enabled=bool(raw.get("ratio_blend_enabled", True)),
        ratio_blend_tolerance=float(raw.get("ratio_blend_tolerance", 0.0)),
        ratio_blend_full_strength=float(raw.get("ratio_blend_full_strength", 0.35)),
        ratio_outlier_low_threshold=float(
            raw.get("ratio_outlier_low_threshold", 3.0)
        ),
        min_n_sales_for_ratio=int(raw.get("min_n_sales_for_ratio", 2)),
    )


def anchor_guardrails_config_from_vinyliq(v: dict[str, Any] | None) -> AnchorGuardrailsConfig:
    """Read ``vinyliq.anchor_guardrails`` with fallback to ``vinyliq.inference.anchor_guardrails``."""
    if not isinstance(v, dict):
        return AnchorGuardrailsConfig()
    raw = v.get("anchor_guardrails")
    if isinstance(raw, dict):
        return anchor_guardrails_config_from_raw(raw)
    inf = v.get("inference")
    if isinstance(inf, dict):
        nested = inf.get("anchor_guardrails")
        if isinstance(nested, dict):
            return anchor_guardrails_config_from_raw(nested)
    return AnchorGuardrailsConfig()
