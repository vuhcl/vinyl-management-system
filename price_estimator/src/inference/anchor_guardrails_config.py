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
    inflated_max_rung_to_sale_reference: float = 2.5
    max_ladder_to_reference: float = 5.0
    mint_outlier_multiple_of_nm: float = 1.15
    ladder_rung_winsorize_multiple_of_nm: float = 0.0


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
        inflated_max_rung_to_sale_reference=float(
            raw.get("inflated_max_rung_to_sale_reference", 2.5)
        ),
        max_ladder_to_reference=float(raw.get("max_ladder_to_reference", 5.0)),
        mint_outlier_multiple_of_nm=float(
            raw.get("mint_outlier_multiple_of_nm", 1.15)
        ),
        ladder_rung_winsorize_multiple_of_nm=float(
            raw.get("ladder_rung_winsorize_multiple_of_nm", 0.0)
        ),
    )
