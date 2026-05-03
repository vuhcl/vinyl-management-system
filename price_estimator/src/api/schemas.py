"""Request/response models for VinylIQ API."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class MarketplaceClientSnapshot(BaseModel):
    """
    Optional Discogs-visible marketplace scalars scraped or read from the
    release/marketplace page. When set, they override server/Redis-backed
    values so anchors and residual reconstruction match browsing context.

    ``price_suggestions_json`` mirrors the Discogs price-suggestions ladder
    (grade → {{value,currency}}, …): pass the raw JSON string or a dict matching
    that shape.

    ``sale_stats_*_usd`` optional fields carry Discogs **recent sale**
    quartet (average, median, high, low USD) scraped from the listing — distinct
    from marketplace listing-floor anchors.
    """

    release_lowest_price: float | None = None
    num_for_sale: int | None = None
    release_num_for_sale: int | None = None
    community_want: int | None = None
    community_have: int | None = None
    blocked_from_sale: bool | int | None = None
    price_suggestions_json: str | dict[str, Any] | None = None
    sale_stats_average_usd: float | None = None
    sale_stats_median_usd: float | None = None
    sale_stats_high_usd: float | None = None
    sale_stats_low_usd: float | None = None


class EstimateRequest(BaseModel):
    release_id: str = Field(..., description="Discogs release ID")
    media_condition: str | None = None
    sleeve_condition: str | None = None
    refresh_stats: bool = Field(
        False,
        description=(
            "If true, bypass Redis/DB read-through and refetch Discogs "
            "(GET /releases/{id} + GET /marketplace/price_suggestions/{id}, "
            "same pair as marketplace collector full mode)"
        ),
    )
    marketplace_client: MarketplaceClientSnapshot | None = Field(
        default=None,
        description="Overrides from client-side scrape (listing floor, PS ladder, depth)",
    )


class EstimateResponse(BaseModel):
    release_id: str
    estimated_price: float | None = None
    confidence_interval: list[float] = Field(default_factory=list)
    baseline_median: float | None = None
    model_version: str = ""
    status: str = ""
    num_for_sale: int = 0
    warnings: list[str] = Field(default_factory=list)
    residual_anchor_usd: float | None = Field(
        default=None,
        description=(
            "USD anchor backing residual reconstruction. With "
            "``use_price_suggestion_condition_anchor`` (default in base config): "
            "mean of the media-rung and sleeve-rung ``price_suggestions`` anchors "
            "for this request's grades (falls back per side to training ``m`` / listing). "
            "When that mode is disabled: ``m`` mirrors training blend (NM ladder cascade / listing), "
            "not necessarily the ladder rungs for the caller's conditions."
        ),
    )


class CollectionItem(BaseModel):
    release_id: str
    media_condition: str | None = None
    sleeve_condition: str | None = None
    marketplace_client: MarketplaceClientSnapshot | None = None


class CollectionValueRequest(BaseModel):
    username: str = Field(
        ...,
        description="Discogs username (for audit/logging only at MVP; no OAuth yet)",
    )
    items: list[CollectionItem]


class CollectionValueResponse(BaseModel):
    total_estimated_value: float
    per_item_breakdown: list[dict[str, Any]]


class InvalidateMarketplaceCacheResponse(BaseModel):
    release_id: str
    redis_cache_enabled: bool = Field(
        ...,
        description="False when REDIS_HOST unset or Redis unreachable at startup",
    )


class HealthResponse(BaseModel):
    status: str = "ok"
    feature_store_count: int | None = None
    model_loaded: bool = False
    model_source: str = "local"
