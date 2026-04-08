"""Request/response models for VinylIQ API."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class EstimateRequest(BaseModel):
    release_id: str = Field(..., description="Discogs release ID")
    media_condition: str | None = None
    sleeve_condition: str | None = None
    refresh_stats: bool = Field(
        False, description="If true, bypass cache and call Discogs API"
    )


class EstimateResponse(BaseModel):
    release_id: str
    estimated_price: float | None = None
    confidence_interval: list[float] = Field(default_factory=list)
    baseline_median: float | None = None
    model_version: str = ""
    status: str = ""


class CollectionItem(BaseModel):
    release_id: str
    media_condition: str | None = None
    sleeve_condition: str | None = None


class CollectionValueRequest(BaseModel):
    username: str = Field(
        ...,
        description="Discogs username (for audit/logging only at MVP; no OAuth yet)",
    )
    items: list[CollectionItem]


class CollectionValueResponse(BaseModel):
    total_estimated_value: float
    per_item_breakdown: list[dict[str, Any]]


class HealthResponse(BaseModel):
    status: str = "ok"
    feature_store_count: int | None = None
    model_loaded: bool = False
