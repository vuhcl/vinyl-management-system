"""Assemble ``releases_features`` row dicts shared by dump ingest and API ingest."""
from __future__ import annotations

from typing import Any


def release_row_from_fields(
    *,
    release_id: str,
    master_id: str | None,
    genre: str | None,
    style: str | None,
    decade: int,
    year: int,
    country: str | None,
    label_tier: int,
    is_original_pressing: bool | int,
    is_colored_vinyl: bool | int,
    is_picture_disc: bool | int,
    is_promo: bool | int,
    format_desc: str | None,
    artists_json: str | None,
    labels_json: str | None,
    genres_json: str | None,
    styles_json: str | None,
    formats_json: str | None,
) -> dict[str, Any]:
    """Canonical shape for ``FeatureStoreDB`` upserts (community counts live elsewhere)."""
    return {
        "release_id": release_id,
        "master_id": master_id,
        "genre": genre,
        "style": style,
        "decade": decade,
        "year": year,
        "country": country,
        "label_tier": label_tier,
        "is_original_pressing": is_original_pressing,
        "is_colored_vinyl": is_colored_vinyl,
        "is_picture_disc": is_picture_disc,
        "is_promo": is_promo,
        "format_desc": format_desc,
        "artists_json": artists_json,
        "labels_json": labels_json,
        "genres_json": genres_json,
        "styles_json": styles_json,
        "formats_json": formats_json,
    }
