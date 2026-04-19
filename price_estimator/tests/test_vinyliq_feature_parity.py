"""Train/serve feature alignment for VinylIQ (residual target)."""
from __future__ import annotations

from price_estimator.src.features.vinyliq_features import (
    residual_training_feature_columns,
    row_dict_for_inference,
)
from price_estimator.src.training.train_vinyliq import (
    residual_z_clip_abs_from_vinyliq,
)


def test_residual_row_dict_ignores_listing_dollar_scalars() -> None:
    """Residual X zeros listing-dollar scalars; depth/community match when stats match."""
    cat: dict = {
        "genre": "rock",
        "country": "us",
        "year": 1975,
        "decade": 1970,
        "is_original_pressing": 1,
        "label_tier": 2.0,
        "is_colored_vinyl": 0,
        "is_picture_disc": 0,
        "is_promo": 0,
        "formats_json": '["Vinyl", "LP"]',
        "format_desc": "",
    }
    # Same non-dollar marketplace snapshot; only dollar fields differ (zeroed in residual).
    zeros = {
        "median_price": 0.0,
        "lowest_price": 0.0,
        "release_lowest_price": 40.0,
        "num_for_sale": 7,
        "community_want": 10,
        "community_have": 100,
        "release_num_for_sale": 7,
        "blocked_from_sale": 0,
    }
    live = {
        "median_price": 99.0,
        "lowest_price": 40.0,
        "release_lowest_price": 40.0,
        "num_for_sale": 7,
        "community_want": 10,
        "community_have": 100,
        "release_num_for_sale": 7,
        "blocked_from_sale": 0,
    }
    row_z = row_dict_for_inference(
        "r1",
        "Near Mint (NM or M-)",
        "Near Mint (NM or M-)",
        zeros,
        cat,
        genre_index=3.0,
        country_index=2.0,
        primary_artist_index=5.0,
        primary_label_index=1.0,
        include_marketplace_scalars_in_features=False,
    )
    row_l = row_dict_for_inference(
        "r1",
        "Near Mint (NM or M-)",
        "Near Mint (NM or M-)",
        live,
        cat,
        genre_index=3.0,
        country_index=2.0,
        primary_artist_index=5.0,
        primary_label_index=1.0,
        include_marketplace_scalars_in_features=False,
    )
    cols = residual_training_feature_columns()
    for c in cols:
        assert row_z[c] == row_l[c], c


def test_residual_z_clip_abs_from_vinyliq() -> None:
    assert residual_z_clip_abs_from_vinyliq({}) is None
    assert (
        residual_z_clip_abs_from_vinyliq(
            {"training_target": {"residual_z_clip_abs": 1.25}}
        )
        == 1.25
    )
    assert (
        residual_z_clip_abs_from_vinyliq(
            {"training_target": {"residual_z_clip_abs": None}}
        )
        is None
    )
    assert (
        residual_z_clip_abs_from_vinyliq(
            {"training_target": {"residual_z_clip_abs": -1.0}}
        )
        is None
    )
