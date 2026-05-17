"""String enums for sale-floor training labels and eligibility policies."""
from __future__ import annotations

from enum import StrEnum


class TrainingLabelMode(StrEnum):
    """``vinyliq.training_label.mode`` values supported by VinylIQ training."""

    SALE_FLOOR = "sale_floor"
    SALE_FLOOR_BLEND = "sale_floor_blend"


class SaleConditionPolicy(StrEnum):
    """``sale_condition_policy`` under ``sale_floor_blend`` (label construction path)."""

    NM_SUBSTRINGS_ONLY = "nm_substrings_only"
    ORDINAL_CASCADE = "ordinal_cascade"
