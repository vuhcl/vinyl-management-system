"""Top-level keys in ``grade_delta_scale.json`` that merge into serving ``grade_delta_scale``."""
from __future__ import annotations

# Fit artifact / placeholder scalars (excluding optional ``alpha`` / ``beta``).
GRADE_DELTA_FIT_TOP_LEVEL_NUMERIC_KEYS: tuple[str, ...] = (
    "price_ref_usd",
    "price_gamma",
    "price_scale_min",
    "price_scale_max",
    "age_k",
    "age_center_year",
)
