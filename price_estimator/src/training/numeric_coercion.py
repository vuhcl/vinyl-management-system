"""Shared numeric coercion helpers for training / sale-floor modules."""
from __future__ import annotations

from typing import Any


def strictly_positive_float(v: Any) -> float | None:
    """Parse ``v`` as float; return only if finite and > 0."""
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if x > 0 else None
