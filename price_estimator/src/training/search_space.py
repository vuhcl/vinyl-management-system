"""Sample hyperparameters from YAML search_space dicts."""
from __future__ import annotations

from typing import Any

import numpy as np


def _is_numeric_range_pair(v: list[Any]) -> bool:
    if len(v) != 2:
        return False
    return all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v)


def sample_from_space(space: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    """
    For each key, if value is a list of length 2 of numbers → uniform sample in [lo, hi]
    (inclusive for ints when both endpoints are int). Otherwise → random choice from list.
    """
    out: dict[str, Any] = {}
    for k, v in space.items():
        if not isinstance(v, list) or len(v) == 0:
            continue
        if _is_numeric_range_pair(v):
            lo, hi = v[0], v[1]
            if isinstance(lo, int) and isinstance(hi, int):
                out[k] = int(rng.integers(int(lo), int(hi) + 1))
            else:
                out[k] = float(rng.uniform(float(lo), float(hi)))
        else:
            out[k] = v[int(rng.integers(0, len(v)))]
    return out
