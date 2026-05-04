"""Train/val split by release_id (no row leakage)."""

from __future__ import annotations

import numpy as np


def train_test_split_by_release(
    rids: list[str],
    test_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[set[str], set[str]]:
    rng = np.random.default_rng(seed)
    uniq = sorted(set(rids))
    rng.shuffle(uniq)
    n_test = max(1, int(len(uniq) * test_fraction))
    test_set = set(uniq[:n_test])
    train_set = set(uniq[n_test:])
    return train_set, test_set

