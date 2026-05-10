"""Monitoring smoke tests for recommender summaries (schema + interaction drift)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

pytestmark = pytest.mark.monitoring


def test_monitoring_interaction_summary_schema() -> None:
    df = pd.DataFrame(
        {
            "window_id": ["2025-W01"],
            "n_users": [1200],
            "n_items": [45000],
            "nnz": [890000],
            "item_degree_p90": [14.0],
            "sparsity": [0.016],
        }
    )
    assert list(df.columns) == [
        "window_id",
        "n_users",
        "n_items",
        "nnz",
        "item_degree_p90",
        "sparsity",
    ]


def test_monitoring_degree_distribution_shift() -> None:
    rng = np.random.default_rng(99)
    ref = rng.poisson(12, size=500).astype(float)
    cur = rng.poisson(45, size=500).astype(float)
    res = stats.ks_2samp(ref, cur, method="auto")
    assert res.pvalue < 0.01
    assert float(res.statistic) > 0.2
