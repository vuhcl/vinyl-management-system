"""Sanity helpers for residual target (distribution + constant-z baselines)."""
from __future__ import annotations

import numpy as np

from price_estimator.src.models.fitted_regressor import (
    TARGET_KIND_DOLLAR_LOG1P,
    TARGET_KIND_RESIDUAL_LOG_MEDIAN,
    log1p_dollar_targets_for_metrics,
    median_ape_dollars,
    pred_log1p_dollar_for_metrics,
)
from price_estimator.src.training.train_vinyliq import report_residual_target_sanity


def test_report_residual_skips_non_residual(capsys) -> None:
    report_residual_target_sanity(
        np.array([0.1, 0.2]),
        np.array([10.0, 20.0]),
        TARGET_KIND_DOLLAR_LOG1P,
    )
    assert "Residual target sanity" not in capsys.readouterr().out


def test_constant_z0_matches_tiny_residual_mape() -> None:
    """If z≈0, pred z=0 + correct anchor ≈ truth → MdAPE ~0."""
    m = np.array([25.0, 30.0, 40.0])
    z = np.array([0.0, 1e-8, -1e-8])
    y_lp = log1p_dollar_targets_for_metrics(z, m, TARGET_KIND_RESIDUAL_LOG_MEDIAN)
    pred_lp = pred_log1p_dollar_for_metrics(np.zeros_like(z), m, TARGET_KIND_RESIDUAL_LOG_MEDIAN)
    assert median_ape_dollars(y_lp, pred_lp) < 1e-6


def test_shuffled_anchor_blows_up_mape_when_z_nonzero() -> None:
    """Wrong anchor + z=0 should not match true dollars when residual is non-trivial."""
    rng = np.random.default_rng(0)
    m = rng.uniform(10.0, 80.0, size=200)
    z = rng.normal(0, 0.15, size=200)
    y_lp = log1p_dollar_targets_for_metrics(z, m, TARGET_KIND_RESIDUAL_LOG_MEDIAN)
    pred_bad = pred_log1p_dollar_for_metrics(
        np.zeros_like(z), rng.permutation(m), TARGET_KIND_RESIDUAL_LOG_MEDIAN
    )
    assert median_ape_dollars(y_lp, pred_bad) > 0.05


def test_report_residual_prints_blocks(capsys) -> None:
    rng = np.random.default_rng(1)
    n = 500
    m = rng.uniform(15.0, 60.0, size=n)
    z = rng.normal(0, 0.05, size=n)
    report_residual_target_sanity(z, m, TARGET_KIND_RESIDUAL_LOG_MEDIAN, seed=0)
    out = capsys.readouterr().out
    assert "Residual target sanity" in out
    assert "Constant z=0 + correct" in out
    assert "shuffled anchor" in out
