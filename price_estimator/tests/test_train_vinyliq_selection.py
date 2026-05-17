"""Tuning champion selection metric resolution."""
from __future__ import annotations

from price_estimator.src.training.train_vinyliq.training_config import (
    _mlflow_flags,
    _resolve_tuning_selection_metric,
)


def test_resolve_default_is_median_ape() -> None:
    assert _resolve_tuning_selection_metric({}) == ("mdape", "val_median_ape_dollars")
    assert _resolve_tuning_selection_metric({"selection_metric": "median_ape"}) == (
        "mdape",
        "val_median_ape_dollars",
    )


def test_resolve_wape_and_mae() -> None:
    assert _resolve_tuning_selection_metric({"selection_metric": "wape"}) == (
        "wape",
        "val_wape_dollars",
    )
    assert _resolve_tuning_selection_metric({"selection_metric": "mae_dollars"}) == (
        "mae",
        "val_mae_dollars_approx",
    )


def test_resolve_unknown_falls_back_to_median_ape() -> None:
    assert _resolve_tuning_selection_metric({"selection_metric": "unknown"}) == (
        "mdape",
        "val_median_ape_dollars",
    )


def test_resolve_composite_maps_to_mdape_placeholder() -> None:
    """``_resolve_tuning_selection_metric`` is legacy; composite uses ``parse_selection_objective``."""
    assert _resolve_tuning_selection_metric({"selection_metric": "composite"}) == (
        "mdape",
        "val_median_ape_dollars",
    )


def test_mlflow_flags_defaults() -> None:
    on, art = _mlflow_flags({})
    assert on is True and art is True


def test_mlflow_flags_disabled_or_metrics_only() -> None:
    assert _mlflow_flags({"mlflow": {"enabled": False}}) == (False, False)
    assert _mlflow_flags({"mlflow": {"enabled": True, "log_artifacts": False}}) == (
        True,
        False,
    )
