"""Tuning champion selection metric resolution."""
from __future__ import annotations

from price_estimator.src.training.train_vinyliq.training_config import _mlflow_flags


def test_mlflow_flags_defaults() -> None:
    on, art = _mlflow_flags({})
    assert on is True and art is True


def test_mlflow_flags_disabled_or_metrics_only() -> None:
    assert _mlflow_flags({"mlflow": {"enabled": False}}) == (False, False)
    assert _mlflow_flags({"mlflow": {"enabled": True, "log_artifacts": False}}) == (
        True,
        False,
    )
