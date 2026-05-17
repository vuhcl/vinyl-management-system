"""
Train VinylIQ regressor on marketplace labels + feature store.

Usage (from repo root):
  PYTHONPATH=. python -m price_estimator.src.training.train_vinyliq
  PYTHONPATH=. python -m price_estimator.src.training.train_vinyliq \\
    --google-application-credentials /path/to/service-account.json

For remote MLflow with GCS artifacts, set ``GOOGLE_APPLICATION_CREDENTIALS`` in ``.env``,
``mlflow.google_application_credentials`` in YAML (repo-relative path), or the CLI flag above.

Set ``vinyliq.tuning.enabled: true`` in config for multi-model search + MLflow registry.

MLflow cost control: ``mlflow.enabled: false`` or ``--no-mlflow`` skips tracking entirely.
``mlflow.log_artifacts: false`` or ``--mlflow-no-artifacts`` logs params/metrics only (no model
bundle / pyfunc / registry uploads to GCS).
"""

from __future__ import annotations

from .cli import main
from .release_train_split import train_test_split_by_release
from .training_config import (
    ensemble_blend_config_from_vinyliq,
    load_config,
    residual_z_clip_abs_from_vinyliq,
    training_target_kind_from_vinyliq,
)
from .training_frame import load_training_frame, report_residual_target_sanity

__all__ = [
    "ensemble_blend_config_from_vinyliq",
    "load_config",
    "load_training_frame",
    "main",
    "report_residual_target_sanity",
    "residual_z_clip_abs_from_vinyliq",
    "train_test_split_by_release",
    "training_target_kind_from_vinyliq",
]
