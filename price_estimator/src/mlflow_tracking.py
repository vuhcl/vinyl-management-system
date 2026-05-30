"""
MLflow setup for VinylIQ training and inference.

URI resolution and client bootstrap live in ``shared.mlflow_tracking``; this
module adds GCS credential wiring for remote artifact stores.
"""

from __future__ import annotations

from typing import Any, MutableMapping

from core.config import get_project_root
from shared.mlflow_tracking import (
    configure_mlflow_client_from_config,
    is_remote_mlflow_tracking_uri,
    resolve_mlflow_tracking_uri,
)

__all__ = [
    "apply_google_credentials_from_mlflow_config",
    "configure_mlflow_from_config",
    "is_remote_mlflow_tracking_uri",
    "resolve_mlflow_tracking_uri",
]


def apply_google_credentials_from_mlflow_config(
    mlflow_cfg: dict[str, Any] | None,
    *,
    repo_root=None,
) -> None:
    from shared.mlflow_tracking import apply_google_credentials_from_mlflow_config as _apply

    root = repo_root or get_project_root()
    _apply(mlflow_cfg, repo_root=root)


def configure_mlflow_from_config(config: MutableMapping[str, Any]) -> str:
    return configure_mlflow_client_from_config(
        config,
        repo_root=get_project_root(),
        apply_gcs_credentials=True,
    )
