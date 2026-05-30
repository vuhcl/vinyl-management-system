"""Shared MLflow tracking URI resolution and client bootstrap."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Mapping, MutableMapping
from urllib.parse import urlparse

from shared.project_env import load_project_dotenv

logger = logging.getLogger(__name__)

DEFAULT_LOCAL_TRACKING_URI = "sqlite:///grader/experiments/mlflow.db"


def mlflow_section(config: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(config or {}).get("mlflow") or {}


def mlflow_enabled(config: Mapping[str, Any] | None) -> bool:
    return bool(mlflow_section(config).get("enabled", True))


def mlflow_log_artifacts_enabled(config: Mapping[str, Any] | None) -> bool:
    if not mlflow_enabled(config):
        return False
    return bool(mlflow_section(config).get("log_artifacts", True))


def is_remote_mlflow_tracking_uri(uri: str | None) -> bool:
    s = (uri or "").strip()
    if not s:
        return False
    parsed = urlparse(s)
    scheme = (parsed.scheme or "").lower()
    return scheme in ("http", "https") or scheme.startswith("databricks")


def resolve_mlflow_tracking_uri(
    mlflow_cfg: Mapping[str, Any] | None,
    *,
    default_local: str = DEFAULT_LOCAL_TRACKING_URI,
) -> str:
    mlflow_cfg = dict(mlflow_cfg or {})
    env_uri = os.environ.get("MLFLOW_TRACKING_URI", "").strip()
    if env_uri:
        return env_uri
    fb = (mlflow_cfg.get("tracking_uri_fallback") or "").strip()
    if fb:
        return fb
    legacy = mlflow_cfg.get("tracking_uri")
    if legacy is not None:
        s = str(legacy).strip()
        if s and not s.startswith("${") and s.lower() != "null":
            return s
    return default_local


def apply_google_credentials_from_mlflow_config(
    mlflow_cfg: Mapping[str, Any] | None,
    *,
    repo_root: Path,
) -> None:
    ml = dict(mlflow_cfg or {})
    configured = str(ml.get("google_application_credentials") or "").strip()
    env_raw = str(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")).strip()
    if env_raw:
        env_path = Path(env_raw).expanduser()
        if env_path.is_file():
            return
        logger.warning(
            "GOOGLE_APPLICATION_CREDENTIALS=%r is not a file; GCS may fail.",
            env_raw,
        )
    if not configured:
        return
    path = Path(configured).expanduser()
    if not path.is_absolute():
        path = (repo_root / path).resolve()
    else:
        path = path.resolve()
    if not path.is_file():
        logger.warning(
            "mlflow.google_application_credentials=%r not found at %s",
            configured,
            path,
        )
        return
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(path)
    logger.info("Set GOOGLE_APPLICATION_CREDENTIALS for MLflow/GCS: %s", path)


def configure_mlflow_client_from_config(
    config: MutableMapping[str, Any],
    *,
    repo_root: Path | None = None,
    apply_gcs_credentials: bool = False,
    default_local: str = DEFAULT_LOCAL_TRACKING_URI,
) -> str:
    import mlflow

    load_project_dotenv()
    ml_pre = config.get("mlflow") or {}
    if apply_gcs_credentials:
        root = repo_root or Path(__file__).resolve().parents[1]
        apply_google_credentials_from_mlflow_config(ml_pre, repo_root=root)
    if not mlflow_enabled(config):
        return ""
    ml = config.setdefault("mlflow", {})
    uri = resolve_mlflow_tracking_uri(ml, default_local=default_local)
    ml["tracking_uri"] = uri
    mlflow.set_tracking_uri(uri)
    exp = ml.get("experiment_name")
    if exp:
        mlflow.set_experiment(str(exp))
    logger.info("MLflow tracking URI (runs go here): %s", uri)
    return uri
