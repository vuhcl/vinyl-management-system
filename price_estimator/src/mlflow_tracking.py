"""
MLflow setup aligned with the grader (``grader/src/mlflow_tracking.py``).

Tracking URI precedence:
  1. ``MLFLOW_TRACKING_URI`` (non-empty)
  2. ``mlflow.tracking_uri_fallback`` in config
  3. Legacy ``mlflow.tracking_uri`` if set and not a placeholder
  4. Default local DB path same as grader (shared metadata when local)

Remote server: artifacts use the server's ``--default-artifact-root`` (e.g.
``gs://…``). Set ``GOOGLE_APPLICATION_CREDENTIALS`` in ``.env``, or
``mlflow.google_application_credentials`` in YAML (path to service account JSON),
or pass ``--google-application-credentials`` when running training.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Mapping, MutableMapping
from urllib.parse import urlparse

from shared.project_env import load_project_dotenv

logger = logging.getLogger(__name__)


def _repo_root() -> Path:
    """``vinyl_management_system/`` (parent of ``price_estimator/``)."""
    return Path(__file__).resolve().parents[2]


def apply_google_credentials_from_mlflow_config(
    mlflow_cfg: Mapping[str, Any] | None,
    *,
    repo_root: Path | None = None,
) -> None:
    """
    Ensure Application Default Credentials work for MLflow → GCS artifact uploads.

    Precedence:
      1. ``GOOGLE_APPLICATION_CREDENTIALS`` if it points to an existing file.
      2. ``mlflow.google_application_credentials`` in YAML (path relative to repo
         root unless absolute).

    Call after ``load_project_dotenv()`` so ``.env`` can set the env var.
    """
    root = repo_root or _repo_root()
    ml = dict(mlflow_cfg or {})
    configured = str(ml.get("google_application_credentials") or "").strip()

    env_raw = str(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")).strip()
    if env_raw:
        env_path = Path(env_raw).expanduser()
        if env_path.is_file():
            logger.debug("Using GOOGLE_APPLICATION_CREDENTIALS from environment")
            return
        logger.warning(
            "GOOGLE_APPLICATION_CREDENTIALS=%r is not a file; GCS may fail.",
            env_raw,
        )

    if not configured:
        return

    path = Path(configured).expanduser()
    if not path.is_absolute():
        path = (root / path).resolve()
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

# Match grader: one local SQLite store when MLFLOW_TRACKING_URI unset.
_DEFAULT_LOCAL = "sqlite:///grader/experiments/mlflow.db"


def is_remote_mlflow_tracking_uri(uri: str | None) -> bool:
    s = (uri or "").strip()
    if not s:
        return False
    parsed = urlparse(s)
    scheme = (parsed.scheme or "").lower()
    if scheme in ("http", "https"):
        return True
    if scheme.startswith("databricks"):
        return True
    return False


def resolve_mlflow_tracking_uri(mlflow_cfg: Mapping[str, Any] | None) -> str:
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

    return _DEFAULT_LOCAL


def configure_mlflow_from_config(config: MutableMapping[str, Any]) -> str:
    """
    Set global MLflow tracking URI and experiment from *config*.

    Writes resolved URI to ``config["mlflow"]["tracking_uri"]``.
    """
    import mlflow

    load_project_dotenv()

    ml_pre = config.get("mlflow") or {}
    apply_google_credentials_from_mlflow_config(ml_pre)

    ml = config.setdefault("mlflow", {})
    uri = resolve_mlflow_tracking_uri(ml)
    ml["tracking_uri"] = uri
    mlflow.set_tracking_uri(uri)
    exp = ml.get("experiment_name")
    if exp:
        mlflow.set_experiment(str(exp))
    logger.info("MLflow tracking URI (runs go here): %s", uri)
    return uri
