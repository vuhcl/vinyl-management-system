"""
Resolve MLflow tracking URI: ``MLFLOW_TRACKING_URI`` overrides YAML fallback.

Order of precedence:
  1. Environment variable ``MLFLOW_TRACKING_URI`` (non-empty)
  2. ``mlflow.tracking_uri_fallback`` in config
  3. Legacy ``mlflow.tracking_uri`` if set and not a ``${...}`` placeholder
  4. Default local SQLite store under ``grader/experiments/``

``configure_mlflow_from_config`` loads repo-root ``.env`` first so tuning and
training pick up ``MLFLOW_TRACKING_URI`` without a manual ``export``.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Mapping, MutableMapping
from urllib.parse import urlparse

from grader.src.project_env import load_project_dotenv

logger = logging.getLogger(__name__)

_DEFAULT_LOCAL = "sqlite:///grader/experiments/mlflow.db"


def is_remote_mlflow_tracking_uri(uri: str | None) -> bool:
    """
    True when the client talks to a remote MLflow tracking server (HTTP(S) or
    Databricks), where large standalone artifact uploads are often redundant
    with the pyfunc model bundle or prone to timeouts.

    Local file/SQLite stores return False.
    """
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


def configure_mlflow_from_config(
    config: MutableMapping[str, Any],
) -> str:
    """
    Apply resolved tracking URI and experiment from *config* to the global
    MLflow client. Writes the resolved URI back to
    ``config["mlflow"]["tracking_uri"]`` for introspection.
    """
    import mlflow

    load_project_dotenv()

    ml = config.setdefault("mlflow", {})
    uri = resolve_mlflow_tracking_uri(ml)
    ml["tracking_uri"] = uri
    mlflow.set_tracking_uri(uri)
    exp = ml.get("experiment_name")
    if exp:
        mlflow.set_experiment(str(exp))
    logger.info("MLflow tracking URI (runs go here): %s", uri)
    return uri
