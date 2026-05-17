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

import contextlib
import logging
import os
from pathlib import Path
from typing import Any, Iterator, Mapping, MutableMapping
from urllib.parse import urlparse

from grader.src.config_io import load_yaml
from grader.src.project_env import load_project_dotenv

logger = logging.getLogger(__name__)

_DEFAULT_LOCAL = "sqlite:///grader/experiments/mlflow.db"


def _mlflow_section(config: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(config or {}).get("mlflow") or {}


def mlflow_enabled(config: Mapping[str, Any] | None) -> bool:
    """When false, training should not create runs or log to MLflow."""
    return bool(_mlflow_section(config).get("enabled", True))


def mlflow_log_artifacts_enabled(config: Mapping[str, Any] | None) -> bool:
    """
    When false (and MLflow is enabled), log params/metrics/tags only — no
    log_artifact / pyfunc bundle / registry uploads.
    """
    if not mlflow_enabled(config):
        return False
    return bool(_mlflow_section(config).get("log_artifacts", True))


@contextlib.contextmanager
def mlflow_start_run_ctx(
    config: MutableMapping[str, Any],
    run_name: str,
) -> Iterator[None]:
    """
    Active MLflow run when ``mlflow.enabled`` is true; otherwise no-op context.
    """
    if not mlflow_enabled(config):
        yield
        return
    import mlflow

    configure_mlflow_from_config(config)
    with mlflow.start_run(run_name=run_name):
        yield


@contextlib.contextmanager
def mlflow_pipeline_step_run_ctx(
    config: MutableMapping[str, Any],
    run_name: str,
) -> Iterator[bool]:
    """
    Optional nested run for lightweight ETL / feature steps (ingest,
    harmonize, preprocess, TF-IDF, …).

    Yields True when an ``mlflow.start_run`` is active and callers may call
    ``mlflow.log_*``. Yields False when MLflow is disabled **or** when
    ``mlflow.log_pipeline_step_runs`` is false (default) — the pipeline step
    still executes; only server noise is skipped. Model training
    (baseline, transformer, …) opens its own runs regardless of this flag.
    """
    if not mlflow_enabled(config):
        yield False
        return
    if not bool(_mlflow_section(config).get("log_pipeline_step_runs", False)):
        yield False
        return
    import mlflow

    configure_mlflow_from_config(config)
    with mlflow.start_run(run_name=run_name):
        yield True


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


def _walk_run_artifact_paths(run_id: str, path: str | None) -> list[str]:
    """
    List artifact file paths under *path* (recursive).

    Empty on error or no children.
    """
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    out: list[str] = []

    def walk(prefix: str | None) -> None:
        try:
            batch = client.list_artifacts(run_id, path=prefix)
        except Exception:
            return
        for a in batch:
            if a.is_dir:
                walk(a.path)
            else:
                out.append(a.path)

    walk(path)
    return out


def vinyl_grader_pyfunc_has_python_model(run_id: str) -> bool:
    """
    True if the run has a usable pyfunc bundle under ``vinyl_grader/``.

    Accepts either legacy ``python_model.pkl`` (CloudPickle) or MLflow 2.12+
    models-from-code (``python_function.model_code_path`` in ``MLmodel``).

    Call before ``register_model(..., "runs:/<id>/vinyl_grader")`` so a failed
    remote ``log_pyfunc_model`` does not register a broken version.
    """
    from mlflow.tracking import MlflowClient

    paths = _walk_run_artifact_paths(run_id, "vinyl_grader")
    if any(
        p.endswith("python_model.pkl") or p.endswith("python_model.pkl.gz")
        for p in paths
    ):
        return True
    client = MlflowClient()
    for p in paths:
        if not p.endswith("MLmodel"):
            continue
        try:
            local = client.download_artifacts(run_id, p)
            doc = load_yaml(Path(local))
        except Exception:
            continue
        pyfunc = (doc or {}).get("flavors", {}).get("python_function", {})
        if pyfunc.get("model_code_path"):
            return True
    return False


def configure_mlflow_for_transformer_init(
    config: MutableMapping[str, Any],
    *,
    tuning: bool,
) -> None:
    """
    Point the MLflow client at the right store before ``TransformerTrainer``
    calls ``set_tracking_uri`` / ``set_experiment``.

    When ``tuning`` is true (``transformer_tune``), use the tuning SQLite URI
    and experiment so sweep runs do not flood the remote tracking server
    (Option A). Override with ``MLFLOW_TUNING_TRACKING_URI`` for a dedicated
    remote tuning experiment if desired.
    """
    if not mlflow_enabled(config):
        return

    import mlflow

    ml = config.setdefault("mlflow", {})
    if tuning:
        load_project_dotenv()
        uri = os.environ.get("MLFLOW_TUNING_TRACKING_URI", "").strip()
        if not uri:
            uri = str(ml.get("tuning_tracking_uri_fallback") or "").strip()
        if not uri:
            uri = "sqlite:///grader/experiments/mlflow_tuning.db"
        exp = str(ml.get("tuning_experiment_name") or "vinyl_grader_tune")
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(exp)
        ml["tracking_uri"] = uri
        ml["experiment_name"] = exp
        logger.info("MLflow tuning session — uri=%s experiment=%s", uri, exp)
        return

    configure_mlflow_from_config(config)


def configure_mlflow_from_config(
    config: MutableMapping[str, Any],
) -> str:
    """
    Apply resolved tracking URI and experiment from *config* to the global
    MLflow client. Writes the resolved URI back to
    ``config["mlflow"]["tracking_uri"]`` for introspection.
    """
    if not mlflow_enabled(config):
        return ""

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
