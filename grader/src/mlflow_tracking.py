"""
Resolve MLflow tracking URI: ``MLFLOW_TRACKING_URI`` overrides YAML fallback.

Grader-specific helpers (pipeline step runs, pyfunc artifact checks, transformer
tuning URI) stay in this module; URI resolution and client bootstrap live in
``shared.mlflow_tracking``.
"""

from __future__ import annotations

import contextlib
import logging
import os
from pathlib import Path
from typing import Any, Iterator, Mapping, MutableMapping

from grader.src.config_io import load_yaml
from shared.mlflow_tracking import (
    configure_mlflow_client_from_config,
    is_remote_mlflow_tracking_uri,
    mlflow_enabled,
    mlflow_log_artifacts_enabled,
    mlflow_section,
    resolve_mlflow_tracking_uri,
)
from shared.project_env import load_project_dotenv

logger = logging.getLogger(__name__)

__all__ = [
    "configure_mlflow_for_transformer_init",
    "configure_mlflow_from_config",
    "is_remote_mlflow_tracking_uri",
    "mlflow_enabled",
    "mlflow_log_artifacts_enabled",
    "mlflow_pipeline_step_run_ctx",
    "mlflow_start_run_ctx",
    "resolve_mlflow_tracking_uri",
    "vinyl_grader_pyfunc_has_python_model",
]


@contextlib.contextmanager
def mlflow_start_run_ctx(
    config: MutableMapping[str, Any],
    run_name: str,
) -> Iterator[None]:
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
    if not mlflow_enabled(config):
        yield False
        return
    if not bool(mlflow_section(config).get("log_pipeline_step_runs", False)):
        yield False
        return
    import mlflow

    configure_mlflow_from_config(config)
    with mlflow.start_run(run_name=run_name):
        yield True


def _walk_run_artifact_paths(run_id: str, path: str | None) -> list[str]:
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
    if not mlflow_enabled(config):
        return ""
    return configure_mlflow_client_from_config(config)
