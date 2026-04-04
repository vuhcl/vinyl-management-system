"""Load MLflow pyfunc model from registry or run URI."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import mlflow
from google.auth.exceptions import InvalidOperation as GoogleAuthInvalidOperation

from grader.src.project_env import load_project_dotenv

logger = logging.getLogger(__name__)


def _ensure_google_cloud_project_from_sa_json() -> None:
    """Set GOOGLE_CLOUD_PROJECT from the key file when MLflow/GCS needs a project id."""
    if os.environ.get("GOOGLE_CLOUD_PROJECT", "").strip():
        return
    gac = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    if not gac:
        return
    path = Path(gac)
    if not path.is_file():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return
    pid = str(data.get("project_id") or "").strip()
    if pid:
        os.environ["GOOGLE_CLOUD_PROJECT"] = pid
        logger.info("Set GOOGLE_CLOUD_PROJECT from credentials file: %s", pid)


def load_grader_pyfunc():
    """
    Load the vinyl grader pyfunc model.

    Environment:
        MLFLOW_MODEL_URI — required, e.g. ``models:/VinylGrader/latest``
        MLFLOW_TRACKING_URI — optional; if unset, MLflow uses its default
        MLFLOW_REGISTRY_URI — optional separate registry server

    Large artifact downloads (e.g. ``python_model.pkl``) over the tracking
    server's HTTP artifact API can hit ``Connection reset by peer``; urllib3
    will retry automatically. If loads still fail, raise timeouts/retries (see
    MLflow's ``mlflow.environment_variables``): ``MLFLOW_HTTP_REQUEST_TIMEOUT``
    (default 120s), ``MLFLOW_HTTP_REQUEST_MAX_RETRIES``. Prefer storing run
    artifacts in ``gs://`` (or similar) so the client pulls from object
    storage instead of long single HTTP responses through the tracker.
    """
    load_project_dotenv()
    model_uri = os.environ.get("MLFLOW_MODEL_URI", "").strip()
    if not model_uri:
        raise RuntimeError(
            "MLFLOW_MODEL_URI is not set (e.g. models:/VinylGrader/latest)"
        )

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "").strip()
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        logger.info("MLflow tracking URI: %s", tracking_uri)

    registry_uri = os.environ.get("MLFLOW_REGISTRY_URI", "").strip()
    if registry_uri:
        mlflow.set_registry_uri(registry_uri)
        logger.info("MLflow registry URI: %s", registry_uri)

    gac = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    if gac:
        p = Path(gac)
        if p.is_dir():
            raise RuntimeError(
                f"GOOGLE_APPLICATION_CREDENTIALS={gac!r} is a directory, not a JSON key file. "
                "Docker often does this when -v mounts a host path that does not exist: Docker "
                "creates an empty directory on the host and mounts it. Use the full path to the "
                "service-account .json file on the host, e.g. "
                "-v /home/you/keys/sa.json:/secrets/sa.json:ro "
                "-e GOOGLE_APPLICATION_CREDENTIALS=/secrets/sa.json"
            )

    _ensure_google_cloud_project_from_sa_json()

    logger.info("Loading pyfunc model: %s", model_uri)
    try:
        return mlflow.pyfunc.load_model(model_uri)
    except FileNotFoundError as exc:
        if "python_model.pkl" in str(exc):
            raise RuntimeError(
                "Pyfunc bundle is incomplete: python_model.pkl missing after download. "
                "The Model Registry version may reference storage without the full model "
                "(e.g. registration fell back to an internal models:/ URI, or upload "
                "to the current artifact root failed). Fix: in the MLflow UI open the "
                "source run and confirm vinyl_grader/python_model.pkl exists; then load "
                "with MLFLOW_MODEL_URI=runs:/<run_id>/vinyl_grader or run "
                "mlflow.register_model('runs:/<run_id>/vinyl_grader', 'VinylGrader') "
                "after a successful train/log."
            ) from exc
        raise
    except OSError as exc:
        if "Project was not passed" in str(exc) or "could not be determined" in str(
            exc
        ):
            raise RuntimeError(
                "Google Cloud Storage needs a project id. Set environment variable "
                "GOOGLE_CLOUD_PROJECT to your GCP project (the same project that owns "
                "the MLflow artifact bucket), e.g. "
                "-e GOOGLE_CLOUD_PROJECT=my-gcp-project "
                "or ensure the service-account JSON includes a project_id field."
            ) from exc
        raise
    except GoogleAuthInvalidOperation as exc:
        raise RuntimeError(
            "MLflow is trying to download model artifacts from Google Cloud "
            "Storage, but no valid GCP credentials are available inside the "
            "container. Mount a service-account JSON with access to the "
            "artifact bucket and set GOOGLE_APPLICATION_CREDENTIALS to its "
            "path (e.g. docker run -v /path/to/key.json:/secrets/sa.json:ro "
            "-e GOOGLE_APPLICATION_CREDENTIALS=/secrets/sa.json ...). "
            "Do not point GOOGLE_APPLICATION_CREDENTIALS at an empty or "
            "/dev/null mount."
        ) from exc
