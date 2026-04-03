"""Load MLflow pyfunc model from registry or run URI."""

from __future__ import annotations

import logging
import os

import mlflow

from grader.src.project_env import load_project_dotenv

logger = logging.getLogger(__name__)


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

    logger.info("Loading pyfunc model: %s", model_uri)
    return mlflow.pyfunc.load_model(model_uri)
