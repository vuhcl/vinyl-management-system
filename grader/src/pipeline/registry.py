"""MLflow model registry hook after full pipeline training."""

from __future__ import annotations

import logging
from typing import Any

import mlflow

from grader.src.mlflow_tracking import (
    configure_mlflow_from_config,
    mlflow_log_artifacts_enabled,
    vinyl_grader_pyfunc_has_python_model,
)

logger = logging.getLogger(__name__)


class PipelineRegistryMixin:
    """Register transformer pyfunc to MLflow model registry when configured."""

    config: dict

    def _register_transformer_to_registry(
        self,
        transformer_results: dict,
        *,
        want_register: bool,
        registry_model_name: str,
    ) -> None:
        if not want_register:
            return
        if not mlflow_log_artifacts_enabled(self.config):
            logger.info(
                "Skipping MLflow model registry (mlflow.log_artifacts: false)."
            )
            return
        run_id = (transformer_results or {}).get("mlflow_run_id") or ""
        if not run_id:
            logger.warning(
                "Skipping MLflow model registry: empty mlflow_run_id "
                "(enable MLflow for the transformer step to register)."
            )
            return
        configure_mlflow_from_config(self.config)
        if not vinyl_grader_pyfunc_has_python_model(run_id):
            logger.warning(
                "Skipping MLflow model registry: run %s has no usable "
                "vinyl_grader pyfunc (no python_model.pkl and no models-from-code "
                "entry in MLmodel). Remote pyfunc logging may have failed "
                "(see training logs for 'pyfunc logging failed'); increase "
                "MLFLOW_HTTP_REQUEST_TIMEOUT and related upload settings — "
                "grader/serving/README.md.",
                run_id,
            )
            return
        model_uri = f"runs:/{run_id}/vinyl_grader"
        try:
            ev = transformer_results.get("eval", {}).get("test", {})
            s_f1 = float(ev.get("sleeve", {}).get("macro_f1", 0.0))
            m_f1 = float(ev.get("media", {}).get("macro_f1", 0.0))
            mean_f1 = (s_f1 + m_f1) / 2.0
            mv = mlflow.register_model(
                model_uri=model_uri,
                name=registry_model_name,
                tags={
                    "source": "full_pipeline",
                    "test_mean_macro_f1": str(round(mean_f1, 4)),
                },
            )
            logger.info(
                "Registered model '%s' version %s from run %s (%s)",
                registry_model_name,
                mv.version,
                run_id,
                model_uri,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Model registration failed: %s", exc)
