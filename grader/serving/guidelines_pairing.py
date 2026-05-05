"""
Runtime pairing check: MLflow training tags vs ``grading_guidelines.yaml``.

Environment:
    MLFLOW_MODEL_URI — comparison uses the corresponding MLflow run tags.
    GRADER_STRICT_GUIDELINES_PAIRING — ``1`` / ``true`` raises on mismatch.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_last_model_guidelines_version: str | None = None


def get_model_guidelines_version_tag() -> str | None:
    """Version tag read from the MLflow run backing the loaded pyfunc model."""
    return _last_model_guidelines_version


def fetch_mlflow_run_guidelines_version(model_uri: str) -> str | None:
    """
    Resolve ``guidelines_version`` from the MLflow run associated with
    ``model_uri`` (``runs:/`` or ``models:/``).
    """
    uri = (model_uri or "").strip()
    if not uri:
        return None
    try:
        from mlflow.tracking import MlflowClient

        tu = os.environ.get("MLFLOW_TRACKING_URI", "").strip()
        client = MlflowClient(tracking_uri=tu if tu else None)
        run_id: str | None = None
        if uri.startswith("runs:/"):
            rest = uri[len("runs:/"):]
            run_id = rest.split("/")[0].strip() or None
        elif uri.startswith("models:/"):
            rest = uri[len("models:/"):]
            parts = rest.split("/")
            name = parts[0].strip()
            if not name:
                return None
            ver_spec = parts[1].strip() if len(parts) > 1 else "latest"
            source: str | None = None
            if ver_spec == "latest":
                mvs = client.get_latest_versions(name, stages=["None"])
                if not mvs:
                    mvs = client.search_model_versions(f"name='{name}'")
                if not mvs:
                    return None
                mv = max(mvs, key=lambda x: int(x.version))
                source = mv.source
            else:
                mv = client.get_model_version(name, ver_spec)
                source = mv.source
            if source and source.startswith("runs:/"):
                run_id = source[len("runs:/"):].split("/")[0].strip() or None
        if not run_id:
            return None
        run = client.get_run(run_id)
        tag = run.data.tags.get("guidelines_version")
        return tag.strip() if isinstance(tag, str) and tag.strip() else None
    except Exception as exc:
        logger.debug("Could not fetch MLflow guidelines_version tag: %s", exc)
        return None


def verify_serving_guidelines_pairing(model_uri: str | None = None) -> None:
    """
    Compare MLflow ``guidelines_version`` tag to runtime RuleEngine YAML.

    Sets :func:`get_model_guidelines_version_tag` when a tag is found.
    """
    global _last_model_guidelines_version
    _last_model_guidelines_version = None
    uri = (model_uri or os.environ.get("MLFLOW_MODEL_URI", "") or "").strip()
    if not uri:
        return
    model_ver = fetch_mlflow_run_guidelines_version(uri)
    _last_model_guidelines_version = model_ver
    try:
        from grader.serving.rule_postprocess import get_rule_engine
    except Exception:
        return
    try:
        rule_engine = get_rule_engine()
    except RuntimeError:
        return
    from grader.src.guidelines_identity import guidelines_version_from_mapping

    runtime_ver = guidelines_version_from_mapping(rule_engine.guidelines)
    strict_raw = os.environ.get("GRADER_STRICT_GUIDELINES_PAIRING", "")
    strict = strict_raw.strip().lower() in ("1", "true", "yes")
    if model_ver is None:
        logger.warning(
            "MLflow model has no guidelines_version tag on source run — "
            "cannot verify rubric pairing with runtime YAML (%s). "
            "Retrain/log model with an MLflow client that sets tags.",
            runtime_ver,
        )
        return
    if model_ver != runtime_ver:
        msg = (
            "guidelines_version mismatch: MLflow model tag="
            f"{model_ver!r} runtime YAML={runtime_ver!r}"
        )
        if strict:
            raise RuntimeError(msg)
        logger.warning(msg)
    else:
        logger.info(
            "Guidelines pairing OK — MLflow and runtime both "
            "guidelines_version=%s",
            runtime_ver,
        )
