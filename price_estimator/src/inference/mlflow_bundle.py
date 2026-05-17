"""Resolve VinylIQ champion artifacts from MLflow and download into a local cache."""
from __future__ import annotations

import hashlib
import logging
import os
import shutil
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

RUNS_PREFIX = "runs:/"


def resolve_downloaded_vinyliq_bundle_root(bundle: Path) -> Path:
    """
    Normalize ``bundle`` so ``load_fitted_regressor`` sees weights at directory root.

    ``mlflow.artifacts.download_artifacts`` varies by backing store/version: artifacts
    may land under ``bundle/`` or nested ``bundle/vinyliq_artifacts/``. Health checks and
    the joblib/XGB loaders expect ``model_manifest.json`` / regressors beside that root.
    """
    b = Path(bundle)

    def has_weight_files(p: Path) -> bool:
        return (
            (p / "model_manifest.json").is_file()
            or (p / "regressor.joblib").is_file()
            or (p / "xgb_model.joblib").is_file()
        )

    if has_weight_files(b):
        return b
    nested = b / "vinyliq_artifacts"
    if nested.is_dir() and has_weight_files(nested):
        logger.info(
            "MLflow bundle artifacts under subdirectory; using %s for inference weights",
            nested,
        )
        return nested
    if b.is_dir():
        for child in sorted(p for p in b.iterdir() if p.is_dir()):
            if has_weight_files(child):
                logger.warning(
                    "Unexpected MLflow artifact layout; using first directory with weights: %s",
                    child,
                )
                return child
    return b


def _latest_registry_model_version(client: Any, registered_name: str) -> Any:
    """Resolve ``models:/Name/latest`` to the highest numeric ``version_number``."""
    name = registered_name.strip()
    escaped = name.replace("'", "''")
    results = client.search_model_versions(
        filter_string=f"name='{escaped}'",
        order_by=["version_number DESC"],
        max_results=1,
    )
    if not results:
        raise ValueError(
            f"No registered versions found for model name {name!r} (URI uses /latest)",
        )
    return results[0]


def _normalize_runs_bundle_uri(uri: str) -> str:
    """Ensure URI points at the logged champion directory ``vinyliq_artifacts``."""
    u = uri.strip()
    if not u.startswith(RUNS_PREFIX):
        return u
    body = u[len(RUNS_PREFIX) :]
    if "/" not in body:
        return f"{RUNS_PREFIX}{body}/vinyliq_artifacts"
    if u.endswith("/vinyliq_model"):
        return u[: -len("/vinyliq_model")] + "/vinyliq_artifacts"
    return u


def _artifact_uri_from_registry_source(source: str) -> str:
    """Training registers pyfunc under ``…/vinyliq_model``; API loads flat bundle ``…/vinyliq_artifacts``."""
    if "/vinyliq_model" in source:
        return source.replace("/vinyliq_model", "/vinyliq_artifacts")
    return source


def resolve_vinyliq_bundle_artifact_uri(
    model_uri: str,
    *,
    tracking_uri: str | None,
) -> tuple[str, str]:
    """
    Map a user-facing VinylIQ MLflow URI to an ``artifact_uri`` for ``download_artifacts``.

    Supports:

    - ``runs:/<run_id>`` → ``runs:/<run_id>/vinyliq_artifacts``
    - ``runs:/<run_id>/vinyliq_model`` → ``runs:/<run_id>/vinyliq_artifacts``
    - ``runs:/<run_id>/vinyliq_artifacts`` → unchanged
    - ``models:/<name>/<version>`` → champion bundle under that version's run (integer version)
    - ``models:/<name>/latest`` → champion bundle under the highest registered version number
    - ``models:/<name>@<alias>`` → same (MLflow 3 alias syntax)

    Returns ``(artifact_uri, cache_stamp)`` where ``cache_stamp`` changes when registry
    versions advance or the URI changes.
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    mu = model_uri.strip()
    if mu.startswith(RUNS_PREFIX):
        au = _normalize_runs_bundle_uri(mu)
        return au, au

    if mu.startswith("models:/"):
        rest = mu[len("models:/") :]
        client = MlflowClient()
        if "@" in rest:
            name, alias = rest.split("@", 1)
            mv = client.get_model_version_by_alias(name.strip(), alias.strip())
        elif "/" in rest:
            name, ver = rest.split("/", 1)
            n, v = name.strip(), ver.strip()
            if v.lower() == "latest":
                mv = _latest_registry_model_version(client, n)
            else:
                mv = client.get_model_version(n, v)
        else:
            raise ValueError(
                "models: URI must be models:/RegisteredModelName/<version>|latest "
                "or models:/RegisteredModelName@alias "
                f"(got {model_uri!r})",
            )
        src = str(mv.source).strip()
        au = _artifact_uri_from_registry_source(src)
        au = _normalize_runs_bundle_uri(au)
        stamp = f"{mv.name}:{mv.version}:{au}"
        return au, stamp

    raise ValueError(
        "VinylIQ MLflow URI must start with runs:/ or models:/ "
        f"(got {model_uri!r})",
    )


def download_vinyliq_bundle_to_cache(
    *,
    model_uri: str,
    tracking_uri: str | None,
    cache_root: Path,
    force: bool = False,
) -> Path:
    """
    Download champion artifacts to ``cache_root`` and return the directory containing
    ``regressor.joblib`` / ``model_manifest.json`` / encoders.

    Uses a content-addressed subdirectory keyed by ``cache_stamp`` so promoting a new
    registry version invalidates the cache automatically.
    """
    import tempfile

    import mlflow

    artifact_uri, stamp = resolve_vinyliq_bundle_artifact_uri(
        model_uri,
        tracking_uri=tracking_uri,
    )
    cache_root.mkdir(parents=True, exist_ok=True)
    slug = hashlib.sha256(stamp.encode("utf-8")).hexdigest()[:24]
    dest = cache_root / f"vinyliq_bundle_{slug}"
    marker = dest.with_suffix(".stamp")

    if (
        dest.is_dir()
        and marker.is_file()
        and marker.read_text(encoding="utf-8").strip() == stamp.strip()
        and not force
    ):
        logger.info("Using cached VinylIQ MLflow bundle at %s", dest)
        return resolve_downloaded_vinyliq_bundle_root(dest)

    tmp = tempfile.mkdtemp(prefix="vinyliq_mlflow_dl_")
    try:
        local = mlflow.artifacts.download_artifacts(
            artifact_uri=artifact_uri,
            dst_path=tmp,
            tracking_uri=tracking_uri or None,
        )
        src_path = Path(local)
        if dest.exists():
            shutil.rmtree(dest, ignore_errors=True)
        shutil.move(str(src_path), str(dest))
        marker.write_text(stamp + "\n", encoding="utf-8")
        root = resolve_downloaded_vinyliq_bundle_root(dest)
        logger.info(
            "Downloaded VinylIQ MLflow artifact %s → %s (bundle root %s)",
            artifact_uri,
            dest,
            root,
        )
        return root
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def mlflow_pull_enabled(model_uri: str | None) -> bool:
    env_first = (os.environ.get("VINYLIQ_MLFLOW_MODEL_URI") or "").strip()
    return bool(env_first or (model_uri or "").strip())
