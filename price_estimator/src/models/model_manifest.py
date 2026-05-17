"""Parsed ``model_manifest.json`` for MLflow bundles and ``load_fitted_regressor``."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .regressor_constants import TARGET_KIND_DOLLAR_LOG1P


@dataclass(frozen=True)
class ModelManifest:
    """VinylIQ on-disk / MLflow manifest (schema v2+ carries ``target_kind``)."""

    backend: str
    schema_version: int
    target_kind: str
    ensemble: dict[str, Any] | None

    @classmethod
    def from_dict(cls, manifest: dict[str, Any]) -> ModelManifest:
        raw = dict(manifest)
        backend = str(raw.get("backend", "")).strip()
        schema = int(raw.get("schema_version", 1))
        tk = str(raw.get("target_kind", "")).strip()
        if schema >= 2 and tk:
            target_kind = tk
        else:
            target_kind = TARGET_KIND_DOLLAR_LOG1P
        ens_raw = raw.get("ensemble")
        ensemble = (
            ens_raw if isinstance(ens_raw, dict) and ens_raw.get("enabled") else None
        )
        return cls(
            backend=backend,
            schema_version=schema,
            target_kind=target_kind,
            ensemble=ensemble,
        )
