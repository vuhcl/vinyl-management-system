"""Ensemble model manifest + per-head estimator persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ...models.fitted_regressor import TARGET_KIND_DOLLAR_LOG1P


def _save_ensemble_manifest_and_estimators(
    model_dir: Path,
    *,
    backend: str,
    target_kind: str,
    target_was_log1p: bool,
    feature_columns: list[str],
    champ_nm: Any,
    champ_ord: Any,
    blend_t: float,
    blend_s: float,
) -> None:
    """Write schema_version 3 manifest + per-head estimator joblibs for pyfunc."""
    import joblib

    model_dir.mkdir(parents=True, exist_ok=True)
    nm_path = "regressor_ensemble_nm.joblib"
    ord_path = "regressor_ensemble_ord.joblib"
    joblib.dump(champ_nm.estimator, model_dir / nm_path)
    joblib.dump(champ_ord.estimator, model_dir / ord_path)
    # Legacy artifact name (ordinal head); unused by ensemble pyfunc but keeps layouts consistent.
    joblib.dump(champ_ord.estimator, model_dir / "regressor.joblib")
    joblib.dump(feature_columns, model_dir / "feature_columns.joblib")
    joblib.dump(
        target_kind == TARGET_KIND_DOLLAR_LOG1P and target_was_log1p,
        model_dir / "target_log1p.joblib",
    )
    manifest = {
        "schema_version": 3,
        "backend": backend,
        "target_kind": target_kind,
        "ensemble": {
            "enabled": True,
            "blend": {
                "kind": "log_anchor_sigmoid",
                "t": float(blend_t),
                "s": float(blend_s),
            },
            "regressor_nm": nm_path,
            "regressor_ord": ord_path,
        },
    }
    (model_dir / "model_manifest.json").write_text(json.dumps(manifest, indent=2))
