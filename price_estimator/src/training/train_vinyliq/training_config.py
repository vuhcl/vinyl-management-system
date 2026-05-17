"""YAML config, training target / ensemble settings, MLflow label params."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml

from ...models.fitted_regressor import (
    TARGET_KIND_DOLLAR_LOG1P,
    TARGET_KIND_RESIDUAL_LOG_MEDIAN,
)
from ..sale_floor_enums import TrainingLabelMode
from ..vinyliq_tuning_selection import _resolve_single_selection_metric


def confidence_interval_settings_from_vinyliq(v: dict | None) -> dict[str, Any]:
    """Defaults for ``vinyliq.confidence_intervals`` (YAML)."""
    vv = v if isinstance(v, dict) else {}
    raw = vv.get("confidence_intervals")
    if not isinstance(raw, dict):
        raw = {}
    xgb_ov = raw.get("xgboost")
    return {
        "enabled": bool(raw.get("enabled", False)),
        "lower_alpha": float(raw.get("lower_alpha", 0.1)),
        "upper_alpha": float(raw.get("upper_alpha", 0.9)),
        "residual_abs_error_quantile": float(raw.get("residual_abs_error_quantile", 0.8)),
        "min_half_width_usd": float(raw.get("min_half_width_usd", 1.0)),
        "min_holdout_n": int(raw.get("min_holdout_n", 50)),
        "xgboost_overrides": xgb_ov if isinstance(xgb_ov, dict) else {},
    }


def training_target_kind_from_vinyliq(v: dict | None) -> str:
    raw = (v or {}).get("training_target") or {}
    if not isinstance(raw, dict):
        raw = {}
    k = str(raw.get("kind", "residual_log_median")).strip().lower()
    if k in ("residual_log_median", "residual", "residual_log1p_median"):
        return TARGET_KIND_RESIDUAL_LOG_MEDIAN
    return TARGET_KIND_DOLLAR_LOG1P


def residual_z_clip_abs_from_vinyliq(v: dict | None) -> float | None:
    """Optional winsor on ``z = log1p(y_label) - log1p(m)`` to ``[-c, c]`` (null = off).

    ``c`` is a fixed half-width in **log1p-dollar residual space** (not tied to removed
    ``median_price`` columns).
    """
    raw = (v or {}).get("training_target") or {}
    if not isinstance(raw, dict):
        return None
    c = raw.get("residual_z_clip_abs")
    if c is None:
        return None
    try:
        x = float(c)
    except (TypeError, ValueError):
        return None
    return x if x > 0 else None


def _tuning_sample_weight_mode(v: dict[str, Any]) -> str | None:
    t = v.get("tuning") or {}
    sw = t.get("sample_weight")
    if sw is None:
        return None
    s = str(sw).strip()
    return s if s else None


def _format_sample_weight_multipliers(v: dict[str, Any]) -> dict[str, float] | None:
    t = v.get("tuning") or {}
    raw = t.get("format_sample_weight_multipliers")
    if not isinstance(raw, dict) or not raw:
        return None
    out: dict[str, float] = {}
    for k, val in raw.items():
        try:
            out[str(k)] = float(val)
        except (TypeError, ValueError):
            continue
    return out or None


def _training_label_console_summary(tl: dict[str, object]) -> str:
    """Human-readable label config: only keys that apply to ``mode``."""
    mode = str(tl.get("mode", TrainingLabelMode.SALE_FLOOR_BLEND)).strip().lower()
    parts: list[str] = [f"mode={mode}"]
    if mode in (TrainingLabelMode.SALE_FLOOR_BLEND, TrainingLabelMode.SALE_FLOOR):
        sfb = tl.get("sale_floor_blend")
        if isinstance(sfb, dict) and sfb:
            parts.append(
                "sale_floor_blend="
                + json.dumps(sfb, separators=(",", ":"), sort_keys=True)
            )
        parts.append(f"price_suggestion_grade(anchor)={tl.get('price_suggestion_grade')!s}")
    else:
        parts.append(
            "note=only sale_floor_blend / sale_floor are supported for VinylIQ training"
        )
    return ", ".join(parts)


def _training_label_mlflow_params(tl: dict[str, object]) -> dict[str, str]:
    """Flat params for MLflow (sale-floor training and optional nested knobs)."""
    mode = str(tl.get("mode", TrainingLabelMode.SALE_FLOOR_BLEND)).strip().lower()
    out: dict[str, str] = {"training_label_mode": mode}
    if mode in (TrainingLabelMode.SALE_FLOOR_BLEND, TrainingLabelMode.SALE_FLOOR):
        sfb = tl.get("sale_floor_blend")
        if isinstance(sfb, dict):
            for k, v in sorted(sfb.items()):
                out[f"training_label_sf_{k}"] = str(v)
        out["training_label_ps_grade_anchor"] = str(tl.get("price_suggestion_grade", ""))
    return out


def _mlflow_log_training_label_params(
    mlflow: Any,
    tl: dict[str, object],
) -> None:
    for k, v in _training_label_mlflow_params(tl).items():
        mlflow.log_param(k, v)


def _root() -> Path:
    # train_vinyliq/training_config.py -> parents[3] == price_estimator package root
    return Path(__file__).resolve().parents[3]


def load_config() -> dict:
    env = os.environ.get("VINYLIQ_CONFIG")
    p = Path(env) if env else (_root() / "configs" / "base.yaml")
    with open(p) as f:
        return yaml.safe_load(f) or {}


def _mlflow_flags(cfg: dict) -> tuple[bool, bool]:
    """
    Returns ``(tracking_enabled, upload_artifacts)``.

    ``upload_artifacts`` is False when tracking is off. When true, config dir + model dir
    and pyfunc are uploaded; registry requires this path.
    """
    ml = cfg.get("mlflow") or {}
    on = bool(ml.get("enabled", True))
    art = bool(ml.get("log_artifacts", True))
    return on, art and on


def _config_path_for_mlflow(root: Path) -> Path:
    env_cfg = os.environ.get("VINYLIQ_CONFIG")
    if env_cfg:
        cfg_path = Path(env_cfg)
        if not cfg_path.is_absolute():
            cfg_path = root / cfg_path
        return cfg_path
    return root / "configs" / "base.yaml"
def _blend_sweep_pairs_from_ensemble_dict(
    raw: dict[str, Any],
    *,
    default_t: float,
    default_s: float,
) -> list[tuple[float, float]] | None:
    """
    Optional Cartesian grid or explicit ``pairs`` for post-hoc val selection of ``(t, s)``.

    Returns ``None`` when sweep is disabled; otherwise a non-empty list of ``(t, s)`` pairs.
    """
    sw = raw.get("blend_sweep")
    if not isinstance(sw, dict) or not sw.get("enabled", False):
        return None
    if "pairs" in sw:
        expl = sw.get("pairs")
        if not isinstance(expl, list):
            raise ValueError("ensemble.blend_sweep.pairs must be a list")
        if not expl:
            raise ValueError("ensemble.blend_sweep.pairs is empty")
        out: list[tuple[float, float]] = []
        for row in expl:
            if isinstance(row, (list, tuple)) and len(row) >= 2:
                out.append((float(row[0]), float(row[1])))
        if not out:
            raise ValueError("ensemble.blend_sweep.pairs has no valid [t, s] rows")
        return out
    ts = sw.get("t")
    ss = sw.get("s")
    if isinstance(ts, (list, tuple)) and isinstance(ss, (list, tuple)):
        if not ts or not ss:
            return [(default_t, default_s)]
        return [(float(t), float(s)) for t in ts for s in ss]
    return [(default_t, default_s)]


def ensemble_blend_config_from_vinyliq(v: dict[str, Any] | None) -> dict[str, Any] | None:
    raw = (v or {}).get("ensemble")
    if not isinstance(raw, dict) or not raw.get("enabled", False):
        return None
    blend = raw.get("blend") or {}
    kind = str(blend.get("kind", "log_anchor_sigmoid")).strip().lower()
    if kind != "log_anchor_sigmoid":
        raise ValueError(
            f"Unsupported vinyliq.ensemble.blend.kind {kind!r} (only log_anchor_sigmoid)"
        )
    dt = float(blend.get("t", 4.0))
    ds = float(blend.get("s", 0.35))
    sweep_pairs = _blend_sweep_pairs_from_ensemble_dict(
        raw, default_t=dt, default_s=ds
    )
    return {
        "kind": kind,
        "t": dt,
        "s": ds,
        "share_hparams": bool(raw.get("share_hparams", True)),
        "blend_sweep_pairs": sweep_pairs,
    }

def _write_training_label_config(
    model_dir: Path,
    training_label: dict[str, object],
    *,
    training_target: dict[str, object] | None = None,
) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {"schema_version": 1, **training_label}
    if training_target:
        payload["training_target"] = dict(training_target)
    (model_dir / "training_label.json").write_text(json.dumps(payload, indent=2))

def _resolve_tuning_selection_metric(
    tuning: dict | None,
) -> tuple[str, str]:
    """
    Legacy single-metric resolution (``composite`` falls back to MdAPE here).

    Prefer ``parse_selection_objective`` for full tuning behavior.
    """
    raw = str((tuning or {}).get("selection_metric", "median_ape")).strip().lower()
    if raw == "composite":
        return ("mdape", "val_median_ape_dollars")
    return _resolve_single_selection_metric(tuning)


def _enabled_families(v: dict) -> list[str]:
    mf = v.get("model_families") or {}
    order = [
        "xgboost",
        "lightgbm",
        "catboost",
        "sklearn_hist_gbrt",
        "sklearn_rf",
        "sklearn_et",
    ]
    out: list[str] = []
    for name in order:
        if mf.get(name, False):
            out.append(name)
    return out
