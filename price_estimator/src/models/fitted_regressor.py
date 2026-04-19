"""Unified fitted regressor for VinylIQ (log1p target) across boosting backends."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

MANIFEST_FILE = "model_manifest.json"
REGRESSOR_FILE = "regressor.joblib"
FEATURE_COLUMNS_FILE = "feature_columns.joblib"
TARGET_LOG1P_FILE = "target_log1p.joblib"
LEGACY_XGB_FILE = "xgb_model.joblib"

TARGET_KIND_DOLLAR_LOG1P = "dollar_log1p"
TARGET_KIND_RESIDUAL_LOG_MEDIAN = "residual_log_median"


def log1p_dollar_from_residual(
    pred_z: np.ndarray,
    median_price_dollar: np.ndarray,
) -> np.ndarray:
    """``log1p(y_dollar) ≈ pred_z + log1p(median_anchor)``."""
    z = np.asarray(pred_z, dtype=np.float64)
    m = np.maximum(np.asarray(median_price_dollar, dtype=np.float64), 0.0)
    return z + np.log1p(m)


def log1p_dollar_targets_for_metrics(
    y_stored: np.ndarray,
    median_anchors: np.ndarray,
    target_kind: str,
) -> np.ndarray:
    """Convert training targets to log1p(dollar) for MAE/MdAPE when using residual target."""
    y = np.asarray(y_stored, dtype=np.float64)
    if target_kind == TARGET_KIND_RESIDUAL_LOG_MEDIAN:
        return y + np.log1p(np.maximum(np.asarray(median_anchors, dtype=np.float64), 0.0))
    return y


def pred_log1p_dollar_for_metrics(
    pred_stored: np.ndarray,
    median_anchors: np.ndarray,
    target_kind: str,
) -> np.ndarray:
    """Convert model predictions to log1p(dollar) for metrics and tuning (then ``expm1`` in MAE/MdAPE/WAPE)."""
    p = np.asarray(pred_stored, dtype=np.float64)
    if target_kind == TARGET_KIND_RESIDUAL_LOG_MEDIAN:
        return p + np.log1p(np.maximum(np.asarray(median_anchors, dtype=np.float64), 0.0))
    return p


def ensemble_blend_weight_log_anchor(
    median_anchor_usd: np.ndarray,
    *,
    center_log1p: float,
    scale: float,
) -> np.ndarray:
    """
    Sigmoid weight for the **NM-substrings** head vs **ordinal-cascade** head.

    ``w -> 1`` as ``log1p(anchor)`` increases past ``center_log1p``; ordinal weight is ``1 - w``.
    """
    m = np.maximum(np.asarray(median_anchor_usd, dtype=np.float64), 0.0)
    lx = np.log1p(m)
    c = float(center_log1p)
    s = max(float(scale), 1e-9)
    z = (lx - c) / s
    return 1.0 / (1.0 + np.exp(-z))


def metrics_dollar_from_log1p_masked(
    y_true_log1p: np.ndarray,
    pred_log1p: np.ndarray,
    mask: np.ndarray,
    *,
    min_count: int = 15,
) -> tuple[float, float, float]:
    """MAE / WAPE / MdAPE on a boolean row mask; NaNs if ``sum(mask) < min_count``."""
    y = np.asarray(y_true_log1p, dtype=np.float64)
    p = np.asarray(pred_log1p, dtype=np.float64)
    m = (
        np.asarray(mask, dtype=bool)
        & np.isfinite(y)
        & np.isfinite(p)
    )
    if int(np.sum(m)) < int(min_count):
        return (float("nan"), float("nan"), float("nan"))
    return (
        mae_dollars(y[m], p[m]),
        wape_dollars(y[m], p[m]),
        median_ape_dollars(y[m], p[m]),
    )


@dataclass
class FittedVinylIQRegressor:
    """Thin wrapper so inference and pyfunc share one predict path."""

    backend: str
    estimator: Any
    feature_columns: list[str]
    target_was_log1p: bool = True
    target_kind: str = TARGET_KIND_DOLLAR_LOG1P

    def predict_log1p(self, X: np.ndarray) -> np.ndarray:
        if self.estimator is None:
            raise RuntimeError("Model not fitted")
        out = self.estimator.predict(X)
        return np.asarray(out, dtype=np.float64).ravel()

    def predict_dollars(self, X: np.ndarray) -> np.ndarray:
        logp = self.predict_log1p(X)
        if self.target_kind == TARGET_KIND_RESIDUAL_LOG_MEDIAN:
            raise TypeError(
                "predict_dollars requires median anchor; use inference service or "
                "log1p_dollar_from_residual(predict_log1p(X), median) then expm1"
            )
        if self.target_was_log1p:
            return np.expm1(np.clip(logp, 0, 20))
        return np.clip(logp, 0, None)

    def save(self, directory: Path | str) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema_version": 2,
            "backend": self.backend,
            "target_kind": self.target_kind,
        }
        (d / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2))
        joblib.dump(self.estimator, d / REGRESSOR_FILE)
        joblib.dump(self.feature_columns, d / FEATURE_COLUMNS_FILE)
        joblib.dump(
            self.target_kind == TARGET_KIND_DOLLAR_LOG1P and self.target_was_log1p,
            d / TARGET_LOG1P_FILE,
        )
        if self.backend == "xgboost":
            joblib.dump(self.estimator, d / LEGACY_XGB_FILE)


def load_fitted_regressor(directory: Path | str) -> FittedVinylIQRegressor | None:
    """Load manifest bundle, or legacy XGB-only artifact layout."""
    d = Path(directory)
    mf = d / MANIFEST_FILE
    if mf.is_file():
        try:
            manifest = json.loads(mf.read_text())
        except (json.JSONDecodeError, OSError):
            return None
        backend = str(manifest.get("backend", "")).strip()
        if not backend or not (d / REGRESSOR_FILE).is_file():
            return None
        schema = int(manifest.get("schema_version", 1))
        tk = str(manifest.get("target_kind", "")).strip()
        if schema >= 2 and tk:
            target_kind = tk
        else:
            target_kind = TARGET_KIND_DOLLAR_LOG1P
        tw = bool(joblib.load(d / TARGET_LOG1P_FILE))
        if target_kind == TARGET_KIND_RESIDUAL_LOG_MEDIAN:
            tw = False
        return FittedVinylIQRegressor(
            backend=backend,
            estimator=joblib.load(d / REGRESSOR_FILE),
            feature_columns=list(joblib.load(d / FEATURE_COLUMNS_FILE)),
            target_was_log1p=tw,
            target_kind=target_kind,
        )
    if (d / LEGACY_XGB_FILE).is_file() and (d / FEATURE_COLUMNS_FILE).is_file():
        from .xgb_vinyliq import XGBVinylIQModel

        legacy = XGBVinylIQModel.load(d)
        return FittedVinylIQRegressor(
            backend="xgboost",
            estimator=legacy.model_,
            feature_columns=list(legacy.feature_columns_),
            target_was_log1p=bool(legacy.target_was_log1p_),
            target_kind=TARGET_KIND_DOLLAR_LOG1P,
        )
    return None


def _strip_unknown_xgb_params(params: dict[str, Any]) -> dict[str, Any]:
    import xgboost as xgb

    valid = set(xgb.XGBRegressor().get_params(deep=False).keys())
    return {k: v for k, v in params.items() if k in valid}


def _strip_unknown_lgb_params(params: dict[str, Any]) -> dict[str, Any]:
    import lightgbm as lgb

    valid = set(lgb.LGBMRegressor().get_params(deep=False).keys())
    return {k: v for k, v in params.items() if k in valid}


def _strip_unknown_cat_params(params: dict[str, Any]) -> dict[str, Any]:
    from catboost import CatBoostRegressor

    valid = set(CatBoostRegressor().get_params(deep=False).keys())
    return {k: v for k, v in params.items() if k in valid}


def _strip_unknown_sklearn_hgb(params: dict[str, Any]) -> dict[str, Any]:
    from sklearn.ensemble import HistGradientBoostingRegressor

    valid = set(HistGradientBoostingRegressor().get_params(deep=False).keys())
    return {k: v for k, v in params.items() if k in valid}


def _strip_unknown_rf(params: dict[str, Any]) -> dict[str, Any]:
    from sklearn.ensemble import RandomForestRegressor

    valid = set(RandomForestRegressor().get_params(deep=False).keys())
    return {k: v for k, v in params.items() if k in valid}


def _strip_unknown_et(params: dict[str, Any]) -> dict[str, Any]:
    from sklearn.ensemble import ExtraTreesRegressor

    valid = set(ExtraTreesRegressor().get_params(deep=False).keys())
    return {k: v for k, v in params.items() if k in valid}


def fit_regressor(
    family: str,
    params: dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_columns: list[str],
    *,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    early_stopping_rounds: int | None = None,
    random_state: int = 42,
    target_kind: str = TARGET_KIND_DOLLAR_LOG1P,
    sample_weight: np.ndarray | None = None,
) -> tuple[FittedVinylIQRegressor, dict[str, Any]]:
    """
    Fit one regressor. If early_stopping_rounds and val set are provided, uses them
    for boosting families; returns meta with best_iteration when available.
    """
    meta: dict[str, Any] = {"best_iteration": None}
    family = family.strip().lower()

    if family == "xgboost":
        import xgboost as xgb

        p = {"random_state": random_state, "verbosity": 0, **params}
        p = _strip_unknown_xgb_params(p)
        obj = str(p.get("objective", "reg:squarederror"))
        if "quantile" in obj.lower() and "quantile_alpha" not in p:
            p["quantile_alpha"] = 0.5
        use_es = (
            early_stopping_rounds is not None
            and early_stopping_rounds > 0
            and X_val is not None
            and y_val is not None
            and len(X_val) > 0
        )
        if use_es:
            p["early_stopping_rounds"] = int(early_stopping_rounds)
        model = xgb.XGBRegressor(**p)
        if use_es:
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
                **({"sample_weight": sample_weight} if sample_weight is not None else {}),
            )
            bi = getattr(model, "best_iteration", None)
            if bi is not None:
                meta["best_iteration"] = int(bi)
        else:
            model.fit(
                X_train,
                y_train,
                verbose=False,
                **({"sample_weight": sample_weight} if sample_weight is not None else {}),
            )
        return FittedVinylIQRegressor(
            "xgboost",
            model,
            feature_columns,
            target_was_log1p=(target_kind == TARGET_KIND_DOLLAR_LOG1P),
            target_kind=target_kind,
        ), meta

    if family == "lightgbm":
        import lightgbm as lgb

        p = {"random_state": random_state, "verbosity": -1, **params}
        p = _strip_unknown_lgb_params(p)
        obj_l = str(p.get("objective", "regression")).lower()
        if obj_l == "quantile" and "alpha" not in p:
            p["alpha"] = 0.5
        model = lgb.LGBMRegressor(**p)
        use_es = (
            early_stopping_rounds is not None
            and early_stopping_rounds > 0
            and X_val is not None
            and y_val is not None
            and len(X_val) > 0
        )
        if use_es:
            fit_kw: dict[str, Any] = {
                "eval_set": [(X_val, y_val)],
                "callbacks": [
                    lgb.early_stopping(early_stopping_rounds, verbose=False),
                    lgb.log_evaluation(period=0),
                ],
            }
            if sample_weight is not None:
                fit_kw["sample_weight"] = sample_weight
            model.fit(X_train, y_train, **fit_kw)
            bi = getattr(model, "best_iteration_", None)
            if bi is not None:
                meta["best_iteration"] = int(bi)
        else:
            model.fit(
                X_train,
                y_train,
                **({"sample_weight": sample_weight} if sample_weight is not None else {}),
            )
        return FittedVinylIQRegressor(
            "lightgbm",
            model,
            feature_columns,
            target_was_log1p=(target_kind == TARGET_KIND_DOLLAR_LOG1P),
            target_kind=target_kind,
        ), meta

    if family == "catboost":
        from catboost import CatBoostRegressor

        p = {**params}
        p = _strip_unknown_cat_params(p)
        p.setdefault("loss_function", "RMSE")
        p.setdefault("thread_count", -1)
        model = CatBoostRegressor(
            random_seed=random_state,
            verbose=False,
            allow_writing_files=False,
            **p,
        )
        use_es = (
            early_stopping_rounds is not None
            and early_stopping_rounds > 0
            and X_val is not None
            and y_val is not None
            and len(X_val) > 0
        )
        if use_es:
            model.fit(
                X_train,
                y_train,
                eval_set=(X_val, y_val),
                early_stopping_rounds=int(early_stopping_rounds),
                verbose=False,
                **({"sample_weight": sample_weight} if sample_weight is not None else {}),
            )
            bi = model.get_best_iteration()
            if bi is not None:
                meta["best_iteration"] = int(bi)
        else:
            model.fit(
                X_train,
                y_train,
                verbose=False,
                **({"sample_weight": sample_weight} if sample_weight is not None else {}),
            )
        return FittedVinylIQRegressor(
            "catboost",
            model,
            feature_columns,
            target_was_log1p=(target_kind == TARGET_KIND_DOLLAR_LOG1P),
            target_kind=target_kind,
        ), meta

    if family == "sklearn_hist_gbrt":
        from sklearn.ensemble import HistGradientBoostingRegressor

        p = {"random_state": random_state, **params}
        p = _strip_unknown_sklearn_hgb(p)
        model = HistGradientBoostingRegressor(**p)
        model.fit(
            X_train,
            y_train,
            **({"sample_weight": sample_weight} if sample_weight is not None else {}),
        )
        return FittedVinylIQRegressor(
            "sklearn_hist_gbrt",
            model,
            feature_columns,
            target_was_log1p=(target_kind == TARGET_KIND_DOLLAR_LOG1P),
            target_kind=target_kind,
        ), meta

    if family == "sklearn_rf":
        from sklearn.ensemble import RandomForestRegressor

        p = {"random_state": random_state, "n_jobs": -1, **params}
        p = _strip_unknown_rf(p)
        model = RandomForestRegressor(**p)
        model.fit(
            X_train,
            y_train,
            **({"sample_weight": sample_weight} if sample_weight is not None else {}),
        )
        return FittedVinylIQRegressor(
            "sklearn_rf",
            model,
            feature_columns,
            target_was_log1p=(target_kind == TARGET_KIND_DOLLAR_LOG1P),
            target_kind=target_kind,
        ), meta

    if family == "sklearn_et":
        from sklearn.ensemble import ExtraTreesRegressor

        p = {"random_state": random_state, "n_jobs": -1, **params}
        p = _strip_unknown_et(p)
        model = ExtraTreesRegressor(**p)
        model.fit(
            X_train,
            y_train,
            **({"sample_weight": sample_weight} if sample_weight is not None else {}),
        )
        return FittedVinylIQRegressor(
            "sklearn_et",
            model,
            feature_columns,
            target_was_log1p=(target_kind == TARGET_KIND_DOLLAR_LOG1P),
            target_kind=target_kind,
        ), meta

    raise ValueError(f"Unknown model family: {family!r}")


def refit_champion(
    family: str,
    params: dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_columns: list[str],
    *,
    best_iteration: int | None,
    random_state: int = 42,
    target_kind: str = TARGET_KIND_DOLLAR_LOG1P,
    sample_weight: np.ndarray | None = None,
) -> FittedVinylIQRegressor:
    """Refit on full training data; shrink tree count for boosters when ES was used."""
    family = family.strip().lower()
    p = dict(params)

    if family == "xgboost" and best_iteration is not None:
        p["n_estimators"] = max(1, int(best_iteration) + 1)

    if family == "lightgbm" and best_iteration is not None:
        p["n_estimators"] = max(1, int(best_iteration) + 1)

    if family == "catboost" and best_iteration is not None:
        p["iterations"] = max(1, int(best_iteration) + 1)

    reg, _ = fit_regressor(
        family,
        p,
        X_train,
        y_train,
        feature_columns,
        X_val=None,
        y_val=None,
        early_stopping_rounds=None,
        random_state=random_state,
        target_kind=target_kind,
        sample_weight=sample_weight,
    )
    return reg


def _dollars_from_log1p(y_true_log1p: np.ndarray, pred_log1p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    yt = np.expm1(np.asarray(y_true_log1p, dtype=np.float64))
    yp = np.expm1(np.asarray(pred_log1p, dtype=np.float64))
    return yt, yp


def mae_dollars(y_true_log1p: np.ndarray, pred_log1p: np.ndarray) -> float:
    yt, yp = _dollars_from_log1p(y_true_log1p, pred_log1p)
    return float(np.mean(np.abs(yp - yt)))


def wape_dollars(y_true_log1p: np.ndarray, pred_log1p: np.ndarray) -> float:
    """
    Weighted Absolute Percentage Error on dollar prices:
    ``sum(|pred - true|) / sum(|true|)``.

    Interprets average error **relative to total dollar volume** of the batch (unlike MAE, which
    weights cheap and expensive records equally in absolute dollars).
    """
    yt, yp = _dollars_from_log1p(y_true_log1p, pred_log1p)
    den = float(np.sum(np.abs(yt)))
    if den <= 0:
        return float("nan")
    return float(np.sum(np.abs(yp - yt)) / den)


def median_ape_dollars(
    y_true_log1p: np.ndarray,
    pred_log1p: np.ndarray,
    *,
    price_floor: float = 1.0,
) -> float:
    """
    Median absolute percentage error: ``median(|pred - true| / max(true, price_floor))``.

    A **relative** per-row error (robust to skew vs mean APE). ``price_floor`` avoids huge ratios
    when the label is near zero.
    """
    yt, yp = _dollars_from_log1p(y_true_log1p, pred_log1p)
    floor = max(float(price_floor), 1e-9)
    den = np.maximum(yt, floor)
    ape = np.abs(yp - yt) / den
    return float(np.median(ape))


def median_ape_train_median_baseline(
    y_train_log1p: np.ndarray,
    y_eval_log1p: np.ndarray,
    *,
    price_floor: float = 1.0,
) -> float:
    """
    Median APE if we predict ``median(y_train)`` in log1p space for every eval row.

    Use to see whether the model beats a trivial constant (if not, focus on features/labels).
    """
    yt_tr = np.asarray(y_train_log1p, dtype=np.float64)
    y_ev = np.asarray(y_eval_log1p, dtype=np.float64)
    mu = float(np.median(yt_tr))
    pred = np.full_like(y_ev, mu, dtype=np.float64)
    return median_ape_dollars(y_ev, pred, price_floor=price_floor)


def median_ape_dollar_quartiles(
    y_true_log1p: np.ndarray,
    pred_log1p: np.ndarray,
    *,
    price_floor: float = 1.0,
    n_bins: int = 4,
) -> list[float]:
    """
    Median APE within bins of **true** dollar price (quartiles by default).

    Bin 0 is the cheapest true-price slice; bin ``n_bins - 1`` the most expensive.
    Cheap releases often dominate a single headline MdAPE.
    """
    yt, yp = _dollars_from_log1p(y_true_log1p, pred_log1p)
    floor = max(float(price_floor), 1e-9)
    den = np.maximum(yt, floor)
    ape = np.abs(yp - yt) / den
    n = int(n_bins)
    masks = true_dollar_quartile_masks(yt, n_bins=n)
    out: list[float] = []
    for m in masks:
        if not np.any(m):
            out.append(float("nan"))
        else:
            out.append(float(np.median(ape[m])))
    return out


def training_sample_weights_from_anchors(
    anchors: np.ndarray,
    mode: str | None,
) -> np.ndarray | None:
    """
    Optional per-row weights for ``fit_regressor`` (upweight low-dollar anchors).

    Modes: ``None`` / empty / ``off`` → no weights; ``inv_sqrt_anchor`` →
    ``w ∝ 1/sqrt(max(anchor, 1))`` normalized to mean 1.
    """
    if mode is None or not str(mode).strip():
        return None
    key = str(mode).strip().lower()
    if key in ("none", "null", "off", "false", "0", "no"):
        return None
    m = np.maximum(np.asarray(anchors, dtype=np.float64), 1.0)
    if key == "inv_sqrt_anchor":
        w = 1.0 / np.sqrt(m)
        s = float(np.sum(w))
        if s <= 0:
            return None
        w *= float(len(w)) / s
        return w.astype(np.float64)
    raise ValueError(
        f"Unknown tuning.sample_weight mode {mode!r} (use null or inv_sqrt_anchor)"
    )


def _feature_column_index(feature_columns: list[str], name: str) -> int | None:
    try:
        return feature_columns.index(name)
    except ValueError:
        return None


def mutually_exclusive_format_bucket_masks(
    X: np.ndarray,
    feature_columns: list[str],
) -> dict[str, np.ndarray]:
    """
    One bucket per row for slice metrics (priority: box_multi > 7 > 10 > 12 > lp > cd > other).

    Missing format columns (older models) yield all-zero bits → ``other``.
    """
    nrow = int(X.shape[0])

    def col(name: str) -> np.ndarray:
        j = _feature_column_index(feature_columns, name)
        if j is None:
            return np.zeros(nrow, dtype=np.float64)
        return np.asarray(X[:, j], dtype=np.float64)

    c_box = col("is_box_set")
    c_multi = col("is_multi_disc")
    c7 = col("is_7inch")
    c10 = col("is_10inch")
    c12 = col("is_12inch")
    clp = col("is_lp")
    ccd = col("is_cd")

    box_multi = (c_box >= 0.5) | (c_multi >= 0.5)
    seven = (~box_multi) & (c7 >= 0.5)
    ten = (~box_multi) & (~seven) & (c10 >= 0.5)
    twelve = (~box_multi) & (~seven) & (~ten) & (c12 >= 0.5)
    lp = (~box_multi) & (~seven) & (~ten) & (~twelve) & (clp >= 0.5)
    cd = (~box_multi) & (~seven) & (~ten) & (~twelve) & (~lp) & (ccd >= 0.5)
    other = ~(box_multi | seven | ten | twelve | lp | cd)
    return {
        "box_multi": box_multi,
        "seven": seven,
        "ten": ten,
        "twelve": twelve,
        "lp": lp,
        "cd": cd,
        "other": other,
    }


def combine_anchor_and_format_sample_weights(
    anchors: np.ndarray,
    anchor_mode: str | None,
    X: np.ndarray,
    feature_columns: list[str],
    format_multipliers: dict[str, float] | None,
) -> np.ndarray | None:
    """
    Apply optional per-format multipliers on top of ``training_sample_weights_from_anchors``.

    Final weights are renormalized to mean 1. ``format_multipliers`` maps bucket names
    (``box_multi``, ``seven``, … ``other``) and optional ``default`` for unspecified keys.
    """
    base = training_sample_weights_from_anchors(anchors, anchor_mode)
    return apply_format_multipliers_to_weights(base, X, feature_columns, format_multipliers)


def apply_format_multipliers_to_weights(
    base: np.ndarray | None,
    X: np.ndarray,
    feature_columns: list[str],
    mults: dict[str, float] | None,
) -> np.ndarray | None:
    """Multiply per-row weights by format bucket multipliers; renorm to mean 1."""
    if mults is None or not mults:
        return base
    n = int(X.shape[0])
    if base is None:
        w = np.ones(n, dtype=np.float64)
    else:
        w = np.asarray(base, dtype=np.float64).copy()
    default_m = float(mults.get("default", 1.0))
    buckets = mutually_exclusive_format_bucket_masks(X, feature_columns)
    order = ("box_multi", "seven", "ten", "twelve", "lp", "cd", "other")
    for name in order:
        mf = float(mults.get(name, default_m))
        if mf != 1.0 and name in buckets:
            w[buckets[name]] *= mf
    s = float(np.sum(w))
    if s <= 0:
        return base
    w *= float(n) / s
    return w.astype(np.float64)


def weighted_format_median_ape_dollars(
    y_true_log1p: np.ndarray,
    pred_log1p: np.ndarray,
    X: np.ndarray,
    feature_columns: list[str],
    format_weights: dict[str, float],
    *,
    min_count: int = 15,
) -> float:
    """
    Average of per-bucket median APE (log1p dollar space), weighted by ``format_weights``.

    Buckets use the same mutually exclusive masks as slice tables. Buckets with
    fewer than ``min_count`` rows are skipped.
    """
    buckets = mutually_exclusive_format_bucket_masks(X, feature_columns)
    default_w = float(format_weights.get("default", 1.0))
    order = ("box_multi", "seven", "ten", "twelve", "lp", "cd", "other")
    num = 0.0
    den = 0.0
    for name in order:
        m = buckets[name]
        cnt = int(np.sum(m))
        if cnt < int(min_count):
            continue
        md = median_ape_dollars(y_true_log1p[m], pred_log1p[m])
        if not np.isfinite(md):
            continue
        wf = float(format_weights.get(name, default_w))
        num += wf * float(md)
        den += wf
    if den <= 0:
        return float("nan")
    return float(num / den)


def true_dollar_quartile_masks(yt: np.ndarray, *, n_bins: int = 4) -> list[np.ndarray]:
    """
    Boolean masks partitioning rows by **true** dollar price ``yt`` (cheap → expensive).

    Same edges as ``median_ape_dollar_quartiles`` / ``median_ape_quartile_format_slice_table``.
    """
    y = np.asarray(yt, dtype=np.float64)
    n = int(n_bins)
    qs = np.linspace(0.0, 1.0, n + 1)
    edges = np.quantile(y, qs)
    q_masks: list[np.ndarray] = []
    for i in range(n):
        lo, hi = float(edges[i]), float(edges[i + 1])
        if i == n - 1:
            q_masks.append((y >= lo) & (y <= hi))
        else:
            q_masks.append((y >= lo) & (y < hi))
    return q_masks


def median_ape_quartile_format_slice_diagnostics(
    y_true_log1p: np.ndarray,
    pred_log1p: np.ndarray,
    X: np.ndarray,
    feature_columns: list[str],
    *,
    price_floor: float = 1.0,
    n_quartiles: int = 4,
    min_count: int = 15,
) -> list[dict[str, Any]]:
    """
    Per (quartile × format) cell with ``n_rows >= min_count``: median / mean / p90 / max APE.

    Use to sanity-check console lines that show ``0.0%`` (one-decimal formatting can hide small
    non-zero medians; ``max_ape`` reveals heavy tails).
    """
    yt, yp = _dollars_from_log1p(y_true_log1p, pred_log1p)
    floor = max(float(price_floor), 1e-9)
    den = np.maximum(yt, floor)
    ape = np.abs(yp - yt) / den
    q_masks = true_dollar_quartile_masks(yt, n_bins=int(n_quartiles))
    buckets = mutually_exclusive_format_bucket_masks(X, feature_columns)
    order = ("box_multi", "seven", "ten", "twelve", "lp", "cd", "other")
    out: list[dict[str, Any]] = []
    for qi, qm in enumerate(q_masks):
        for name in order:
            mask = qm & buckets[name]
            cnt = int(np.sum(mask))
            if cnt < int(min_count):
                continue
            a = ape[mask]
            out.append(
                {
                    "quartile": qi,
                    "slice": name,
                    "n_rows": cnt,
                    "median_ape": float(np.median(a)),
                    "mean_ape": float(np.mean(a)),
                    "p90_ape": float(np.percentile(a, 90)),
                    "max_ape": float(np.max(a)),
                }
            )
    return out


def median_ape_quartile_format_slice_table(
    y_true_log1p: np.ndarray,
    pred_log1p: np.ndarray,
    X: np.ndarray,
    feature_columns: list[str],
    *,
    price_floor: float = 1.0,
    n_quartiles: int = 4,
    min_count: int = 15,
) -> list[dict[str, Any]]:
    """
    Median APE for each (true-dollar quartile × mutually exclusive format bucket).

    Quartiles match ``median_ape_dollar_quartiles`` (Q1 = cheapest true ``y``).

    **Printing note:** ``100 * median_ape`` at one decimal can show ``0.0%`` when the true
    median APE is below ~0.0005 (0.05%). Use ``median_ape_quartile_format_slice_diagnostics``
    for p90/max when spot-checking.
    """
    yt, yp = _dollars_from_log1p(y_true_log1p, pred_log1p)
    floor = max(float(price_floor), 1e-9)
    den = np.maximum(yt, floor)
    ape = np.abs(yp - yt) / den
    n_bins = int(n_quartiles)
    q_masks = true_dollar_quartile_masks(yt, n_bins=n_bins)

    buckets = mutually_exclusive_format_bucket_masks(X, feature_columns)
    order = ("box_multi", "seven", "ten", "twelve", "lp", "cd", "other")
    out: list[dict[str, Any]] = []
    for qi, qm in enumerate(q_masks):
        for name in order:
            bm = buckets[name]
            mask = qm & bm
            cnt = int(np.sum(mask))
            if cnt < int(min_count):
                md = float("nan")
            else:
                md = float(np.median(ape[mask]))
            out.append(
                {
                    "quartile": qi,
                    "slice": name,
                    "median_ape": md,
                    "n_rows": cnt,
                }
            )
    return out
