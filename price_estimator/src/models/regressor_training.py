"""Fit / refit VinylIQ regressors across sklearn / boosting backends."""
from __future__ import annotations

from typing import Any

import numpy as np

from .regressor_constants import TARGET_KIND_DOLLAR_LOG1P
from .regressor_fitted import FittedVinylIQRegressor


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
