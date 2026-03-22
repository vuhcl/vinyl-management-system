"""
End-to-end pipeline: ingest → preprocess → features → train → evaluate → save.

Run from project root:
  PYTHONPATH=. python -m price_estimator.src.pipeline
"""
from pathlib import Path
import os
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml

from .data.ingest import load_sales_and_metadata
from .data.preprocess import preprocess_price_data
from .features.historical_price import build_historical_price_features
from .features.condition_features import encode_condition_features
from .features.embeddings import build_genre_artist_features
from .models.baseline import train_baseline, BaselinePriceModel
from .models.gradient_boosting import train_gradient_boosting, GradientBoostingPriceModel
from .evaluation.metrics import compute_metrics
from .evaluation.prediction_interval import prediction_interval_coverage, interval_width


def _project_root() -> Path:
    """Price estimator subproject root (price_estimator/)."""
    return Path(__file__).resolve().parents[1]


def load_config(config_path: Path | None = None) -> dict:
    if config_path is None:
        config_path = _project_root() / "configs" / "base.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def _build_item_level_dataset(
    df: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, pd.Series, np.ndarray, dict[str, Any]]:
    """Build one row per item_id. Returns (X, y, item_ids, encoder_state)."""
    feat_cfg = config.get("features", {})
    hist = build_historical_price_features(
        df,
        group_col="item_id",
        price_col="sale_price",
        date_col="sale_date",
        rolling_window_days=feat_cfg.get("rolling_window_days", 180),
        time_decay_halflife_days=feat_cfg.get("time_decay_halflife_days"),
    )
    target_df = df.groupby("item_id", as_index=False)["sale_price"].median()
    target_df = target_df.rename(columns={"sale_price": "target_price"})
    out = hist.merge(target_df, on="item_id", how="inner")

    if "sleeve_condition" in df.columns or "media_condition" in df.columns:
        agg_spec = {}
        if "sleeve_condition" in df.columns:
            agg_spec["sleeve_condition"] = ("sleeve_condition", "first")
        if "media_condition" in df.columns:
            agg_spec["media_condition"] = ("media_condition", "first")
        if agg_spec:
            cond_agg = df.groupby("item_id").agg(**agg_spec).reset_index()
            out = out.merge(cond_agg, on="item_id", how="left")
        out = encode_condition_features(
            out,
            encode=feat_cfg.get("condition_encode", "one_hot"),
            prefix="cond",
        )

    for col in ["genre", "artist", "release_year"]:
        if col in df.columns:
            first = df.groupby("item_id", as_index=False)[col].first()
            out = out.merge(first, on="item_id", how="left")
    out, encoder_state = build_genre_artist_features(
        out,
        genre_top_k=feat_cfg.get("genre_top_k", 50),
        artist_top_k=min(100, feat_cfg.get("genre_top_k", 50) * 2),
    )

    exclude = {
        "item_id", "target_price", "sleeve_condition", "media_condition",
        "genre", "artist", "release_year", "year_col",
    }
    feature_cols = [c for c in out.columns if c not in exclude]
    if not feature_cols:
        feature_cols = [c for c in out.columns if c not in ("item_id", "target_price")]
    X = out[feature_cols].fillna(0)
    y = out["target_price"]
    item_ids = out["item_id"].values
    encoder_state["feature_names"] = list(X.columns)
    return X, y, item_ids, encoder_state


def _time_based_split(
    df: pd.DataFrame,
    item_id_col: str = "item_id",
    date_col: str = "sale_date",
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assign each item to train/val/test by latest sale date. Returns (train, val, test) item_ids."""
    latest = df.groupby(item_id_col)[date_col].max().reset_index()
    latest = latest.sort_values(date_col)
    n = len(latest)
    n_val = max(1, int(n * val_fraction))
    n_test = max(1, int(n * test_fraction))
    n_train = n - n_val - n_test
    train_items = latest[item_id_col].iloc[:n_train].values
    val_items = latest[item_id_col].iloc[n_train : n_train + n_val].values
    test_items = latest[item_id_col].iloc[n_train + n_val :].values
    return train_items, val_items, test_items


def run_pipeline(
    config_path: Path | None = None,
    data_dir: Path | None = None,
    processed_dir: Path | None = None,
    artifacts_dir: Path | None = None,
    skip_ingest: bool = False,
    log_mlflow: bool = True,
    phase: str = "baseline",
) -> dict[str, Any]:
    """
    Run full pipeline: load data → preprocess → features → time-based split → train → evaluate → save.
    phase: "baseline" (Linear Regression) or "gradient_boosting" (LightGBM + prediction intervals).
    """
    config = load_config(config_path)
    root = _project_root()
    seed = config.get("seed", 42)
    np.random.seed(seed)

    paths = config.get("paths", {})
    raw_path = data_dir or root / paths.get("raw_data", "data/raw")
    processed_path = processed_dir or root / paths.get("processed_data", "data/processed")
    artifacts_path = artifacts_dir or root / paths.get("artifacts", "artifacts")
    raw_path = raw_path if raw_path.is_absolute() else root / raw_path
    processed_path = processed_path if processed_path.is_absolute() else root / processed_path
    artifacts_path = artifacts_path if artifacts_path.is_absolute() else root / artifacts_path
    artifacts_path.mkdir(parents=True, exist_ok=True)
    processed_path.mkdir(parents=True, exist_ok=True)

    if not skip_ingest:
        sales, metadata = load_sales_and_metadata(
            data_dir=raw_path,
            from_discogs=config.get("discogs", {}).get("use_api", False),
            discogs_token=os.environ.get("DISCOGS_USER_TOKEN") or config.get("discogs", {}).get("token"),
        )
    else:
        sales_path = processed_path / "sales_processed.csv"
        if not sales_path.exists():
            sales, metadata = load_sales_and_metadata(data_dir=raw_path)
        else:
            sales = pd.read_csv(sales_path)
            sales["sale_date"] = pd.to_datetime(sales["sale_date"])
            metadata = (
                pd.read_csv(processed_path / "metadata.csv")
                if (processed_path / "metadata.csv").exists()
                else pd.DataFrame()
            )

    if sales.empty or "sale_price" not in sales.columns:
        raise ValueError(
            "No sales data. Add data/raw/sales.csv with columns: item_id, sale_price, sale_date."
        )

    feat_cfg = config.get("features", {})
    df = preprocess_price_data(
        sales,
        metadata if not metadata.empty else None,
        max_years_back=feat_cfg.get("historical_years", 3),
    )
    df.to_csv(processed_path / "sales_processed.csv", index=False)
    if not metadata.empty:
        metadata.to_csv(processed_path / "metadata.csv", index=False)

    X, y, item_ids, encoder_state = _build_item_level_dataset(df, config)
    if X.empty or len(y) < 10:
        raise ValueError("Not enough items after feature building (need at least 10).")

    split_cfg = config.get("split", {})
    val_frac = split_cfg.get("val_fraction", 0.15)
    test_frac = split_cfg.get("test_fraction", 0.15)
    date_col = split_cfg.get("time_column", "sale_date")
    train_items, val_items, test_items = _time_based_split(
        df, date_col=date_col, val_fraction=val_frac, test_fraction=test_frac,
    )
    train_mask = np.isin(item_ids, train_items)
    val_mask = np.isin(item_ids, val_items)
    test_mask = np.isin(item_ids, test_items)
    train_idx = np.where(train_mask)[0]
    val_idx = np.where(val_mask)[0]
    test_idx = np.where(test_mask)[0]

    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]

    if phase == "baseline":
        base_cfg = config.get("baseline", {})
        model = train_baseline(
            X_train, y_train,
            fit_intercept=base_cfg.get("fit_intercept", True),
            positive=base_cfg.get("positive", False),
        )
        y_pred = model.predict(X_test)
        metrics = compute_metrics(
            y_test.values if hasattr(y_test, "values") else y_test, y_pred
        )
    else:
        gb_cfg = config.get("gradient_boosting", {})
        quants = gb_cfg.get("prediction_interval_quantiles", [0.1, 0.9])
        model = train_gradient_boosting(
            X_train, y_train,
            n_estimators=gb_cfg.get("n_estimators", 200),
            max_depth=gb_cfg.get("max_depth", 6),
            learning_rate=gb_cfg.get("learning_rate", 0.05),
            subsample=gb_cfg.get("subsample", 0.8),
            colsample_bytree=gb_cfg.get("colsample_bytree", 0.8),
            min_child_samples=gb_cfg.get("min_child_samples", 20),
            prediction_interval_quantiles=quants,
            random_state=seed,
        )
        point, lower, upper = model.predict_interval(X_test, quants[0], quants[-1])
        y_pred = point
        metrics = compute_metrics(
            y_test.values if hasattr(y_test, "values") else y_test, y_pred
        )
        metrics["interval_coverage"] = prediction_interval_coverage(
            y_test.values if hasattr(y_test, "values") else y_test, lower, upper,
        )
        metrics["interval_width"] = interval_width(lower, upper)

    model_dir = artifacts_path / ("baseline" if phase == "baseline" else "gradient_boosting")
    model.save(model_dir)
    joblib.dump(encoder_state, model_dir / "encoder_state.joblib")

    if log_mlflow:
        try:
            import mlflow
            mlflow_cfg = config.get("mlflow", {})
            mlflow.set_tracking_uri(mlflow_cfg.get("tracking_uri") or "./mlruns")
            mlflow.set_experiment(mlflow_cfg.get("experiment_name", "price_estimator"))
            with mlflow.start_run():
                mlflow.log_params({"seed": seed, "phase": phase})
                mlflow.log_metrics(metrics)
                mlflow.log_artifact(str(config_path or _project_root() / "configs" / "base.yaml"))
        except Exception:
            pass

    return {"metrics": metrics, "model_path": str(model_dir)}


def estimate(
    release_id: str,
    sleeve_condition: str | None = None,
    media_condition: str | None = None,
    artifacts_dir: Path | None = None,
    *,
    features_row: pd.DataFrame | None = None,
    phase: str = "baseline",
) -> dict:
    """
    Return price estimate for a release. Loads model from artifacts_dir.
    If features_row is provided (one row with model's feature columns), returns
    predicted_price and prediction_interval; otherwise returns stub with status "no_features".
    """
    root = _project_root()
    cfg = load_config()
    paths = cfg.get("paths", {})
    art = Path(artifacts_dir or root / paths.get("artifacts", "artifacts"))
    art = art if art.is_absolute() else root / art
    model_dir = art / ("baseline" if phase == "baseline" else "gradient_boosting")
    if not model_dir.exists():
        return {
            "release_id": release_id,
            "sleeve_condition": sleeve_condition,
            "media_condition": media_condition,
            "estimate_usd": None,
            "interval_low": None,
            "interval_high": None,
            "status": "no_model",
        }
    if features_row is None:
        return {
            "release_id": release_id,
            "sleeve_condition": sleeve_condition,
            "media_condition": media_condition,
            "estimate_usd": None,
            "interval_low": None,
            "interval_high": None,
            "status": "no_features",
        }
    try:
        if phase == "baseline":
            model = BaselinePriceModel.load(model_dir)
            out = model.predict_item(str(release_id), features_row)
        else:
            model = GradientBoostingPriceModel.load(model_dir)
            out = model.predict_item(
                str(release_id), features_row,
                lower_quantile=0.1, upper_quantile=0.9,
            )
        return {
            "release_id": out["item_id"],
            "sleeve_condition": sleeve_condition,
            "media_condition": media_condition,
            "estimate_usd": out["predicted_price"],
            "interval_low": out["prediction_interval"][0],
            "interval_high": out["prediction_interval"][1],
            "status": "ok",
        }
    except Exception:
        return {
            "release_id": release_id,
            "sleeve_condition": sleeve_condition,
            "media_condition": media_condition,
            "estimate_usd": None,
            "interval_low": None,
            "interval_high": None,
            "status": "error",
        }


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--skip-ingest", action="store_true")
    p.add_argument("--no-mlflow", action="store_true", dest="no_mlflow")
    p.add_argument("--phase", choices=["baseline", "gradient_boosting"], default="baseline")
    args = p.parse_args()
    result = run_pipeline(
        config_path=args.config,
        skip_ingest=args.skip_ingest,
        log_mlflow=not args.no_mlflow,
        phase=args.phase,
    )
    print("Metrics:", result["metrics"])
    print("Model saved to:", result["model_path"])
