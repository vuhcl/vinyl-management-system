"""
End-to-end pipeline: ingest → preprocess → split → train (baseline/transformer) → evaluate → save.

Run from project root:
  PYTHONPATH=. python -m nlp_condition_classifier.src.pipeline
  or
  cd nlp_condition_classifier && PYTHONPATH=.. python -m src.pipeline
"""
from pathlib import Path
import os
import yaml

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .data.ingest import load_labeled_condition_data
from .data.preprocess import preprocess_dataset, load_config as load_preprocess_config
from .models.baseline import train_baseline, BaselineConditionClassifier
from .evaluation.metrics import compute_metrics
from .evaluation.calibration import expected_calibration_error


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_config(config_path: Path | None = None) -> dict:
    if config_path is None:
        config_path = _project_root() / "configs" / "base.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_pipeline(
    config_path: Path | None = None,
    data_dir: Path | None = None,
    processed_dir: Path | None = None,
    artifacts_dir: Path | None = None,
    skip_ingest: bool = False,
    log_mlflow: bool = True,
    phase: str = "baseline",
) -> dict:
    """
    Run full pipeline: load data → preprocess → stratified split → train → evaluate → save.

    phase: "baseline" (TF-IDF + LR) or "transformer" (stub for Phase 2)
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

    # Ingest
    if not skip_ingest:
        discogs_cfg = config.get("discogs", {})
        df = load_labeled_condition_data(
            data_dir=raw_path,
            from_discogs=discogs_cfg.get("use_api", False),
            discogs_token=os.environ.get("DISCOGS_USER_TOKEN") or discogs_cfg.get("token"),
        )
    else:
        import pandas as pd
        csv_path = processed_path / "condition_processed.csv"
        if not csv_path.exists():
            df = load_labeled_condition_data(data_dir=raw_path)
        else:
            df = pd.read_csv(csv_path)
    if df.empty or "seller_notes" not in df.columns:
        raise ValueError(
            "No labeled condition data. Add condition_labeled.csv to data/raw/ with columns: "
            "item_id, seller_notes, sleeve_condition, media_condition"
        )

    # Preprocess
    preprocess_cfg = config.get("preprocess", {})
    df = preprocess_dataset(
        df,
        text_column="seller_notes",
        lowercase=preprocess_cfg.get("lowercase", True),
        remove_urls=preprocess_cfg.get("remove_urls", True),
        strip_whitespace=preprocess_cfg.get("strip_whitespace", True),
        max_length_chars=preprocess_cfg.get("max_length_chars", 2000),
        min_tokens=preprocess_cfg.get("min_tokens", 2),
    )
    text_col = "cleaned_notes" if "cleaned_notes" in df.columns else "seller_notes"
    df = df.dropna(subset=["sleeve_condition", "media_condition"])
    if df.empty:
        raise ValueError("No rows left after dropping missing sleeve/media condition.")
    df.to_csv(processed_path / "condition_processed.csv", index=False)

    X = df[text_col].astype(str).tolist()
    y_sleeve = df["sleeve_condition"].values
    y_media = df["media_condition"].values
    item_ids = df["item_id"].values

    # Stratified split
    split_cfg = config.get("split", {})
    val_frac = split_cfg.get("val_fraction", 0.15)
    test_frac = split_cfg.get("test_fraction", 0.15)
    stratify = split_cfg.get("stratify", True)
    strat = y_sleeve if stratify else None
    X_train, X_rest, y_sleeve_train, y_sleeve_rest, y_media_train, y_media_rest, id_train, id_rest = train_test_split(
        X, y_sleeve, y_media, item_ids, test_size=(val_frac + test_frac), stratify=strat, random_state=seed
    )
    n_rest = len(X_rest)
    test_n = max(1, int(n_rest * test_frac / (val_frac + test_frac)))
    X_val, X_test = X_rest[:-test_n], X_rest[-test_n:]
    y_sleeve_val, y_sleeve_test = y_sleeve_rest[:-test_n], y_sleeve_rest[-test_n:]
    y_media_val, y_media_test = y_media_rest[:-test_n], y_media_rest[-test_n:]
    id_val, id_test = id_rest[:-test_n], id_rest[-test_n:]

    # Train
    if phase == "baseline":
        baseline_cfg = config.get("baseline", {})
        tfidf_cfg = baseline_cfg.get("tfidf", {})
        if "ngram_range" in tfidf_cfg and isinstance(tfidf_cfg["ngram_range"], list):
            tfidf_cfg = {**tfidf_cfg, "ngram_range": tuple(tfidf_cfg["ngram_range"])}
        model = train_baseline(
            X_train,
            y_sleeve_train,
            y_media_train,
            tfidf_config=tfidf_cfg,
            logistic_config=baseline_cfg.get("logistic", {}),
            random_state=seed,
        )
    else:
        from .models.transformer import TransformerConditionClassifier
        trans_cfg = config.get("transformer", {})
        model = TransformerConditionClassifier(
            model_name=trans_cfg.get("model_name", "distilbert-base-uncased"),
            max_length=trans_cfg.get("max_length", 128),
            random_state=seed,
        )
        model.fit(
            X_train,
            y_sleeve=y_sleeve_train,
            y_media=y_media_train,
            epochs=trans_cfg.get("epochs", 3),
            batch_size=trans_cfg.get("batch_size", 16),
            learning_rate=trans_cfg.get("learning_rate", 2e-5),
        )

    # Evaluate
    if phase == "baseline":
        y_sleeve_pred = model.predict_sleeve(X_test)
        y_media_pred = model.predict_media(X_test)
        sleeve_proba = model.predict_proba_sleeve(X_test)
        media_proba = model.predict_proba_media(X_test)
    else:
        y_sleeve_pred = np.array([model.predict_item(iid, txt)["predicted_sleeve_condition"] for iid, txt in zip(id_test, X_test)])
        y_media_pred = np.array([model.predict_item(iid, txt)["predicted_media_condition"] for iid, txt in zip(id_test, X_test)])
        sleeve_proba = np.zeros((len(y_sleeve_test), len(model.classes_)))
        media_proba = np.zeros((len(y_media_test), len(model.classes_)))
    eval_cfg = config.get("evaluation", {})
    n_bins = eval_cfg.get("calibration_bins", 10)
    metrics = {}
    metrics.update(compute_metrics(y_sleeve_test, y_sleeve_pred, prefix="sleeve"))
    metrics.update(compute_metrics(y_media_test, y_media_pred, prefix="media"))
    metrics["sleeve_ece"] = expected_calibration_error(y_sleeve_test, sleeve_proba, n_bins=n_bins)
    metrics["media_ece"] = expected_calibration_error(y_media_test, media_proba, n_bins=n_bins)

    # Save artifacts
    model_dir = artifacts_path / "baseline" if phase == "baseline" else artifacts_path / "transformer"
    model.save(model_dir)

    if log_mlflow:
        try:
            import mlflow
            mlflow_cfg = config.get("mlflow", {})
            mlflow.set_tracking_uri(mlflow_cfg.get("tracking_uri") or "./mlruns")
            mlflow.set_experiment(mlflow_cfg.get("experiment_name", "nlp_condition_classifier"))
            with mlflow.start_run():
                mlflow.log_params({"seed": seed, "phase": phase})
                mlflow.log_metrics(metrics)
                mlflow.log_artifact(str(config_path or _project_root() / "configs" / "base.yaml"))
        except Exception:
            pass

    return {"metrics": metrics, "model_path": str(model_dir)}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--skip-ingest", action="store_true")
    p.add_argument("--no-mlflow", action="store_true", dest="no_mlflow")
    p.add_argument("--phase", choices=["baseline", "transformer"], default="baseline")
    args = p.parse_args()
    result = run_pipeline(
        config_path=args.config,
        skip_ingest=args.skip_ingest,
        log_mlflow=not args.no_mlflow,
        phase=args.phase,
    )
    print("Metrics:", result["metrics"])
    print("Model saved to:", result["model_path"])
