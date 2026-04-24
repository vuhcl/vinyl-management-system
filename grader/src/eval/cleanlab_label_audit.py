"""
Train-split label-issue audit using out-of-fold baseline probabilities + Cleanlab.

Loads the same sparse train matrices and LR settings as ``BaselineModel``,
runs stratified K-fold ``predict_proba`` (uncalibrated heads only), then
``cleanlab`` scores / issue flags for human triage.

**Caveats (see plan):**
  - Residual text noise can look like label errors — read flagged rows.
  - ``StratifiedKFold`` needs each class count >= ``n_splits``; we auto-cap
    ``n_splits`` by the rarest class (minimum 2).
  - Cleanlab treats grades as nominal classes, not ordinal distances.
  - Run separate audits per target (sleeve / media).
  - Each output CSV includes both ``sleeve_label`` and ``media_label`` from the
    train row (for context); Cleanlab scores still apply only to ``target``.

**TF-IDF-only fallback:** If ``train_{target}_X.npz`` is missing, refits TF-IDF
on ``TFIDFFeatureBuilder.extract_texts`` and runs OOF LR on that matrix only.
When ``engineered_features`` is enabled in config, that path is **not**
comparable to the full baseline — a warning is logged.

Usage:
    uv run --package vinyl-grader --extra data_quality python -m \\
        grader.src.eval.cleanlab_label_audit --config grader/configs/grader.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import pickle
import warnings
from pathlib import Path
from typing import Any, Optional

import numpy as np
import scipy.sparse as sp
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from grader.src.features.tfidf_features import TFIDFFeatureBuilder
from grader.src.models.baseline import TARGETS, BaselineModel

logger = logging.getLogger(__name__)


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_train_records(splits_dir: Path) -> list[dict]:
    path = splits_dir / "train.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Missing train split: {path}")
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_encoder(artifacts_dir: Path, target: str):
    enc_path = artifacts_dir / f"label_encoder_{target}.pkl"
    if not enc_path.is_file():
        raise FileNotFoundError(f"Missing label encoder: {enc_path}")
    with open(enc_path, "rb") as f:
        return pickle.load(f)


def _load_baseline_lr(artifacts_dir: Path, target: str) -> LogisticRegression:
    p = artifacts_dir / f"baseline_{target}.pkl"
    if not p.is_file():
        raise FileNotFoundError(
            f"Missing baseline pickle {p} — train baseline first "
            "so OOF uses the same C and hyperparameters."
        )
    with open(p, "rb") as f:
        clf = pickle.load(f)
    if not isinstance(clf, LogisticRegression):
        raise TypeError(f"Expected LogisticRegression in {p}, got {type(clf)}")
    return clf


def _lr_kwargs_from_fitted(clf: LogisticRegression) -> dict[str, Any]:
    """Hyperparameters for fresh OOF clones (same as pickled head)."""
    return {
        "C": float(clf.C),
        "max_iter": int(clf.max_iter),
        "tol": float(clf.tol),
        "class_weight": clf.class_weight,
        "solver": clf.solver,
        "random_state": (
            int(clf.random_state)
            if clf.random_state is not None
            else 42
        ),
    }


def _effective_n_splits(
    y: np.ndarray, requested: int, *, n_classes: int
) -> int:
    """StratifiedKFold needs n_splits <= count of rarest class present in y."""
    if y.size == 0:
        raise ValueError("Empty label array.")
    counts = np.bincount(y.astype(int, copy=False), minlength=n_classes)
    positive = counts[counts > 0]
    if positive.size == 0:
        raise ValueError("No labels in y.")
    min_class = int(positive.min())
    k = min(int(requested), min_class)
    if k < 2:
        raise ValueError(
            f"Stratified OOF needs n_splits <= rarest-class count ({min_class}) "
            f"and n_splits >= 2. Got requested={requested}. Add data or lower "
            "n_splits."
        )
    if k < requested:
        logger.warning(
            "Lowering n_splits from %d to %d (rarest class count=%d).",
            requested,
            k,
            min_class,
        )
    return k


def oof_pred_proba_lr(
    X: sp.csr_matrix,
    y: np.ndarray,
    *,
    n_classes: int,
    lr_kwargs: dict[str, Any],
    n_splits: int,
    random_state: int,
) -> tuple[np.ndarray, int]:
    """
    Out-of-fold ``predict_proba`` with shape (n_samples, n_classes).

    ``n_classes`` must match the fitted label encoder (full canonical class
    count), not ``y.max() + 1`` (rare classes may be absent from ``y``).
    """
    y_int = y.astype(np.int64, copy=False)
    n_samples = X.shape[0]
    if y_int.max() >= n_classes or y_int.min() < 0:
        raise ValueError(
            f"y out of range for n_classes={n_classes}: min={y_int.min()} "
            f"max={y_int.max()}"
        )
    probs = np.zeros((n_samples, n_classes), dtype=np.float64)
    k = _effective_n_splits(y_int, n_splits, n_classes=n_classes)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    for train_idx, val_idx in skf.split(X, y_int):
        clf = LogisticRegression(**lr_kwargs)
        clf.fit(X[train_idx], y_int[train_idx])
        p_val = clf.predict_proba(X[val_idx])
        model_classes = clf.classes_.astype(np.int64, copy=False)
        for j, c in enumerate(model_classes):
            probs[val_idx, int(c)] = p_val[:, j]
    return probs, k


def _verify_jsonl_y_alignment(
    records: list[dict],
    y_disk: np.ndarray,
    encoder,
    target: str,
) -> None:
    if len(records) != len(y_disk):
        raise ValueError(
            f"Row count mismatch: train.jsonl has {len(records)} rows but "
            f"train_{target}_y.npy has {len(y_disk)}."
        )
    field = f"{target}_label"
    bad: list[int] = []
    for i, rec in enumerate(records):
        lab = rec[field]
        yi = int(y_disk[i])
        if int(encoder.transform([str(lab)])[0]) != yi:
            bad.append(i)
            if len(bad) >= 5:
                break
    if bad:
        raise ValueError(
            f"Label mismatch vs encoder/y.npy at indices {bad} (target={target}). "
            "Re-run preprocess + tfidf_features so splits match artifacts."
        )


def _build_tfidf_only_train_matrix(
    config_path: str,
    config: dict[str, Any],
    records: list[dict],
    target: str,
) -> tuple[sp.csr_matrix, np.ndarray, LabelEncoder]:
    """Refit TF-IDF on train texts only (no engineered columns)."""
    builder = TFIDFFeatureBuilder(config_path, config=config)
    texts = builder.extract_texts(records)
    vectorizer = builder.build_vectorizer(texts, target)
    labels = builder.extract_labels(records, target)
    encoder = builder.build_encoder(labels, target)
    y = encoder.transform(labels).astype(np.int64, copy=False)
    X = builder.transform_texts(vectorizer, texts)
    return X, y, encoder


def run_target_audit(
    *,
    config: dict[str, Any],
    config_path: str,
    target: str,
    records: list[dict],
    n_splits: int,
    random_state: int,
    use_tfidf_fallback: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """
    Returns (meta, rows) where ``rows`` are dicts ready for CSV/DictWriter.
    """
    try:
        from cleanlab.filter import find_label_issues
        from cleanlab.rank import get_label_quality_scores
    except ImportError as e:
        raise ImportError(
            "cleanlab is required. Install with: "
            "uv sync --package vinyl-grader --extra data_quality"
        ) from e

    artifacts_dir = Path(config["paths"]["artifacts"])
    features_dir = artifacts_dir / "features"
    x_path = features_dir / f"train_{target}_X.npz"
    y_path = features_dir / f"train_{target}_y.npy"
    feature_source = "disk"
    eng_cfg = config.get("models", {}).get("baseline", {}).get(
        "engineered_features", {}
    )
    engineered_on = bool(eng_cfg.get("enabled", False))

    encoder: LabelEncoder
    lr_source: str

    if x_path.is_file() and y_path.is_file():
        X = sp.load_npz(str(x_path))
        y = np.load(str(y_path))
        encoder = _load_encoder(artifacts_dir, target)
        _verify_jsonl_y_alignment(records, y, encoder, target)
        baseline_lr = _load_baseline_lr(artifacts_dir, target)
        lr_kwargs = _lr_kwargs_from_fitted(baseline_lr)
        lr_source = "baseline_pickle"
    else:
        if not use_tfidf_fallback:
            raise FileNotFoundError(
                f"Missing {x_path} or {y_path} and TF-IDF fallback disabled."
            )
        if engineered_on:
            warnings.warn(
                "TF-IDF-only OOF path: engineered_features.enabled is true — "
                "probs will not match the full baseline X.",
                UserWarning,
                stacklevel=2,
            )
        logger.warning(
            "Using TF-IDF-only train matrix (missing %s / %s). "
            "OOF LR uses grader.yaml C (not baseline_*.pkl) because feature "
            "dimensions differ from the pickled baseline.",
            x_path.name,
            y_path.name,
        )
        X, y, encoder = _build_tfidf_only_train_matrix(
            config_path, config, records, target
        )
        feature_source = "tfidf_refit"
        bm = BaselineModel(config_path=config_path, config=config)
        lr_kwargs = _lr_kwargs_from_fitted(bm.build_model(target))
        lr_source = "config_lr_clone"

    n_classes = len(encoder.classes_)
    pred_probs, k_used = oof_pred_proba_lr(
        X,
        y,
        n_classes=n_classes,
        lr_kwargs=lr_kwargs,
        n_splits=n_splits,
        random_state=random_state,
    )

    issues = find_label_issues(y, pred_probs)
    scores = get_label_quality_scores(y, pred_probs, method="self_confidence")

    builder = TFIDFFeatureBuilder(config_path, config=config)
    modeling_texts = builder.extract_texts(records)
    pred_idx = np.argmax(pred_probs, axis=1)
    class_names = [str(c) for c in encoder.classes_]

    rows_out: list[dict[str, Any]] = []
    for i, rec in enumerate(records):
        given = str(rec.get(f"{target}_label", ""))
        pred_label = class_names[int(pred_idx[i])] if pred_idx[i] < len(
            class_names
        ) else ""
        rows_out.append(
            {
                "item_id": rec.get("item_id", ""),
                "source": rec.get("source", ""),
                "label_confidence": rec.get("label_confidence", ""),
                "target": target,
                "sleeve_label": str(rec.get("sleeve_label", "")),
                "media_label": str(rec.get("media_label", "")),
                f"{target}_label": given,
                "oof_pred_label": pred_label,
                "cleanlab_label_issue": bool(issues[i]),
                "cleanlab_self_confidence": float(scores[i]),
                "feature_source": feature_source,
                "oof_n_splits": k_used,
                "modeling_text_snippet": modeling_texts[i][:500],
            }
        )

    meta = {
        "target": target,
        "n_rows": len(records),
        "n_splits_effective": k_used,
        "feature_source": feature_source,
        "lr_C": lr_kwargs["C"],
        "lr_source": lr_source,
        "n_classes": n_classes,
    }
    return meta, rows_out


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Cleanlab label-issue audit on train (OOF baseline probs)."
    )
    parser.add_argument(
        "--config",
        default="grader/configs/grader.yaml",
        help="Path to grader.yaml",
    )
    parser.add_argument(
        "--output-dir",
        default="grader/reports",
        help="Directory for cleanlab_label_audit_<target>.csv",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Stratified K-fold count (capped by rarest-class frequency).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="RNG seed for StratifiedKFold shuffle.",
    )
    parser.add_argument(
        "--targets",
        nargs="*",
        choices=TARGETS,
        default=list(TARGETS),
        help="Subset of targets (default: sleeve media).",
    )
    parser.add_argument(
        "--no-tfidf-fallback",
        action="store_true",
        help="Fail if train_*_X.npz is missing instead of refitting TF-IDF.",
    )
    args = parser.parse_args(argv)

    cfg_path = Path(args.config)
    config = _load_yaml(cfg_path)
    splits_dir = Path(config["paths"]["splits"])
    records = _load_train_records(splits_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for target in args.targets:
        meta, rows = run_target_audit(
            config=config,
            config_path=str(cfg_path),
            target=target,
            records=records,
            n_splits=args.n_splits,
            random_state=args.random_state,
            use_tfidf_fallback=not args.no_tfidf_fallback,
        )
        out_csv = out_dir / f"cleanlab_label_audit_{target}.csv"
        if not rows:
            logger.warning("No rows for target=%s — skipping CSV.", target)
            continue
        fieldnames = list(rows[0].keys())
        with open(out_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        logger.info(
            "Wrote %s (%d rows, n_splits=%s, feature_source=%s, C=%s).",
            out_csv,
            meta["n_rows"],
            meta["n_splits_effective"],
            meta["feature_source"],
            meta["lr_C"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
