#!/usr/bin/env python3
"""
Run a **synthetic** end-to-end benchmark for the vinyl **grader** package and
write JSON suitable for resume talking points.

This uses the same record recipe as ``grader/tests/conftest.py`` (canonical
grades from ``grading_guidelines.yaml`` + representative texts per grade).

**Important:** Numbers are on *synthetic* data for pipeline validation — for
production claims, run ``python -m grader.src.pipeline train --baseline-only``
on real ingested listings and read MLflow / saved reports.

Usage (from repo root, venv active):

  python scripts/grader_eval_resume.py
  python scripts/grader_eval_resume.py --output artifacts/grader_eval_resume.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _build_feature_matrices(
    records: list[dict],
    random_state: int = 42,
):
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder

    from grader.src.eval.synthetic_data import load_canonical_grades

    guidelines = _REPO_ROOT / "grader/configs/grading_guidelines.yaml"
    sleeve_grades, media_grades = load_canonical_grades(guidelines)

    texts = [r["text"] for r in records]
    n = len(texts)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    idx = {
        "train": list(range(0, train_end)),
        "val": list(range(train_end, val_end)),
        "test": list(range(val_end, n)),
    }

    vectorizers = {}
    for target in ("sleeve", "media"):
        vec = TfidfVectorizer(max_features=200, ngram_range=(1, 2), min_df=1)
        vec.fit(texts)
        vectorizers[target] = vec

    encoders = {}
    for grades, target in ((sleeve_grades, "sleeve"), (media_grades, "media")):
        enc = LabelEncoder()
        enc.fit(grades)
        encoders[target] = enc

    features: dict = {}
    for split, split_idx in idx.items():
        features[split] = {}
        split_texts = [texts[i] for i in split_idx]
        for target in ("sleeve", "media"):
            labels = [records[i][f"{target}_label"] for i in split_idx]
            X = vectorizers[target].transform(split_texts)
            y = encoders[target].transform(labels)
            features[split][target] = {"X": X, "y": y}

    models = {}
    for target in ("sleeve", "media"):
        X_train = features["train"][target]["X"]
        y_train = features["train"][target]["y"]
        lr = LogisticRegression(
            max_iter=200,
            random_state=random_state,
            class_weight="balanced",
            solver="lbfgs",
        )
        lr.fit(X_train, y_train)
        models[target] = lr

    return features, encoders, models


def _evaluate_like_baseline(
    features: dict,
    encoders: dict,
    calibrated: dict,
    split: str,
):
    from grader.src.models.baseline import BaselineModel

    bm = BaselineModel.__new__(BaselineModel)
    bm.encoders = encoders
    bm.calibrated = calibrated
    return BaselineModel.evaluate(bm, features, split)


def _compact_metrics(raw: dict) -> dict:
    """Drop huge per-class dicts from classification_report for summary JSON."""
    out = {}
    for target, m in raw.items():
        out[target] = {
            "macro_f1": float(m["macro_f1"]),
            "accuracy": float(m["accuracy"]),
            "ece": float(m["ece"]),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Synthetic grader benchmark → JSON (resume / portfolio)."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_REPO_ROOT / "artifacts" / "grader_eval_resume.json",
        help="Where to write metrics JSON",
    )
    parser.add_argument(
        "--guidelines",
        type=Path,
        default=_REPO_ROOT / "grader" / "configs" / "grading_guidelines.yaml",
        help="Canonical grade schema YAML",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Seed for LR (matches test fixtures)",
    )
    args = parser.parse_args()

    from grader.src.eval.synthetic_data import (
        build_synthetic_unified_records,
    )

    records = build_synthetic_unified_records(args.guidelines)
    features, encoders, models = _build_feature_matrices(
        records, random_state=args.random_state
    )
    # Same rationale as tests: raw LR as "calibrated" avoids CV subset issues.
    test_metrics = _evaluate_like_baseline(
        features, encoders, models, "test"
    )
    val_metrics = _evaluate_like_baseline(features, encoders, models, "val")
    train_metrics = _evaluate_like_baseline(
        features, encoders, models, "train"
    )

    payload = {
        "package": "grader",
        "benchmark": "synthetic_unified_records",
        "n_records": len(records),
        "guidelines": str(args.guidelines.relative_to(_REPO_ROOT)),
        "random_state": args.random_state,
        "disclaimer": (
            "Synthetic benchmark aligned with grader test fixtures; use real "
            "train/test metrics from grader pipeline for production claims."
        ),
        "metrics": {
            "train": _compact_metrics(train_metrics),
            "val": _compact_metrics(val_metrics),
            "test": _compact_metrics(test_metrics),
        },
        "primary_resume_numbers": {
            "test_macro_f1_sleeve": test_metrics["sleeve"]["macro_f1"],
            "test_macro_f1_media": test_metrics["media"]["macro_f1"],
            "test_accuracy_sleeve": test_metrics["sleeve"]["accuracy"],
            "test_accuracy_media": test_metrics["media"]["accuracy"],
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload["primary_resume_numbers"], indent=2))
    print(f"Wrote full report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
