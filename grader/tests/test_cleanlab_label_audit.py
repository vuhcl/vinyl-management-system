"""Tests for ``grader.src.eval.cleanlab_label_audit``."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from grader.src.eval.cleanlab_label_audit import (
    _effective_n_splits,
    oof_pred_proba_lr,
)


def test_effective_n_splits_caps_by_rarest_class() -> None:
    y = np.array([0, 0, 0, 1, 1, 2, 2], dtype=np.int64)
    assert _effective_n_splits(y, 5, n_classes=3) == 2
    assert _effective_n_splits(y, 2, n_classes=3) == 2


def test_effective_n_splits_raises_when_impossible() -> None:
    y = np.array([0, 1, 2], dtype=np.int64)
    with pytest.raises(ValueError, match="Stratified OOF"):
        _effective_n_splits(y, 5, n_classes=3)


def test_oof_pred_proba_lr_shape_and_stochastic_separability() -> None:
    pytest.importorskip("cleanlab", reason="optional vinyl-grader[data_quality]")
    from cleanlab.filter import find_label_issues
    from cleanlab.rank import get_label_quality_scores

    rng = np.random.default_rng(0)
    n_per = 18
    n_classes = 3
    n_feat = 8
    X_blocks = []
    y_parts = []
    for c in range(n_classes):
        block = rng.standard_normal((n_per, n_feat))
        block[:, c] += 4.0
        X_blocks.append(block)
        y_parts.append(np.full(n_per, c, dtype=np.int64))
    X = sp.csr_matrix(np.vstack(X_blocks))
    y = np.concatenate(y_parts)
    bad_idx = n_per + 3
    y_bad = y.copy()
    y_bad[bad_idx] = (y_bad[bad_idx] + 1) % n_classes

    lr_kwargs = {
        "C": 1.0,
        "max_iter": 2000,
        "tol": 1e-4,
        "class_weight": None,
        "solver": "lbfgs",
        "random_state": 42,
    }
    probs, k = oof_pred_proba_lr(
        X,
        y_bad,
        n_classes=n_classes,
        lr_kwargs=lr_kwargs,
        n_splits=3,
        random_state=0,
    )
    assert probs.shape == (len(y_bad), n_classes)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
    assert k == 3

    issues = find_label_issues(y_bad, probs)
    scores = get_label_quality_scores(y_bad, probs, method="self_confidence")
    assert issues[bad_idx] or scores[bad_idx] <= np.median(scores)


def test_run_target_audit_disk_path_roundtrip(tmp_path: Path) -> None:
    pytest.importorskip("cleanlab", reason="optional vinyl-grader[data_quality]")
    from grader.src.eval.cleanlab_label_audit import run_target_audit

    artifacts = tmp_path / "artifacts"
    feats = artifacts / "features"
    splits = tmp_path / "splits"
    feats.mkdir(parents=True)
    splits.mkdir(parents=True)

    rng = np.random.default_rng(1)
    n, n_feat = 24, 6
    X = sp.csr_matrix(rng.standard_normal((n, n_feat)))
    grades = ["Mint", "Near Mint", "Very Good"]
    enc = LabelEncoder()
    enc.fit(grades)
    y = rng.integers(0, 3, size=n, dtype=np.int64)
    sp.save_npz(feats / "train_sleeve_X.npz", X)
    np.save(feats / "train_sleeve_y.npy", y)

    with open(artifacts / "label_encoder_sleeve.pkl", "wb") as f:
        pickle.dump(enc, f)

    lr = LogisticRegression(
        C=0.5,
        max_iter=2000,
        tol=1e-4,
        class_weight=None,
        solver="lbfgs",
        random_state=42,
    )
    lr.fit(X, y)
    with open(artifacts / "baseline_sleeve.pkl", "wb") as f:
        pickle.dump(lr, f)

    records = []
    for i in range(n):
        records.append(
            {
                "item_id": str(i),
                "source": "test",
                "text": f"note {i}",
                "text_clean": f"note {i}",
                "sleeve_label": str(enc.inverse_transform([int(y[i])])[0]),
                "media_label": "Near Mint",
                "label_confidence": 1.0,
            }
        )
    with open(splits / "train.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    config = {
        "paths": {"artifacts": str(artifacts), "splits": str(splits)},
        "models": {
            "baseline": {
                "tfidf": {
                    "max_features": 200,
                    "ngram_range": [1, 2],
                    "sublinear_tf": True,
                    "min_df": 1,
                },
                "logistic_regression": {
                    "C": 1.0,
                    "max_iter": 2000,
                    "tol": 1e-4,
                    "class_weight": None,
                    "solver": "lbfgs",
                    "random_state": 42,
                },
                "tuning": {"enabled": False},
                "engineered_features": {"enabled": False},
            }
        },
        "evaluation": {"calibration": {"method": "isotonic"}},
        "preprocessing": {},
    }
    cfg_path = tmp_path / "grader.yaml"
    cfg_path.write_text(yaml.dump(config, default_flow_style=False), encoding="utf-8")

    meta, rows = run_target_audit(
        config=config,
        config_path=str(cfg_path),
        target="sleeve",
        records=records,
        n_splits=2,
        random_state=0,
        use_tfidf_fallback=True,
    )
    assert meta["n_rows"] == n
    assert meta["feature_source"] == "disk"
    assert len(rows) == n
    assert "cleanlab_label_issue" in rows[0]
    assert "cleanlab_self_confidence" in rows[0]
