"""Tests for the champion/challenger gate and cohort drift (synthetic predictions)."""
from __future__ import annotations

import numpy as np
import pytest

from grader.src.monitoring.cohort_drift import cohort_drift
from grader.src.monitoring.retrain_gate import evaluate_gate
from grader.src.monitoring.cli import DEFAULT_THRESHOLDS, _thresholds

pytestmark = pytest.mark.monitoring

GRADES = ["M", "NM", "VG+", "VG", "G+"]


def _rows(n: int, acc_sleeve: float, acc_media: float, seed: int, text_mu: float = 200.0):
    rng = np.random.default_rng(seed)
    truth_rng = np.random.default_rng(0)  # same truth for every model
    t_s = truth_rng.choice(GRADES, n, p=[0.05, 0.4, 0.3, 0.2, 0.05])
    t_m = truth_rng.choice(GRADES, n, p=[0.05, 0.4, 0.3, 0.2, 0.05])
    rows = []
    for i in range(n):
        ps = t_s[i] if rng.random() < acc_sleeve else rng.choice(GRADES)
        pm = t_m[i] if rng.random() < acc_media else rng.choice(GRADES)
        rows.append(
            {
                "item_id": str(i), "true_sleeve": t_s[i], "true_media": t_m[i],
                "pred_sleeve": ps, "pred_media": pm,
                "sleeve_conf": float(rng.uniform(0.4, 1)), "media_conf": float(rng.uniform(0.4, 1)),
                "text_len": float(rng.normal(text_mu, 40)),
            }
        )
    return rows


@pytest.fixture
def cfg():
    return _thresholds(str(DEFAULT_THRESHOLDS))


def test_gate_promotes_clear_improvement(cfg):
    champ = _rows(1500, 0.70, 0.75, seed=1)
    chal = _rows(1500, 0.85, 0.88, seed=2)
    r = evaluate_gate(champ, chal, cfg["retrain_gate"])
    assert r["promote"] and r["mean_macro_f1_delta"] > 0.05
    assert r["bootstrap_ci95"][0] > 0


def test_gate_rejects_no_gain(cfg):
    champ = _rows(1500, 0.80, 0.80, seed=1)
    r = evaluate_gate(champ, list(champ), cfg["retrain_gate"])
    assert not r["promote"]


def test_gate_rejects_single_target_regression_even_if_mean_improves(cfg):
    champ = _rows(2000, 0.60, 0.80, seed=1)
    chal = _rows(2000, 0.95, 0.70, seed=2)  # big sleeve gain, media drops ~0.10
    r = evaluate_gate(champ, chal, cfg["retrain_gate"])
    assert r["mean_macro_f1_delta"] >= cfg["retrain_gate"]["min_mean_macro_f1_gain"]
    assert not r["promote"]
    assert any("media regressed" in x for x in r["reasons"])


def test_gate_rejects_mismatched_items(cfg):
    champ = _rows(100, 0.8, 0.8, seed=1)
    with pytest.raises(ValueError):
        evaluate_gate(champ, champ[:-1], cfg["retrain_gate"])


def test_hard_ci_gate_blocks_noisy_win(cfg):
    g = dict(cfg["retrain_gate"], require_ci_lower_gt_zero=True, min_mean_macro_f1_gain=0.0)
    champ = _rows(120, 0.80, 0.80, seed=1)
    chal = _rows(120, 0.81, 0.81, seed=2)
    r = evaluate_gate(champ, chal, g)
    if r["bootstrap_ci95"][0] <= 0:
        assert not r["promote"]


def test_cohort_drift_flags_shifted_cohort(cfg):
    ref = _rows(1500, 0.8, 0.8, seed=1, text_mu=200)
    cur = _rows(1500, 0.8, 0.8, seed=3, text_mu=60)  # much shorter text
    r = cohort_drift(ref, cur, cfg["cohort_drift"])
    assert "text_len" in r["flagged"] and r["drift_detected"]


def test_cohort_drift_quiet_on_same_distribution(cfg):
    ref = _rows(1500, 0.8, 0.8, seed=1)
    cur = _rows(1500, 0.8, 0.8, seed=4)
    r = cohort_drift(ref, cur, cfg["cohort_drift"])
    assert "text_len" not in r["flagged"]


def test_report_includes_accuracy_and_within_one_grade(cfg):
    champ = _rows(500, 0.7, 0.7, seed=1)
    r = evaluate_gate(champ, list(champ), cfg["retrain_gate"])
    t = r["per_target"]["media"]
    assert 0 <= t["accuracy"]["champion"] <= 1
    assert t["within_one_grade"]["rows_scored"]["champion"] >= 0


def test_within_one_grade_excludes_unordered_labels():
    from grader.src.monitoring.retrain_gate import within_one_grade
    rows = [
        {"true_sleeve": "Mint", "pred_sleeve": "Near Mint"},      # adjacent: ok
        {"true_sleeve": "Mint", "pred_sleeve": "Good"},           # far: not ok
        {"true_sleeve": "Generic", "pred_sleeve": "Mint"},        # excluded
    ]
    share, n = within_one_grade(rows, "sleeve")
    assert n == 2 and share == 0.5
