"""Tuning CV folds, constraints, composite objective, and champion pick."""
from __future__ import annotations

import numpy as np

from price_estimator.src.training.vinyliq_tuning_selection import (
    base_selection_score,
    build_cv_fold_val_release_sets,
    build_trial_record,
    is_feasible,
    parse_selection_objective,
    parse_tuning_constraints,
    pick_champion_trial,
    violation_slack,
)


def test_parse_selection_composite() -> None:
    obj = parse_selection_objective(
        {
            "selection_metric": "composite",
            "selection_composite": {
                "w_mdape": 2.0,
                "w_wape": 1.0,
                "w_mae": 0.5,
                "mae_ref_usd": 25.0,
            },
        }
    )
    assert obj.composite is True
    assert obj.mlflow_name == "val_selection_composite"
    s = base_selection_score(obj, mdape=0.1, mae=10.0, wape=0.2)
    assert abs(s - (2.0 * 0.1 + 1.0 * 0.2 + 0.5 * (10.0 / 25.0))) < 1e-9


def test_parse_constraints_defaults() -> None:
    c = parse_tuning_constraints({})
    assert c.enabled is False
    assert c.mdape_max == 0.05
    assert c.wape_max == 0.20
    assert c.violation_fallback == "best_violation"


def test_is_feasible_and_slack() -> None:
    c = parse_tuning_constraints(
        {"constraints": {"enabled": True, "mdape_max": 0.05, "wape_max": 0.2}}
    )
    assert is_feasible(0.04, 0.1, c)
    assert not is_feasible(0.05, 0.1, c)
    assert violation_slack(0.04, 0.1, c) == 0.0
    assert violation_slack(0.1, 0.1, c) > 0.0


def test_pick_champion_feasible_prefers_lower_score() -> None:
    cons = parse_tuning_constraints(
        {"constraints": {"enabled": True, "mdape_max": 0.1, "wape_max": 0.5}}
    )
    obj = parse_selection_objective({"selection_metric": "median_ape"})
    a = build_trial_record(
        family="x",
        params={},
        mdapes=[0.08],
        maes=[5.0],
        wapes=[0.3],
        best_iters=[10],
        cv_agg="mean",
        cons=cons,
        sel_obj=obj,
        cv_folds_used=1,
    )
    b = build_trial_record(
        family="x",
        params={},
        mdapes=[0.06],
        maes=[5.0],
        wapes=[0.3],
        best_iters=[10],
        cv_agg="mean",
        cons=cons,
        sel_obj=obj,
        cv_folds_used=1,
    )
    assert a is not None and b is not None
    best, reason = pick_champion_trial([a, b], cons)
    assert best is not None
    assert best.val_mdape == 0.06
    assert reason == "feasible"


def test_pick_champion_best_violation() -> None:
    cons = parse_tuning_constraints(
        {
            "constraints": {
                "enabled": True,
                "mdape_max": 0.05,
                "wape_max": 0.2,
                "violation_fallback": "best_violation",
            }
        }
    )
    obj = parse_selection_objective({"selection_metric": "median_ape"})
    hi = build_trial_record(
        family="x",
        params={},
        mdapes=[0.2],
        maes=[5.0],
        wapes=[0.5],
        best_iters=[10],
        cv_agg="mean",
        cons=cons,
        sel_obj=obj,
        cv_folds_used=1,
    )
    lo = build_trial_record(
        family="x",
        params={},
        mdapes=[0.06],
        maes=[5.0],
        wapes=[0.25],
        best_iters=[10],
        cv_agg="mean",
        cons=cons,
        sel_obj=obj,
        cv_folds_used=1,
    )
    assert hi is not None and lo is not None
    assert not hi.feasible and not lo.feasible
    best, reason = pick_champion_trial([hi, lo], cons)
    assert best is not None
    assert reason == "best_violation"
    assert best.val_mdape == 0.06


def test_pick_champion_penalize() -> None:
    cons = parse_tuning_constraints(
        {
            "constraints": {
                "enabled": True,
                "mdape_max": 0.05,
                "wape_max": 0.2,
                "violation_fallback": "penalize",
            },
            "constraint_penalty": {"lambda_mdape": 1.0, "lambda_wape": 1.0},
        }
    )
    obj = parse_selection_objective({"selection_metric": "median_ape"})
    a = build_trial_record(
        family="x",
        params={},
        mdapes=[0.5],
        maes=[1.0],
        wapes=[0.5],
        best_iters=[1],
        cv_agg="mean",
        cons=cons,
        sel_obj=obj,
        cv_folds_used=1,
    )
    b = build_trial_record(
        family="x",
        params={},
        mdapes=[0.06],
        maes=[1.0],
        wapes=[0.25],
        best_iters=[1],
        cv_agg="mean",
        cons=cons,
        sel_obj=obj,
        cv_folds_used=1,
    )
    assert a is not None and b is not None
    assert not a.feasible and not b.feasible
    best, reason = pick_champion_trial([a, b], cons)
    assert best is not None
    assert reason == "penalize"
    assert best.pen_score == min(a.pen_score, b.pen_score)


def test_pick_champion_abort() -> None:
    cons = parse_tuning_constraints(
        {
            "constraints": {
                "enabled": True,
                "mdape_max": 0.01,
                "wape_max": 0.01,
                "violation_fallback": "abort",
            }
        }
    )
    obj = parse_selection_objective({"selection_metric": "median_ape"})
    t = build_trial_record(
        family="x",
        params={},
        mdapes=[0.5],
        maes=[5.0],
        wapes=[0.5],
        best_iters=[10],
        cv_agg="mean",
        cons=cons,
        sel_obj=obj,
        cv_folds_used=1,
    )
    assert t is not None
    best, reason = pick_champion_trial([t], cons)
    assert best is None and reason == "abort"


def test_build_cv_folds_partition_train_releases() -> None:
    rids = ["a"] * 3 + ["b"] * 3 + ["c"] * 3 + ["d"] * 3
    med = np.array([10.0] * 12, dtype=np.float64)
    train_r = {"a", "b", "c", "d"}
    folds = build_cv_fold_val_release_sets(
        train_r, rids, med, k=2, seed=0, stratify=None
    )
    assert len(folds) == 2
    u = folds[0] | folds[1]
    assert u == train_r
    assert len(folds[0] & folds[1]) == 0


def test_build_cv_folds_stratify_anchor_quartile() -> None:
    rids = (
        ["cheap"] * 10
        + ["mid1"] * 10
        + ["mid2"] * 10
        + ["pricy"] * 10
        + ["top"] * 10
    )
    med = np.array(
        [1.0] * 10 + [20.0] * 10 + [30.0] * 10 + [80.0] * 10 + [200.0] * 10,
        dtype=np.float64,
    )
    train_r = {"cheap", "mid1", "mid2", "pricy", "top"}
    folds = build_cv_fold_val_release_sets(
        train_r, rids, med, k=3, seed=1, stratify="anchor_quartile"
    )
    assert len(folds) == 3
    assert folds[0] | folds[1] | folds[2] == train_r
