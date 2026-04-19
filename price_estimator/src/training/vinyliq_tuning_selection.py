"""
Tuning objective parsing, release-level CV folds, constraints, and composite scores.

Used by ``train_vinyliq._run_tuning``. Metrics ``mdape`` / ``wape`` are fractions of 1
(e.g. 0.0245 is 2.45% median APE when printed as a percent).
"""
from __future__ import annotations

import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

CvAgg = Literal["mean", "max"]
ViolationFallback = Literal["best_violation", "penalize", "abort"]
CvStratify = Literal["anchor_quartile"] | None


@dataclass(frozen=True)
class TuningConstraints:
    enabled: bool
    mdape_max: float
    wape_max: float
    violation_fallback: ViolationFallback
    lambda_mdape: float
    lambda_wape: float


@dataclass(frozen=True)
class SelectionObjective:
    """How to score a trial from aggregated CV val metrics."""

    composite: bool
    single_field: str | None  # "mdape" | "wape" | "mae" when not composite
    mlflow_name: str
    w_mdape: float
    w_wape: float
    w_mae: float
    mae_ref_usd: float
    #: When True, ``mdape`` passed into ``base_selection_score`` is format-weighted val MdAPE.
    use_weighted_format_mdape: bool = False


def parse_tuning_constraints(tuning: dict[str, Any] | None) -> TuningConstraints:
    raw = (tuning or {}).get("constraints")
    if not isinstance(raw, dict):
        raw = {}
    enabled = bool(raw.get("enabled", False))
    mdape_max = float(raw.get("mdape_max", 0.05))
    wape_max = float(raw.get("wape_max", 0.20))
    fb = str(raw.get("violation_fallback", "best_violation")).strip().lower()
    if fb not in ("best_violation", "penalize", "abort"):
        fb = "best_violation"
    pen = (tuning or {}).get("constraint_penalty") or {}
    if not isinstance(pen, dict):
        pen = {}
    lam_m = float(pen.get("lambda_mdape", 1000.0))
    lam_w = float(pen.get("lambda_wape", 100.0))
    return TuningConstraints(
        enabled=enabled,
        mdape_max=mdape_max,
        wape_max=wape_max,
        violation_fallback=fb,  # type: ignore[arg-type]
        lambda_mdape=lam_m,
        lambda_wape=lam_w,
    )


def _resolve_single_selection_metric(
    tuning: dict[str, Any] | None,
) -> tuple[str, str]:
    raw = str((tuning or {}).get("selection_metric", "median_ape")).strip().lower()
    aliases: dict[str, tuple[str, str]] = {
        "median_ape": ("mdape", "val_median_ape_dollars"),
        "mdape": ("mdape", "val_median_ape_dollars"),
        "val_median_ape_dollars": ("mdape", "val_median_ape_dollars"),
        "wape": ("wape", "val_wape_dollars"),
        "val_wape_dollars": ("wape", "val_wape_dollars"),
        "mae_dollars": ("mae", "val_mae_dollars_approx"),
        "mae": ("mae", "val_mae_dollars_approx"),
        "val_mae_dollars_approx": ("mae", "val_mae_dollars_approx"),
    }
    return aliases.get(raw, ("mdape", "val_median_ape_dollars"))


def parse_selection_format_weights(
    tuning: dict[str, Any] | None,
) -> dict[str, float]:
    """Weights for ``weighted_format_*`` metrics; keys match format slice buckets."""
    raw = (tuning or {}).get("selection_format_weights")
    if not isinstance(raw, dict):
        raw = {}
    default_w = float(raw.get("default", 1.0))
    order = ("box_multi", "seven", "ten", "twelve", "lp", "cd", "other")
    return {name: float(raw.get(name, default_w)) for name in order}


def parse_selection_objective(tuning: dict[str, Any] | None) -> SelectionObjective:
    raw = str((tuning or {}).get("selection_metric", "median_ape")).strip().lower()
    if raw == "composite":
        sc = tuning.get("selection_composite") or {}
        if not isinstance(sc, dict):
            sc = {}
        return SelectionObjective(
            composite=True,
            single_field=None,
            mlflow_name="val_selection_composite",
            w_mdape=float(sc.get("w_mdape", 1.0)),
            w_wape=float(sc.get("w_wape", 0.5)),
            w_mae=float(sc.get("w_mae", 0.3)),
            mae_ref_usd=max(float(sc.get("mae_ref_usd", 50.0)), 1e-6),
            use_weighted_format_mdape=False,
        )
    if raw == "weighted_format_composite":
        sc = tuning.get("selection_composite") or {}
        if not isinstance(sc, dict):
            sc = {}
        return SelectionObjective(
            composite=True,
            single_field=None,
            mlflow_name="val_selection_weighted_format_composite",
            w_mdape=float(sc.get("w_mdape", 1.0)),
            w_wape=float(sc.get("w_wape", 0.5)),
            w_mae=float(sc.get("w_mae", 0.3)),
            mae_ref_usd=max(float(sc.get("mae_ref_usd", 50.0)), 1e-6),
            use_weighted_format_mdape=True,
        )
    if raw == "weighted_format_mdape":
        return SelectionObjective(
            composite=False,
            single_field="mdape",
            mlflow_name="val_weighted_format_median_ape",
            w_mdape=1.0,
            w_wape=0.0,
            w_mae=0.0,
            mae_ref_usd=50.0,
            use_weighted_format_mdape=True,
        )
    field, name = _resolve_single_selection_metric(tuning)
    return SelectionObjective(
        composite=False,
        single_field=field,
        mlflow_name=name,
        w_mdape=1.0,
        w_wape=0.0,
        w_mae=0.0,
        mae_ref_usd=50.0,
        use_weighted_format_mdape=False,
    )


def base_selection_score(
    obj: SelectionObjective,
    mdape: float,
    mae: float,
    wape: float,
) -> float:
    if obj.composite:
        return float(
            obj.w_mdape * mdape
            + obj.w_wape * wape
            + obj.w_mae * (mae / obj.mae_ref_usd)
        )
    assert obj.single_field is not None
    return float({"mae": mae, "wape": wape, "mdape": mdape}[obj.single_field])


def aggregate_cv(values: list[float], agg: CvAgg) -> float:
    if not values:
        return float("nan")
    a = np.asarray(values, dtype=np.float64)
    if agg == "max":
        return float(np.nanmax(a))
    return float(np.nanmean(a))


def is_feasible(mdape: float, wape: float, cons: TuningConstraints) -> bool:
    if not cons.enabled:
        return True
    if not (math.isfinite(mdape) and math.isfinite(wape)):
        return False
    return (mdape < cons.mdape_max) and (wape < cons.wape_max)


def violation_slack(mdape: float, wape: float, cons: TuningConstraints) -> float:
    """ nonnegative; 0 when inside caps (strict < caps for feasibility). """
    if not cons.enabled:
        return 0.0
    sm = max(mdape / max(cons.mdape_max, 1e-12) - 1.0, 0.0)
    sw = max(wape / max(cons.wape_max, 1e-12) - 1.0, 0.0)
    return float(sm + sw)


def penalty_augmented_score(
    base: float,
    mdape: float,
    wape: float,
    cons: TuningConstraints,
) -> float:
    d_m = max(mdape - cons.mdape_max, 0.0)
    d_w = max(wape - cons.wape_max, 0.0)
    return float(base + cons.lambda_mdape * d_m * d_m + cons.lambda_wape * d_w * d_w)


def _median_anchor_per_release(
    train_r: set[str],
    rids: list[str],
    median_all: np.ndarray,
) -> dict[str, float]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for i, rid in enumerate(rids):
        if rid not in train_r:
            continue
        buckets[rid].append(float(median_all[i]))
    return {rid: float(np.median(v)) for rid, v in buckets.items()}


def build_cv_fold_val_release_sets(
    train_r: set[str],
    rids: list[str],
    median_all: np.ndarray,
    k: int,
    seed: int,
    *,
    stratify: CvStratify,
) -> list[set[str]]:
    """
    Partition ``train_r`` into ``k`` disjoint val release sets (one per fold).

    Training rows for fold ``j`` are releases in ``train_r`` minus fold ``j`` val set.
    """
    rels = sorted(train_r)
    n_rel = len(rels)
    if n_rel == 0:
        return []
    eff_k = max(1, min(int(k), n_rel))
    if eff_k == 1:
        return [set(rels)]

    if stratify == "anchor_quartile" and eff_k >= 2:
        med_map = _median_anchor_per_release(train_r, rids, median_all)
        vals = np.array([med_map.get(rid, 0.0) for rid in rels], dtype=np.float64)
        try:
            from sklearn.model_selection import StratifiedKFold

            qs = np.quantile(vals, [0.25, 0.5, 0.75])
            y = np.digitize(vals, qs, right=False).astype(int)
            y = np.clip(y, 0, 3)
            if len(np.unique(y)) > 1:
                skf = StratifiedKFold(
                    n_splits=eff_k, shuffle=True, random_state=seed
                )
                fold_val_sets = [set() for _ in range(eff_k)]
                for fold_idx, (_, test_idx) in enumerate(skf.split(np.zeros(n_rel), y)):
                    for ii in test_idx:
                        fold_val_sets[fold_idx].add(rels[int(ii)])
                return fold_val_sets
        except (ValueError, ImportError):
            pass

    rng = np.random.default_rng(int(seed))
    order = list(rels)
    rng.shuffle(order)
    fold_val_sets = [set() for _ in range(eff_k)]
    for i, rid in enumerate(order):
        fold_val_sets[i % eff_k].add(rid)
    return fold_val_sets


def row_masks_from_release_sets(
    rids: list[str],
    outer_train: set[str],
    val_releases: set[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Row booleans: tune-train (outer train minus val releases), val rows."""
    tt = np.array(
        [(rid in outer_train) and (rid not in val_releases) for rid in rids],
        dtype=bool,
    )
    va = np.array([(rid in val_releases) for rid in rids], dtype=bool)
    return tt, va


def _split_diag_enabled() -> bool:
    import os

    return os.environ.get("VINYLIQ_SPLIT_DIAGNOSTICS", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def log_split_anchor_format_diagnostics(
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    median_all: np.ndarray,
    X_all: np.ndarray,
    cols: list[str],
) -> None:
    """When ``VINYLIQ_SPLIT_DIAGNOSTICS=1``: log ``log1p(anchor)`` and ``format_family`` marginals."""
    if not _split_diag_enabled():
        return
    def _summ(split: str, m: np.ndarray) -> None:
        if not np.any(m):
            print(f"[VINYLIQ_SPLIT_DIAGNOSTICS] {split}: (empty)", file=sys.stderr)
            return
        la = np.log1p(np.maximum(median_all[m], 0.0))
        print(
            f"[VINYLIQ_SPLIT_DIAGNOSTICS] {split}: rows={int(m.sum())} "
            f"log1p(anchor) median={float(np.median(la)):.4f} "
            f"p10..p90={float(np.percentile(la,10)):.4f}..{float(np.percentile(la,90)):.4f}",
            file=sys.stderr,
        )
        if "format_family" in cols:
            j = cols.index("format_family")
            fam = X_all[m, j].astype(np.int64)
            u, c = np.unique(fam, return_counts=True)
            top = sorted(zip(c, u), reverse=True)[:8]
            parts = [f"fmt{int(uu)}={int(cc)}" for cc, uu in top]
            print(f"  format_family counts: " + " ".join(parts), file=sys.stderr)

    _summ("train_outer", train_mask)
    _summ("test_outer", test_mask)


@dataclass
class TrialRecord:
    family: str
    params: dict[str, Any]
    val_mdape: float
    val_mae: float
    val_wape: float
    base_score: float
    feasible: bool
    slack: float
    pen_score: float
    best_iteration: int | None
    cv_folds_used: int


def pick_champion_trial(
    trials: list[TrialRecord],
    cons: TuningConstraints,
) -> tuple[TrialRecord | None, str]:
    """
    Returns ``(best, reason)``. ``reason`` is ``feasible``, ``no_constraints``,
    ``best_violation``, ``penalize``, or ``abort``.
    """
    if not trials:
        return None, "abort"
    feas = [t for t in trials if t.feasible]
    if feas:
        best = min(feas, key=lambda t: t.base_score)
        return best, "feasible" if cons.enabled else "no_constraints"
    if not cons.enabled:
        best = min(trials, key=lambda t: t.base_score)
        return best, "no_constraints"
    fb = cons.violation_fallback
    if fb == "abort":
        return None, "abort"
    if fb == "penalize":
        best = min(trials, key=lambda t: t.pen_score)
        return best, "penalize"
    # best_violation: min slack, tie-break base_score
    best = min(trials, key=lambda t: (t.slack, t.base_score))
    return best, "best_violation"


def build_trial_record(
    *,
    family: str,
    params: dict[str, Any],
    mdapes: list[float],
    maes: list[float],
    wapes: list[float],
    best_iters: list[int | None],
    cv_agg: CvAgg,
    cons: TuningConstraints,
    sel_obj: SelectionObjective,
    cv_folds_used: int,
    selection_mdapes: list[float] | None = None,
) -> TrialRecord | None:
    md = aggregate_cv(mdapes, cv_agg)
    ma = aggregate_cv(maes, cv_agg)
    wa = aggregate_cv(wapes, cv_agg)
    md_sel = aggregate_cv(selection_mdapes, cv_agg) if selection_mdapes else md
    if not all(map(math.isfinite, (md, ma, wa))):
        return None
    if not math.isfinite(md_sel):
        return None
    base = base_selection_score(sel_obj, md_sel, ma, wa)
    feas = is_feasible(md, wa, cons)
    slack = violation_slack(md, wa, cons)
    pen = penalty_augmented_score(base, md, wa, cons)
    bis = [b for b in best_iters if isinstance(b, int) and b >= 0]
    bi: int | None = int(round(float(np.median(bis)))) if bis else None
    return TrialRecord(
        family=family,
        params=params,
        val_mdape=md,
        val_mae=ma,
        val_wape=wa,
        base_score=base,
        feasible=feas,
        slack=slack,
        pen_score=pen,
        best_iteration=bi,
        cv_folds_used=cv_folds_used,
    )

