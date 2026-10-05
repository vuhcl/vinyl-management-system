"""Champion/challenger gate for the condition classifier.

Pure functions over prediction rows, so the decision logic is unit-testable
without loading a model. A prediction row is a dict with:
``item_id, true_sleeve, true_media, pred_sleeve, pred_media``.
Champion and challenger rows must cover the same ``item_id`` set.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import f1_score

TARGETS = ("sleeve", "media")
# Ordinal scale for within-one-grade agreement. Labels outside it (e.g. "Generic")
# are excluded from that metric only; the excluded count is reported.
GRADE_ORDER = {"Poor": 0, "Good": 1, "Very Good": 2, "Very Good Plus": 3, "Near Mint": 4, "Mint": 5}


def accuracy(rows: list[dict], target: str) -> float:
    return sum(r[f"true_{target}"] == r[f"pred_{target}"] for r in rows) / len(rows)


def within_one_grade(rows: list[dict], target: str) -> tuple[float, int]:
    """Share of rows with |true - pred| <= 1 on the ordinal scale, and rows scored."""
    ok = n = 0
    for r in rows:
        y, p = r[f"true_{target}"], r[f"pred_{target}"]
        if y in GRADE_ORDER and p in GRADE_ORDER:
            n += 1
            ok += abs(GRADE_ORDER[y] - GRADE_ORDER[p]) <= 1
    return (ok / n if n else float("nan")), n


def _align(champ: list[dict], chal: list[dict]) -> tuple[list[dict], list[dict]]:
    c_by = {str(r["item_id"]): r for r in champ}
    h_by = {str(r["item_id"]): r for r in chal}
    if set(c_by) != set(h_by):
        only_c, only_h = set(c_by) - set(h_by), set(h_by) - set(c_by)
        raise ValueError(
            f"champion/challenger item_id mismatch: {len(only_c)} only in champion, "
            f"{len(only_h)} only in challenger"
        )
    ids = sorted(c_by)
    return [c_by[i] for i in ids], [h_by[i] for i in ids]


def macro_f1(rows: list[dict], target: str) -> float:
    y = [r[f"true_{target}"] for r in rows]
    p = [r[f"pred_{target}"] for r in rows]
    return float(f1_score(y, p, average="macro", zero_division=0))


def _fast_macro_f1(y: np.ndarray, p: np.ndarray, k: int) -> float:
    """Macro-F1 over classes present in y or p, from integer-coded labels."""
    tp = np.bincount(y[y == p], minlength=k).astype(float)
    ny = np.bincount(y, minlength=k).astype(float)
    npred = np.bincount(p, minlength=k).astype(float)
    denom = ny + npred
    present = denom > 0
    f1 = np.where(denom > 0, 2 * tp / np.where(denom == 0, 1, denom), 0.0)
    return float(f1[present].mean())


def _mean_f1_from_arrays(y, p, idx, k) -> float:
    return float(np.mean([_fast_macro_f1(y[t][idx], p[t][idx], k[t]) for t in TARGETS]))


def paired_bootstrap_delta(
    champ: list[dict], chal: list[dict], n: int = 1000, seed: int = 42
) -> tuple[float, float]:
    """95% CI of (challenger - champion) mean macro-F1, resampling items jointly."""
    champ, chal = _align(champ, chal)
    y, pc, ph, k = {}, {}, {}, {}
    for t in TARGETS:
        labels = sorted({r[f"true_{t}"] for r in champ} | {r[f"pred_{t}"] for r in champ + chal})
        code = {c: i for i, c in enumerate(labels)}
        k[t] = len(labels)
        y[t] = np.array([code[r[f"true_{t}"]] for r in champ])
        pc[t] = np.array([code[r[f"pred_{t}"]] for r in champ])
        ph[t] = np.array([code[r[f"pred_{t}"]] for r in chal])
    rng = np.random.default_rng(seed)
    size = len(champ)
    deltas = np.empty(n)
    for b in range(n):
        idx = rng.integers(0, size, size)
        deltas[b] = _mean_f1_from_arrays(y, ph, idx, k) - _mean_f1_from_arrays(y, pc, idx, k)
    return float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def evaluate_gate(
    champ: list[dict], chal: list[dict], cfg: dict[str, Any]
) -> dict[str, Any]:
    """Return a decision report. ``promote`` is True only if every rule passes."""
    champ, chal = _align(champ, chal)
    per_target = {}
    for t in TARGETS:
        c, h = macro_f1(champ, t), macro_f1(chal, t)
        cw, cn = within_one_grade(champ, t)
        hw, hn = within_one_grade(chal, t)
        per_target[t] = {
            "champion": round(c, 4), "challenger": round(h, 4), "delta": round(h - c, 4),
            "accuracy": {"champion": round(accuracy(champ, t), 4), "challenger": round(accuracy(chal, t), 4)},
            "within_one_grade": {
                "champion": round(cw, 4), "challenger": round(hw, 4),
                "rows_scored": {"champion": cn, "challenger": hn},
            },
        }
    mean_delta = float(np.mean([per_target[t]["delta"] for t in TARGETS]))
    lo, hi = paired_bootstrap_delta(
        champ, chal, int(cfg.get("bootstrap_n", 1000)), int(cfg.get("bootstrap_seed", 42))
    )

    reasons = []
    if mean_delta < float(cfg["min_mean_macro_f1_gain"]):
        reasons.append(
            f"mean macro-F1 gain {mean_delta:.4f} < required {cfg['min_mean_macro_f1_gain']}"
        )
    for t in TARGETS:
        if per_target[t]["delta"] < -float(cfg["max_target_regression"]):
            reasons.append(
                f"{t} regressed {per_target[t]['delta']:.4f} beyond tolerance -{cfg['max_target_regression']}"
            )
    if cfg.get("require_ci_lower_gt_zero") and lo <= 0:
        reasons.append(f"bootstrap CI lower bound {lo:.4f} <= 0")

    return {
        "n_items": len(champ),
        "per_target": per_target,
        "mean_macro_f1_delta": round(mean_delta, 4),
        "bootstrap_ci95": [round(lo, 4), round(hi, 4)],
        "promote": not reasons,
        "reasons": reasons or ["all gate rules passed"],
    }
