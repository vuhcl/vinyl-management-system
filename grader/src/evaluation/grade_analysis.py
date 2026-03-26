"""
grader/src/evaluation/grade_analysis.py

Per-split reports: true-label support (counts) and, for chosen predicted
grades (e.g. Excellent), how often each true grade appears when the model
or rule engine predicts that class.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from grader.src.evaluation.metrics import remap_true_and_encode_predictions


def _encode_labels(
    labels: list[str], class_names: np.ndarray
) -> tuple[np.ndarray, dict[str, int]]:
    """Map grade strings to indices; raises if a label is not in class_names."""
    classes_list = [str(c) for c in class_names]
    name_to_idx = {c: i for i, c in enumerate(classes_list)}
    out = np.empty(len(labels), dtype=int)
    for i, lab in enumerate(labels):
        key = str(lab)
        if key not in name_to_idx:
            raise ValueError(
                f"Predicted label {lab!r} not in encoder classes"
            )
        out[i] = name_to_idx[key]
    return out, name_to_idx


def true_label_support(
    y_true: np.ndarray, class_names: np.ndarray
) -> dict[str, int]:
    """Count of each true grade index (includes zeros)."""
    return {
        str(name): int((y_true == i).sum())
        for i, name in enumerate(class_names)
    }


def prediction_breakdown_for_class(
    y_true: np.ndarray,
    y_pred_idx: np.ndarray,
    class_names: np.ndarray,
    pred_class_name: str,
) -> tuple[int, dict[str, int]]:
    """
    Among rows where predicted class == pred_class_name, count true labels.

    Returns:
        (n_rows_with_that_prediction, {true_grade_str: count, ...})
    """
    name_to_idx = {str(c): i for i, c in enumerate(class_names)}
    if pred_class_name not in name_to_idx:
        return 0, {}
    pidx = name_to_idx[pred_class_name]
    mask = y_pred_idx == pidx
    n = int(mask.sum())
    if n == 0:
        return 0, {}
    yt = y_true[mask]
    breakdown: dict[str, int] = {}
    for i, name in enumerate(class_names):
        c = int((yt == i).sum())
        if c > 0:
            breakdown[str(name)] = c
    return n, breakdown


def build_grade_analysis_report(
    y_true: np.ndarray,
    y_pred_model: list[str],
    y_pred_rule: list[str],
    class_names: np.ndarray,
    target: str,
    split: str,
    focus_classes: Optional[tuple[str, ...]] = None,
    after_for_scoring: Optional[list[str]] = None,
) -> str:
    """
    Human-readable report: support on y_true, then per focus class
    breakdowns for model-only vs rule-adjusted predictions.

    If ``after_for_scoring`` is set (e.g. Excellent→model substitution), adds
    a third breakdown for the labels actually used in rule-adjusted metrics.
    """
    if focus_classes is None:
        focus_classes = ("Excellent",)

    if len(y_true) != len(y_pred_model) or len(y_true) != len(y_pred_rule):
        raise ValueError("y_true and prediction lists must align in length")
    if after_for_scoring is not None and len(after_for_scoring) != len(y_true):
        raise ValueError("after_for_scoring must match y_true length")

    if after_for_scoring is not None:
        y_t2, combined_list, encs = remap_true_and_encode_predictions(
            y_true,
            class_names,
            y_pred_model,
            y_pred_rule,
            after_for_scoring,
        )
        y_model, y_rule, y_scoring = encs[0], encs[1], encs[2]
    else:
        y_t2, combined_list, encs = remap_true_and_encode_predictions(
            y_true, class_names, y_pred_model, y_pred_rule
        )
        y_model, y_rule = encs[0], encs[1]
        y_scoring = None

    combined = np.array(combined_list)

    support = true_label_support(y_t2, combined)
    n = len(y_t2)

    lines: list[str] = [
        f"GRADE ANALYSIS — target={target.upper()} — split={split}",
        f"N = {n}",
    ]
    if after_for_scoring is not None:
        lines.extend(
            [
                "",
                "NOTE: Rule-adjusted macro-F1/accuracy use model-only labels on rows",
                "where the rule-adjusted prediction is Excellent (config:",
                "evaluation.excellent_eval_use_model_prediction).",
            ]
        )
    lines.extend(
        [
            "",
            "True label support (ground truth counts):",
        ]
    )
    for grade, cnt in sorted(support.items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"  {grade:<22} {cnt:>6}")

    for fc in focus_classes:
        lines.extend(["", f"--- When PREDICTED class = {fc!r} ---"])
        if fc not in combined_list:
            lines.append(f"  (skipped — {fc!r} not in merged class list)")
            continue

        nm, bd_model = prediction_breakdown_for_class(
            y_t2, y_model, combined, fc
        )
        nr, bd_rule = prediction_breakdown_for_class(
            y_t2, y_rule, combined, fc
        )

        lines.append(f"  Model only:       {nm} row(s) with this prediction")
        if nm == 0:
            lines.append("    (no breakdown)")
        else:
            lines.append("    True label | count")
            for g, c in sorted(bd_model.items(), key=lambda x: (-x[1], x[0])):
                lines.append(f"      {g:<20} {c:>6}")

        lines.append(f"  After rule engine (raw): {nr} row(s) with this prediction")
        if nr == 0:
            lines.append("    (no breakdown)")
        else:
            lines.append("    True label | count")
            for g, c in sorted(bd_rule.items(), key=lambda x: (-x[1], x[0])):
                lines.append(f"      {g:<20} {c:>6}")

        if y_scoring is not None:
            ns, bd_sc = prediction_breakdown_for_class(
                y_t2, y_scoring, combined, fc
            )
            lines.append(
                f"  Effective for rule-adjusted metrics: {ns} row(s) with pred {fc!r}"
            )
            if ns == 0:
                lines.append("    (no breakdown)")
            else:
                lines.append("    True label | count")
                for g, c in sorted(
                    bd_sc.items(), key=lambda x: (-x[1], x[0])
                ):
                    lines.append(f"      {g:<20} {c:>6}")

    lines.append("")
    return "\n".join(lines)
