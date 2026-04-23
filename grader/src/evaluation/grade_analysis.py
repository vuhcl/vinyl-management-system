"""
grader/src/evaluation/grade_analysis.py

Per-split reports: true-label support (counts) and, for chosen predicted
grades (e.g. Excellent), how often each true grade appears when the model
or rule engine predicts that class.

Also exposes **rule-owned slice** helpers: given a rule-owned true grade G
(e.g. ``Poor`` / ``Generic``), report how model-only and rule-adjusted
predictions distribute and what slice recall is attained. Used by the
pipeline to append a ``RULE-OWNED SLICE`` banner to
``grade_analysis_{split}.txt``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from grader.src.evaluation.metrics import remap_true_and_encode_predictions


# ---------------------------------------------------------------------------
# Rule-owned grade resolution (shared by report + metrics)
# ---------------------------------------------------------------------------
_TARGETS = ("sleeve", "media")


def resolve_rule_owned_grades(guidelines: dict) -> dict[str, list[str]]:
    """
    Return rule-owned grades per target from the loaded guidelines dict.

    A grade is rule-owned when ``grade_owners[grade] == "rule_engine"``.
    Per-target filtering uses the grade's ``applies_to`` list
    (e.g. ``Generic`` only applies to sleeve).

    Args:
        guidelines: parsed grading_guidelines.yaml

    Returns:
        ``{"sleeve": [...], "media": [...]}`` — lists are stable in the
        order grades appear in ``guidelines["grades"]``.
    """
    owners = guidelines.get("grade_owners", {}) or {}
    grade_defs = guidelines.get("grades", {}) or {}

    out: dict[str, list[str]] = {t: [] for t in _TARGETS}
    for grade, gdef in grade_defs.items():
        if owners.get(grade) != "rule_engine":
            continue
        applies_to = gdef.get("applies_to", list(_TARGETS)) or []
        for t in _TARGETS:
            if t in applies_to:
                out[t].append(str(grade))
    return out


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

    Default ``focus_classes`` is an empty tuple — no predicted-class spotlight.
    Callers that want pred-conditioned sections (e.g. Excellent workflows or
    rule-owned grade audits) must pass ``focus_classes`` explicitly.
    """
    if focus_classes is None:
        focus_classes = ()

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


# ---------------------------------------------------------------------------
# True-label-conditioned slice helpers (rule-owned grades)
# ---------------------------------------------------------------------------
def true_label_breakdown_for_grade(
    y_true: np.ndarray,
    y_pred_idx: np.ndarray,
    class_names: np.ndarray,
    true_class_name: str,
) -> tuple[int, dict[str, int]]:
    """
    Among rows where ``y_true == true_class_name``, count predicted labels.

    Inverts :func:`prediction_breakdown_for_class`: instead of
    "given we predicted X, what was the true label?", returns
    "given the true label is G, what did we predict?".

    Returns:
        ``(n_rows_with_that_true_label, {pred_grade_str: count, ...})``
    """
    name_to_idx = {str(c): i for i, c in enumerate(class_names)}
    if true_class_name not in name_to_idx:
        return 0, {}
    tidx = name_to_idx[true_class_name]
    mask = y_true == tidx
    n = int(mask.sum())
    if n == 0:
        return 0, {}
    yp = y_pred_idx[mask]
    breakdown: dict[str, int] = {}
    for i, name in enumerate(class_names):
        c = int((yp == i).sum())
        if c > 0:
            breakdown[str(name)] = c
    return n, breakdown


def slice_recall_for_grade(
    y_true: np.ndarray,
    y_pred_idx: np.ndarray,
    class_names: np.ndarray,
    true_class_name: str,
) -> Optional[float]:
    """
    Slice recall for a specific true grade: fraction of rows with
    ``y_true == G`` where the predicted label equals ``G``.

    Returns ``None`` if the grade has zero support in the split.
    """
    name_to_idx = {str(c): i for i, c in enumerate(class_names)}
    if true_class_name not in name_to_idx:
        return None
    tidx = name_to_idx[true_class_name]
    mask = y_true == tidx
    n = int(mask.sum())
    if n == 0:
        return None
    return round(float((y_pred_idx[mask] == tidx).sum()) / n, 4)


def slice_precision_for_grade(
    y_true: np.ndarray,
    y_pred_idx: np.ndarray,
    class_names: np.ndarray,
    class_name: str,
) -> Optional[float]:
    """
    Slice precision for a predicted grade: fraction of rows with
    ``y_pred == G`` where the true label also equals ``G``.

    Returns ``None`` if the grade is never predicted in the split.
    """
    name_to_idx = {str(c): i for i, c in enumerate(class_names)}
    if class_name not in name_to_idx:
        return None
    pidx = name_to_idx[class_name]
    mask = y_pred_idx == pidx
    n = int(mask.sum())
    if n == 0:
        return None
    return round(float((y_true[mask] == pidx).sum()) / n, 4)


def build_rule_owned_slice_report(
    y_true: np.ndarray,
    y_pred_model: list[str],
    y_pred_rule: list[str],
    class_names: np.ndarray,
    target: str,
    split: str,
    rule_owned_grades: list[str],
) -> str:
    """
    Render a true-label-conditioned slice report for rule-owned grades.

    For each grade G in ``rule_owned_grades``:

    * row mask where ``y_true == G``;
    * predicted histograms (before / after rule engine);
    * **slice recall** before and after.

    Also reports **slice precision** for each rule-owned grade as a
    predicted class, which is the hard-override-precision analogue
    (fraction of rows predicted G that truly are G).

    Empty sections are rendered with a single "(no support)" line so
    the banner remains grep-able when a split lacks that grade entirely.
    """
    if len(y_true) != len(y_pred_model) or len(y_true) != len(y_pred_rule):
        raise ValueError("y_true and prediction lists must align in length")

    y_t2, combined_list, encs = remap_true_and_encode_predictions(
        y_true, class_names, y_pred_model, y_pred_rule
    )
    y_model, y_rule = encs[0], encs[1]
    combined = np.array(combined_list)

    lines: list[str] = [
        f"RULE-OWNED SLICE — target={target.upper()} — split={split}",
        f"Rule-owned grades (per guidelines): {list(rule_owned_grades) or '—'}",
    ]

    for g in rule_owned_grades:
        lines.extend(["", f"--- True label = {g!r} ---"])
        if g not in combined_list:
            lines.append("  (grade not in merged class list — no support)")
            continue

        n, bd_model = true_label_breakdown_for_grade(
            y_t2, y_model, combined, g
        )
        _, bd_rule = true_label_breakdown_for_grade(y_t2, y_rule, combined, g)
        rec_model = slice_recall_for_grade(y_t2, y_model, combined, g)
        rec_rule = slice_recall_for_grade(y_t2, y_rule, combined, g)
        prec_model = slice_precision_for_grade(y_t2, y_model, combined, g)
        prec_rule = slice_precision_for_grade(y_t2, y_rule, combined, g)

        lines.append(f"  Support (true={g!r}): {n} row(s)")
        if n == 0:
            lines.append("  (no support)")
            continue

        lines.append(
            f"  Slice recall (model only → rule-adjusted): "
            f"{_fmt_opt(rec_model)} → {_fmt_opt(rec_rule)}"
        )
        lines.append(
            f"  Slice precision for predicted {g!r} "
            f"(model only → rule-adjusted): "
            f"{_fmt_opt(prec_model)} → {_fmt_opt(prec_rule)}"
        )

        lines.append("  Predicted histogram — model only:")
        _render_hist(lines, bd_model)
        lines.append("  Predicted histogram — after rule engine:")
        _render_hist(lines, bd_rule)

    lines.append("")
    return "\n".join(lines)


def _fmt_opt(value: Optional[float]) -> str:
    return "—" if value is None else f"{value:.4f}"


def _render_hist(lines: list[str], breakdown: dict[str, int]) -> None:
    if not breakdown:
        lines.append("    (no predictions)")
        return
    lines.append("    Predicted | count")
    for g, c in sorted(breakdown.items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"      {g:<20} {c:>6}")
