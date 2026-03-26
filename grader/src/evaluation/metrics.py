"""
grader/src/evaluation/metrics.py

Stateless evaluation metric utilities for the vinyl condition grader.
Pure functions — no MLflow calls, no file I/O, no class state.
Used by baseline.py, transformer.py, and pipeline.py.

All functions accept numpy arrays as input and return dicts or
formatted strings — model-agnostic by design.

Usage (from any model module):
    from grader.src.evaluation.metrics import compute_metrics, compare_models

    metrics = compute_metrics(
        y_true=y_test,
        y_pred=y_pred,
        y_proba=y_proba,
        class_names=encoder.classes_,
        target="sleeve",
    )
"""

import logging
from typing import Optional

import mlflow
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core metric computation
# ---------------------------------------------------------------------------
def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    class_names: np.ndarray,
    target: str,
    split: str = "test",
) -> dict:
    """
    Compute all evaluation metrics for a single target on a single split.

    Args:
        y_true:       ground truth integer label array
        y_pred:       predicted integer label array
        y_proba:      predicted probability matrix (n_samples x n_classes)
        class_names:  ordered array of grade strings matching label encoder
        target:       "sleeve" or "media" — for labeling only
        split:        "train", "val", or "test" — for labeling only

    Returns:
        Dict with the following keys:
            target        str
            split         str
            macro_f1      float
            accuracy      float
            ece           float
            per_class     dict mapping class name → {f1, precision, recall, support}
            class_names   list of str
    """
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    accuracy = float(accuracy_score(y_true, y_pred))
    ece = compute_ece(y_true, y_proba)

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=list(class_names),
        output_dict=True,
        zero_division=0,
    )

    # Extract per-class metrics — filter out aggregate keys
    per_class = {
        class_name: {
            "f1": round(vals["f1-score"], 4),
            "precision": round(vals["precision"], 4),
            "recall": round(vals["recall"], 4),
            "support": int(vals["support"]),
        }
        for class_name, vals in report.items()
        if isinstance(vals, dict) and class_name in list(class_names)
    }

    return {
        "target": target,
        "split": split,
        "macro_f1": round(macro_f1, 4),
        "accuracy": round(accuracy, 4),
        "ece": round(ece, 4),
        "per_class": per_class,
        "class_names": list(class_names),
    }


def remap_true_and_encode_predictions(
    y_true: np.ndarray,
    class_names: np.ndarray,
    *pred_label_lists: list[str],
) -> tuple[np.ndarray, list[str], list[np.ndarray]]:
    """
    Align integer ``y_true`` (indices into ``class_names``) with prediction
    strings that may include labels missing from the saved encoder (e.g. rules
    assign ``Excellent`` while an older encoder was fit on train-only labels).

    Returns remapped ``y_true``, a sorted merged class list, and encoded
    prediction arrays in the same order as ``pred_label_lists``.
    """
    base = [str(c) for c in class_names]
    pred_tokens: set[str] = set()
    for plist in pred_label_lists:
        pred_tokens.update(str(p) for p in plist)
    extras = pred_tokens - set(base)
    if not extras:
        name_to_idx = {c: i for i, c in enumerate(base)}
        encoded = [
            np.array([name_to_idx[str(p)] for p in plist], dtype=int)
            for plist in pred_label_lists
        ]
        return y_true.copy(), base, encoded

    combined = sorted(set(base) | pred_tokens)
    idx_to_name = {i: base[i] for i in range(len(base))}
    try:
        y_remapped = np.array(
            [combined.index(idx_to_name[int(i)]) for i in y_true],
            dtype=int,
        )
    except KeyError as e:
        raise ValueError(
            f"y_true index out of range for encoder with {len(base)} classes"
        ) from e

    name_to_idx = {c: j for j, c in enumerate(combined)}
    encoded = [
        np.array([name_to_idx[str(p)] for p in plist], dtype=int)
        for plist in pred_label_lists
    ]
    logger.debug(
        "Expanded metrics label space — added %s (encoder had %d classes)",
        sorted(extras),
        len(base),
    )
    return y_remapped, combined, encoded


def substitute_model_when_pred_excellent(
    y_pred_after: list[str],
    y_pred_before: list[str],
    excellent_label: str = "Excellent",
) -> list[str]:
    """
    For each row, if the rule-adjusted prediction equals ``excellent_label``,
    use the model-only label instead; otherwise keep the adjusted label.

    Use when eval data lacks Excellent so rule-assigned Excellent should not
    dominate error metrics — score those rows against the model prediction.
    """
    if len(y_pred_after) != len(y_pred_before):
        raise ValueError("y_pred_after and y_pred_before must have the same length")
    ex = str(excellent_label)
    return [
        str(before) if str(after) == ex else str(after)
        for before, after in zip(y_pred_before, y_pred_after)
    ]


def compute_metrics_from_label_strings(
    y_true: np.ndarray,
    y_pred_labels: list[str],
    class_names: np.ndarray,
    target: str,
    split: str = "test",
) -> dict:
    """
    Macro-F1 and accuracy from integer y_true and string predictions.

    Used after the rule engine adjusts predicted grade strings — there are
    no calibrated probabilities, so ECE is omitted (None).

    Args:
        y_true:         ground truth integer label array (same encoding as training)
        y_pred_labels:  predicted grade strings, one per row, aligned with y_true
        class_names:    ordered label set (e.g. encoder.classes_)
        target:         "sleeve" or "media"
        split:          split name for reporting only

    Returns:
        Same structure as compute_metrics except ``ece`` is None.
    """
    if len(y_pred_labels) != len(y_true):
        raise ValueError(
            f"y_true length {len(y_true)} != len(y_pred_labels) {len(y_pred_labels)}"
        )

    y_t2, classes_list, enc_list = remap_true_and_encode_predictions(
        y_true, class_names, y_pred_labels
    )
    y_pred = enc_list[0]
    n_cls = len(classes_list)

    macro_f1 = float(f1_score(y_t2, y_pred, average="macro"))
    accuracy = float(accuracy_score(y_t2, y_pred))

    report = classification_report(
        y_t2,
        y_pred,
        labels=list(range(n_cls)),
        target_names=classes_list,
        output_dict=True,
        zero_division=0,
    )

    per_class = {
        class_name: {
            "f1": round(vals["f1-score"], 4),
            "precision": round(vals["precision"], 4),
            "recall": round(vals["recall"], 4),
            "support": int(vals["support"]),
        }
        for class_name, vals in report.items()
        if isinstance(vals, dict) and class_name in classes_list
    }

    return {
        "target": target,
        "split": split,
        "macro_f1": round(macro_f1, 4),
        "accuracy": round(accuracy, 4),
        "ece": None,
        "per_class": per_class,
        "class_names": classes_list,
    }


def compute_rule_override_audit(
    y_true: np.ndarray,
    y_pred_before: list[str],
    y_pred_after: list[str],
    class_names: np.ndarray,
    target: str,
    split: str = "test",
) -> dict:
    """
    Compare model-only vs rule-adjusted labels against ground truth.

    For each row where the predicted grade string **changes** after rules:
      * **helpful** — model wrong, rules correct
      * **harmful** — model right, rules wrong
      * **neutral** — model wrong, rules still wrong (override did not fix)

    ``override_precision`` is helpful / (helpful + harmful): when rules flip a
    correct prediction to a wrong one, that counts as harmful (the primary
    “bad override” signal).

    Also reports overall accuracy / macro-F1 before and after (full split).
    """
    if not (len(y_true) == len(y_pred_before) == len(y_pred_after)):
        raise ValueError(
            "y_true and y_pred_before / y_pred_after must have the same length"
        )

    y_t2, _, (y_b, y_a) = remap_true_and_encode_predictions(
        y_true, class_names, y_pred_before, y_pred_after
    )

    correct_b = y_b == y_t2
    correct_a = y_a == y_t2
    changed = np.array(
        [str(a) != str(b) for a, b in zip(y_pred_before, y_pred_after)],
        dtype=bool,
    )

    n_changed = int(changed.sum())
    ch = changed
    mask_helpful = ch & (~correct_b) & correct_a
    mask_harmful = ch & correct_b & (~correct_a)
    mask_neutral = ch & (~correct_b) & (~correct_a)

    n_helpful = int(mask_helpful.sum())
    n_harmful = int(mask_harmful.sum())
    n_neutral = int(mask_neutral.sum())

    macro_f1_b = float(f1_score(y_t2, y_b, average="macro"))
    macro_f1_a = float(f1_score(y_t2, y_a, average="macro"))
    acc_b = float(accuracy_score(y_t2, y_b))
    acc_a = float(accuracy_score(y_t2, y_a))

    denom = n_helpful + n_harmful
    override_precision = (
        round(n_helpful / denom, 4) if denom > 0 else None
    )

    return {
        "target": target,
        "split": split,
        "accuracy_before": round(acc_b, 4),
        "accuracy_after": round(acc_a, 4),
        "delta_accuracy": round(acc_a - acc_b, 4),
        "macro_f1_before": round(macro_f1_b, 4),
        "macro_f1_after": round(macro_f1_a, 4),
        "delta_macro_f1": round(macro_f1_a - macro_f1_b, 4),
        "n_changed": n_changed,
        "n_helpful": n_helpful,
        "n_harmful": n_harmful,
        "n_neutral": n_neutral,
        "override_precision": override_precision,
    }


# ---------------------------------------------------------------------------
# Expected Calibration Error
# ---------------------------------------------------------------------------
def compute_ece(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Compute Expected Calibration Error (ECE).

    Bins predictions by confidence (max predicted probability),
    compares mean confidence to mean accuracy within each bin,
    and returns the weighted average absolute difference.

    Lower is better. 0.0 = perfectly calibrated.

    Args:
        y_true:   ground truth integer label array
        y_proba:  predicted probability matrix (n_samples x n_classes)
        n_bins:   number of confidence bins (default 10)

    Returns:
        ECE as a float.
    """
    confidences = y_proba.max(axis=1)
    predictions = y_proba.argmax(axis=1)
    correct = (predictions == y_true).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
        if not np.any(mask):
            continue
        bin_accuracy = correct[mask].mean()
        bin_confidence = confidences[mask].mean()
        bin_weight = mask.sum() / len(y_true)
        ece += bin_weight * abs(bin_accuracy - bin_confidence)

    return float(ece)


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------
def format_metrics_report(metrics: dict) -> str:
    """
    Format a single metrics dict as a human-readable report string.

    Args:
        metrics: output of compute_metrics()

    Returns:
        Formatted string suitable for printing or saving to a file.
    """
    target = metrics["target"].upper()
    split = metrics["split"].upper()

    lines = [
        "=" * 55,
        f"EVALUATION REPORT — {target} TARGET — {split} SPLIT",
        "=" * 55,
        "",
        f"  Macro-F1:  {metrics['macro_f1']:.4f}",
        f"  Accuracy:  {metrics['accuracy']:.4f}",
        (
            f"  ECE:       {metrics['ece']:.4f}"
            if metrics.get("ece") is not None
            else "  ECE:       N/A (no probabilities — e.g. rule-adjusted)"
        ),
        "",
        f"  {'Grade':<20} {'F1':>6} {'Prec':>6} {'Rec':>6} {'N':>6}",
        "  " + "-" * 47,
    ]

    for class_name, vals in metrics["per_class"].items():
        lines.append(
            f"  {class_name:<20} "
            f"{vals['f1']:>6.4f} "
            f"{vals['precision']:>6.4f} "
            f"{vals['recall']:>6.4f} "
            f"{vals['support']:>6}"
        )

    lines += ["", "=" * 55, ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------
def compare_models(
    baseline_metrics: dict[str, dict],
    transformer_metrics: dict[str, dict],
    split: str = "test",
) -> str:
    """
    Generate a side-by-side comparison table between baseline and
    transformer metrics. Designed for the resume framing goal —
    clearly shows macro-F1 improvement.

    Args:
        baseline_metrics:    dict mapping target → metrics dict
        transformer_metrics: dict mapping target → metrics dict
        split:               split name for display (default "test")

    Returns:
        Formatted comparison table string.
    """
    targets = ["sleeve", "media"]
    metric_keys = [
        ("macro_f1", "Macro-F1", True),  # (key, label, higher_is_better)
        ("accuracy", "Accuracy", True),
        ("ece", "ECE", False),
    ]

    lines = [
        "=" * 61,
        f"MODEL COMPARISON — {split.upper()} SPLIT",
        "=" * 61,
        f"  {'Metric':<28} {'Baseline':>9} {'Transformer':>12} {'Δ':>8}",
        "  " + "-" * 57,
    ]

    for target in targets:
        b = baseline_metrics.get(target, {})
        t = transformer_metrics.get(target, {})

        for key, label, higher_better in metric_keys:
            b_val = b.get(key, 0.0)
            t_val = t.get(key, 0.0)
            delta = t_val - b_val

            # Format delta with sign and improvement indicator
            delta_str = f"{delta:+.4f}"
            improved = (delta > 0) == higher_better
            indicator = "✓" if improved else "✗"

            row_label = f"{target.capitalize()} {label}"
            lines.append(
                f"  {row_label:<28} "
                f"{b_val:>9.4f} "
                f"{t_val:>12.4f} "
                f"{delta_str:>7} {indicator}"
            )

        # Blank line between targets
        if target != targets[-1]:
            lines.append("  " + "-" * 57)

    lines += [
        "=" * 61,
        "  ✓ = improvement  ✗ = regression",
        "=" * 61,
        "",
    ]

    return "\n".join(lines)


def compare_models_per_class(
    baseline_metrics: dict[str, dict],
    transformer_metrics: dict[str, dict],
    *,
    split_title: str = "TEST SPLIT",
) -> str:
    """
    Generate per-class F1 comparison between baseline and transformer.
    Shows which grades improved most — useful for understanding where
    the transformer adds value over TF-IDF.

    Args:
        baseline_metrics:    dict mapping target → metrics dict
        transformer_metrics: dict mapping target → metrics dict
        split_title:         label for the report header (e.g. ``TEST_THIN``)

    Returns:
        Formatted per-class comparison string.
    """
    lines = [
        "=" * 61,
        f"PER-CLASS F1 COMPARISON — {split_title}",
        "=" * 61,
    ]

    for target in ["sleeve", "media"]:
        b = baseline_metrics.get(target, {})
        t = transformer_metrics.get(target, {})
        class_names = b.get("class_names", [])

        lines += [
            "",
            f"  {target.upper()} TARGET",
            f"  {'Grade':<22} {'Baseline F1':>12} {'Transformer F1':>15} {'Δ':>8}",
            "  " + "-" * 59,
        ]

        for class_name in class_names:
            b_f1 = b.get("per_class", {}).get(class_name, {}).get("f1", 0.0)
            t_f1 = t.get("per_class", {}).get(class_name, {}).get("f1", 0.0)
            delta = t_f1 - b_f1
            delta_str = f"{delta:+.4f}"
            lines.append(
                f"  {class_name:<22} "
                f"{b_f1:>12.4f} "
                f"{t_f1:>15.4f} "
                f"{delta_str:>8}"
            )

    lines += ["", "=" * 61, ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# MLflow logging utilities
# ---------------------------------------------------------------------------
def log_metrics_to_mlflow(
    metrics: dict,
    prefix: Optional[str] = None,
) -> None:
    """
    Log a metrics dict to the active MLflow run.

    Args:
        metrics: output of compute_metrics()
        prefix:  optional prefix to prepend to all metric keys
                 (e.g. "baseline" → "baseline_test_sleeve_macro_f1")
    """
    target = metrics["target"]
    split = metrics["split"]
    base = f"{split}_{target}"
    if prefix:
        base = f"{prefix}_{base}"

    mlflow.log_metrics(
        {
            f"{base}_macro_f1": metrics["macro_f1"],
            f"{base}_accuracy": metrics["accuracy"],
            f"{base}_ece": metrics["ece"],
        }
    )

    # Per-class F1 — primary test splits only to keep MLflow uncluttered
    if split in ("test", "test_thin"):
        for class_name, vals in metrics["per_class"].items():
            clean = class_name.lower().replace(" ", "_")
            mlflow.log_metric(f"{base}_{clean}_f1", vals["f1"])


def log_comparison_to_mlflow(
    baseline_metrics: dict[str, dict],
    transformer_metrics: dict[str, dict],
    *,
    key_suffix: str = "",
) -> None:
    """
    Log model comparison deltas to the active MLflow run.
    Logs the macro-F1 improvement for each target as a dedicated metric.
    """
    suf = f"_{key_suffix}" if key_suffix else ""
    for target in ["sleeve", "media"]:
        b_f1 = baseline_metrics.get(target, {}).get("macro_f1", 0.0)
        t_f1 = transformer_metrics.get(target, {}).get("macro_f1", 0.0)
        mlflow.log_metric(
            f"macro_f1_improvement_{target}{suf}", round(t_f1 - b_f1, 4)
        )
