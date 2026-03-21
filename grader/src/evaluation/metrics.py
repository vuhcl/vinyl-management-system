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
        f"  ECE:       {metrics['ece']:.4f}",
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
) -> str:
    """
    Generate per-class F1 comparison between baseline and transformer.
    Shows which grades improved most — useful for understanding where
    the transformer adds value over TF-IDF.

    Args:
        baseline_metrics:    dict mapping target → metrics dict
        transformer_metrics: dict mapping target → metrics dict

    Returns:
        Formatted per-class comparison string.
    """
    lines = [
        "=" * 61,
        "PER-CLASS F1 COMPARISON — TEST SPLIT",
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

    # Per-class F1 — test split only to keep MLflow uncluttered
    if split == "test":
        for class_name, vals in metrics["per_class"].items():
            clean = class_name.lower().replace(" ", "_")
            mlflow.log_metric(f"{base}_{clean}_f1", vals["f1"])


def log_comparison_to_mlflow(
    baseline_metrics: dict[str, dict],
    transformer_metrics: dict[str, dict],
) -> None:
    """
    Log model comparison deltas to the active MLflow run.
    Logs the macro-F1 improvement for each target as a dedicated metric.
    """
    for target in ["sleeve", "media"]:
        b_f1 = baseline_metrics.get(target, {}).get("macro_f1", 0.0)
        t_f1 = transformer_metrics.get(target, {}).get("macro_f1", 0.0)
        mlflow.log_metric(
            f"macro_f1_improvement_{target}", round(t_f1 - b_f1, 4)
        )
