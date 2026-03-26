"""
grader/src/evaluation/calibration.py

Calibration visualization for the vinyl condition grader.
Generates reliability diagrams, confidence histograms, and
per-class calibration curves. Saves plots to grader/reports/
and logs them to MLflow as artifacts.

Kept separate from metrics.py to isolate the matplotlib
dependency — metrics.py can be imported anywhere without
triggering a plot backend.

Usage:
    from grader.src.evaluation.calibration import CalibrationEvaluator

    evaluator = CalibrationEvaluator(config_path="grader/configs/grader.yaml")
    evaluator.run(
        y_true=y_test,
        y_proba=y_proba,
        class_names=encoder.classes_,
        target="sleeve",
        model_name="baseline",
    )
"""

import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for server/CI use
import matplotlib.pyplot as plt
import numpy as np
import yaml
from sklearn.calibration import calibration_curve

from grader.src.evaluation.metrics import compute_ece

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CalibrationEvaluator
# ---------------------------------------------------------------------------
class CalibrationEvaluator:
    """
    Generates calibration visualizations for a trained model.

    Produces three plot types per target per model:
      1. Reliability diagram (overall)
      2. Confidence histogram
      3. Per-class calibration curves

    All plots are saved to grader/reports/calibration/ and logged
    to the active MLflow run as artifacts.

    Config keys read from grader.yaml:
        paths.reports               — base reports directory
        evaluation.calibration.*   — n_bins setting
        mlflow.tracking_uri
        mlflow.experiment_name
    """

    # Plot styling — consistent across all figures
    STYLE = {
        "figure.figsize":   (8, 6),
        "axes.spines.top":  False,
        "axes.spines.right": False,
        "font.size":        11,
    }

    # Grade-specific colors for per-class plots
    GRADE_COLORS = {
        "Mint":           "#2ecc71",
        "Near Mint":      "#27ae60",
        "Excellent":      "#3498db",
        "Very Good Plus": "#2980b9",
        "Very Good":      "#f39c12",
        "Good":           "#e67e22",
        "Poor":           "#e74c3c",
        "Generic":        "#95a5a6",
    }

    def __init__(self, config_path: str) -> None:
        self.config = self._load_yaml(config_path)

        cal_cfg = self.config["evaluation"]["calibration"]
        self.n_bins: int = cal_cfg.get("n_bins", 10)

        reports_dir = Path(self.config["paths"]["reports"])
        self.output_dir = reports_dir / "calibration"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Config loading
    # -----------------------------------------------------------------------
    @staticmethod
    def _load_yaml(path: str) -> dict:
        with open(path, "r") as f:
            return yaml.safe_load(f)

    # -----------------------------------------------------------------------
    # Plot 1 — Reliability diagram (overall)
    # -----------------------------------------------------------------------
    def reliability_diagram(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        target: str,
        model_name: str,
    ) -> Path:
        """
        Plot overall reliability diagram.

        X axis: mean predicted confidence per bin
        Y axis: fraction of correct predictions per bin
        Diagonal: perfect calibration reference

        Uses the confidence of the predicted class (max probability)
        as the calibration signal — consistent with ECE computation.

        Args:
            y_true:     ground truth integer label array
            y_proba:    predicted probability matrix
            target:     "sleeve" or "media"
            model_name: "baseline" or "transformer" — for title/filename

        Returns:
            Path to saved plot.
        """
        confidences = y_proba.max(axis=1)
        predictions = y_proba.argmax(axis=1)
        correct     = (predictions == y_true).astype(float)

        bins         = np.linspace(0.0, 1.0, self.n_bins + 1)
        bin_centers  = []
        bin_accuracy = []
        bin_counts   = []

        for i in range(self.n_bins):
            mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
            if not np.any(mask):
                continue
            bin_centers.append(confidences[mask].mean())
            bin_accuracy.append(correct[mask].mean())
            bin_counts.append(mask.sum())

        ece = compute_ece(y_true, y_proba, self.n_bins)

        with plt.rc_context(self.STYLE):
            fig, ax = plt.subplots()

            # Perfect calibration diagonal
            ax.plot(
                [0, 1], [0, 1],
                linestyle="--",
                color="#bdc3c7",
                linewidth=1.5,
                label="Perfect calibration",
                zorder=1,
            )

            # Bar chart of bin accuracy
            bar_width = 1.0 / self.n_bins
            ax.bar(
                bin_centers,
                bin_accuracy,
                width=bar_width * 0.8,
                alpha=0.7,
                color="#3498db",
                label="Model",
                zorder=2,
            )

            # Gap fill — shows over/underconfidence
            ax.bar(
                bin_centers,
                [c - a for c, a in zip(bin_centers, bin_accuracy)],
                bottom=bin_accuracy,
                width=bar_width * 0.8,
                alpha=0.25,
                color="#e74c3c",
                label="Gap (miscalibration)",
                zorder=2,
            )

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel("Mean predicted confidence")
            ax.set_ylabel("Fraction correct")
            ax.set_title(
                f"Reliability Diagram — {target.capitalize()} "
                f"({model_name}) | ECE={ece:.4f}"
            )
            ax.legend(loc="upper left", fontsize=9)

            path = self.output_dir / f"{model_name}_{target}_reliability.png"
            fig.tight_layout()
            fig.savefig(path, dpi=150)
            plt.close(fig)

        logger.info("Saved reliability diagram: %s", path)
        return path

    # -----------------------------------------------------------------------
    # Plot 2 — Confidence histogram
    # -----------------------------------------------------------------------
    def confidence_histogram(
        self,
        y_proba: np.ndarray,
        target: str,
        model_name: str,
    ) -> Path:
        """
        Plot distribution of max predicted probabilities.

        Shows whether the model is overconfident (mass near 1.0),
        underconfident (mass near 0.5), or well-spread.

        Args:
            y_proba:    predicted probability matrix
            target:     "sleeve" or "media"
            model_name: "baseline" or "transformer"

        Returns:
            Path to saved plot.
        """
        confidences = y_proba.max(axis=1)
        mean_conf   = confidences.mean()
        median_conf = np.median(confidences)

        with plt.rc_context(self.STYLE):
            fig, ax = plt.subplots()

            ax.hist(
                confidences,
                bins=self.n_bins,
                range=(0, 1),
                color="#3498db",
                alpha=0.75,
                edgecolor="white",
                linewidth=0.5,
            )

            ax.axvline(
                mean_conf,
                color="#e74c3c",
                linewidth=1.5,
                linestyle="--",
                label=f"Mean: {mean_conf:.3f}",
            )
            ax.axvline(
                median_conf,
                color="#f39c12",
                linewidth=1.5,
                linestyle=":",
                label=f"Median: {median_conf:.3f}",
            )

            ax.set_xlabel("Max predicted probability (confidence)")
            ax.set_ylabel("Count")
            ax.set_title(
                f"Confidence Distribution — {target.capitalize()} "
                f"({model_name})"
            )
            ax.legend(fontsize=9)

            path = self.output_dir / f"{model_name}_{target}_confidence_hist.png"
            fig.tight_layout()
            fig.savefig(path, dpi=150)
            plt.close(fig)

        logger.info("Saved confidence histogram: %s", path)
        return path

    # -----------------------------------------------------------------------
    # Plot 3 — Per-class calibration curves
    # -----------------------------------------------------------------------
    def per_class_calibration(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        class_names: np.ndarray,
        target: str,
        model_name: str,
    ) -> Path:
        """
        Plot calibration curve for each grade class.

        For each class, treats the problem as binary (class vs rest)
        and plots predicted probability vs actual fraction positive.

        Reveals which grades are well-calibrated and which are not.
        Poor and Generic are expected to have noisier curves due to
        low sample counts.

        Args:
            y_true:      ground truth integer label array
            y_proba:     predicted probability matrix
            class_names: ordered grade strings matching label encoder
            target:      "sleeve" or "media"
            model_name:  "baseline" or "transformer"

        Returns:
            Path to saved plot.
        """
        n_classes = len(class_names)
        n_cols    = 2
        n_rows    = (n_classes + 1) // n_cols

        with plt.rc_context(self.STYLE):
            fig, axes = plt.subplots(
                n_rows, n_cols,
                figsize=(10, n_rows * 3.5),
                squeeze=False,
            )

            for idx, class_name in enumerate(class_names):
                row = idx // n_cols
                col = idx % n_cols
                ax  = axes[row][col]

                # Binary: this class vs rest
                y_binary    = (y_true == idx).astype(int)
                class_proba = y_proba[:, idx]

                # Skip if class has no positive samples in this split
                if y_binary.sum() == 0:
                    ax.text(
                        0.5, 0.5,
                        f"No samples\nfor {class_name}",
                        ha="center", va="center",
                        transform=ax.transAxes,
                        color="#7f8c8d",
                    )
                    ax.set_title(class_name, fontsize=10)
                    continue

                try:
                    fraction_pos, mean_pred = calibration_curve(
                        y_binary,
                        class_proba,
                        n_bins=min(self.n_bins, y_binary.sum()),
                        strategy="uniform",
                    )
                except ValueError:
                    # Too few samples for calibration curve
                    ax.text(
                        0.5, 0.5,
                        f"Too few samples\nfor {class_name}",
                        ha="center", va="center",
                        transform=ax.transAxes,
                        color="#7f8c8d",
                    )
                    ax.set_title(class_name, fontsize=10)
                    continue

                color = self.GRADE_COLORS.get(class_name, "#3498db")

                # Perfect calibration reference
                ax.plot(
                    [0, 1], [0, 1],
                    linestyle="--",
                    color="#bdc3c7",
                    linewidth=1,
                    zorder=1,
                )

                # Class calibration curve
                ax.plot(
                    mean_pred,
                    fraction_pos,
                    marker="o",
                    markersize=5,
                    color=color,
                    linewidth=1.5,
                    label=class_name,
                    zorder=2,
                )

                # Support annotation
                support = y_binary.sum()
                ax.text(
                    0.97, 0.05,
                    f"n={support}",
                    ha="right", va="bottom",
                    transform=ax.transAxes,
                    fontsize=8,
                    color="#7f8c8d",
                )

                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_title(class_name, fontsize=10, color=color)
                ax.set_xlabel("Mean predicted prob.", fontsize=8)
                ax.set_ylabel("Fraction positive", fontsize=8)

            # Hide any unused subplots
            for idx in range(n_classes, n_rows * n_cols):
                axes[idx // n_cols][idx % n_cols].set_visible(False)

            fig.suptitle(
                f"Per-Class Calibration — {target.capitalize()} ({model_name})",
                fontsize=12,
                y=1.02,
            )

            path = self.output_dir / f"{model_name}_{target}_per_class_calibration.png"
            fig.tight_layout()
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)

        logger.info("Saved per-class calibration plot: %s", path)
        return path

    # -----------------------------------------------------------------------
    # Before/after calibration comparison
    # -----------------------------------------------------------------------
    def calibration_comparison(
        self,
        y_true: np.ndarray,
        y_proba_before: np.ndarray,
        y_proba_after: np.ndarray,
        target: str,
        model_name: str,
    ) -> Path:
        """
        Side-by-side reliability diagrams before and after
        isotonic calibration. Shows the effect of post-hoc calibration.

        Args:
            y_true:          ground truth integer label array
            y_proba_before:  uncalibrated probability matrix
            y_proba_after:   calibrated probability matrix
            target:          "sleeve" or "media"
            model_name:      "baseline" or "transformer"

        Returns:
            Path to saved plot.
        """
        ece_before = compute_ece(y_true, y_proba_before)
        ece_after  = compute_ece(y_true, y_proba_after)

        with plt.rc_context(self.STYLE):
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            for ax, y_proba, label, ece in [
                (axes[0], y_proba_before, "Before calibration", ece_before),
                (axes[1], y_proba_after,  "After calibration",  ece_after),
            ]:
                confidences = y_proba.max(axis=1)
                predictions = y_proba.argmax(axis=1)
                correct     = (predictions == y_true).astype(float)

                bins         = np.linspace(0, 1, self.n_bins + 1)
                bin_centers  = []
                bin_accuracy = []

                for i in range(self.n_bins):
                    mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
                    if not np.any(mask):
                        continue
                    bin_centers.append(confidences[mask].mean())
                    bin_accuracy.append(correct[mask].mean())

                ax.plot(
                    [0, 1], [0, 1],
                    linestyle="--",
                    color="#bdc3c7",
                    linewidth=1.5,
                    zorder=1,
                )
                bar_width = 1.0 / self.n_bins
                ax.bar(
                    bin_centers,
                    bin_accuracy,
                    width=bar_width * 0.8,
                    alpha=0.7,
                    color="#3498db",
                    zorder=2,
                )
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_xlabel("Mean predicted confidence")
                ax.set_ylabel("Fraction correct")
                ax.set_title(f"{label} | ECE={ece:.4f}")

            fig.suptitle(
                f"Calibration Effect — {target.capitalize()} ({model_name})",
                fontsize=12,
            )

            path = self.output_dir / f"{model_name}_{target}_calibration_comparison.png"
            fig.tight_layout()
            fig.savefig(path, dpi=150)
            plt.close(fig)

        logger.info("Saved calibration comparison: %s", path)
        return path

    # -----------------------------------------------------------------------
    # Orchestration
    # -----------------------------------------------------------------------
    def run(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        class_names: np.ndarray,
        target: str,
        model_name: str,
        y_proba_uncalibrated: Optional[np.ndarray] = None,
        log_to_mlflow: bool = True,
    ) -> list[Path]:
        """
        Generate all calibration plots for a single target and model.

        Args:
            y_true:               ground truth integer label array
            y_proba:              calibrated probability matrix
            class_names:          ordered grade strings
            target:               "sleeve" or "media"
            model_name:           "baseline" or "transformer"
            y_proba_uncalibrated: if provided, generates before/after
                                  comparison plot
            log_to_mlflow:        whether to log plots to active MLflow run

        Returns:
            List of paths to all saved plot files.
        """
        import mlflow

        plot_paths = []

        # Plot 1 — Reliability diagram
        path = self.reliability_diagram(y_true, y_proba, target, model_name)
        plot_paths.append(path)

        # Plot 2 — Confidence histogram
        path = self.confidence_histogram(y_proba, target, model_name)
        plot_paths.append(path)

        # Plot 3 — Per-class calibration curves
        path = self.per_class_calibration(
            y_true, y_proba, class_names, target, model_name
        )
        plot_paths.append(path)

        # Plot 4 — Before/after comparison (if uncalibrated probas provided)
        if y_proba_uncalibrated is not None:
            path = self.calibration_comparison(
                y_true,
                y_proba_uncalibrated,
                y_proba,
                target,
                model_name,
            )
            plot_paths.append(path)

        # Log all plots to MLflow
        if log_to_mlflow:
            for path in plot_paths:
                mlflow.log_artifact(str(path))
            logger.info(
                "Logged %d calibration plots to MLflow — "
                "target=%s model=%s",
                len(plot_paths),
                target,
                model_name,
            )

        return plot_paths


# ---------------------------------------------------------------------------
# Type hint fix — Optional not imported at module level
# ---------------------------------------------------------------------------
from typing import Optional  # noqa: E402
