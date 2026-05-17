"""
grader/src/models/baseline/

Two-head logistic regression baseline for vinyl condition grading.
One head per target (sleeve, media) trained on TF-IDF features.

Pipeline:
  1. Load pre-built TF-IDF features from artifacts/features/
  2. Fit two LogisticRegression heads on train split
  3. Calibrate probabilities on val split using isotonic regression
  4. Evaluate on val and test splits
  5. Save fitted models and log metrics to MLflow

Rule engine post-processing is NOT applied here.
That responsibility belongs to pipeline.py.
This package is a pure model training and evaluation unit.

Output artifacts:
  grader/artifacts/baseline_sleeve.pkl
  grader/artifacts/baseline_media.pkl
  grader/artifacts/baseline_sleeve_calibrated.pkl
  grader/artifacts/baseline_media_calibrated.pkl
  grader/artifacts/confusion_matrix_sleeve.txt
  grader/artifacts/confusion_matrix_media.txt

Usage:
    python -m grader.src.models.baseline
    python -m grader.src.models.baseline --dry-run
"""

from .constants import SPLITS, TARGETS
from .model import BaselineModel

__all__ = ["BaselineModel", "SPLITS", "TARGETS"]
