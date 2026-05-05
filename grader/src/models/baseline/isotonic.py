"""Post-hoc isotonic calibration wrapper for sklearn logistic heads."""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.isotonic import IsotonicRegression


class _IsotonicCalibrator:
    """
    Post-hoc probability calibrator using per-class isotonic regression.

    Wraps a fitted sklearn classifier and applies a separate isotonic
    regression to each class column of its predict_proba output.

    Critically, this calibrator always outputs ``n_classes`` columns where
    ``n_classes`` is the total number of canonical classes (passed in at
    construction time). If the base model was trained without some rare
    class (e.g. sleeve Excellent never appears in training data), those
    columns stay at zero so downstream blending logic sees a consistent
    shape.

    Drop-in replacement for CalibratedClassifierCV(cv="prefit") which was
    removed in sklearn 1.6+.
    """

    def __init__(self, base_clf, n_classes: int) -> None:
        self.base_clf = base_clf
        self.n_classes = n_classes
        self.calibrators_: list[Optional[IsotonicRegression]] = []
        self.classes_ = None

    def fit(self, X, y) -> "_IsotonicCalibrator":
        self.classes_ = self.base_clf.classes_  # model's actual classes
        model_classes = list(self.classes_)
        raw_proba = self.base_clf.predict_proba(X)  # (n_samples, len(model_classes))

        # One-hot encode y against the model's actual class list
        y_onehot = np.zeros((len(y), len(model_classes)), dtype=float)
        cls_to_col = {c: i for i, c in enumerate(model_classes)}
        for row, label in enumerate(y):
            if label in cls_to_col:
                y_onehot[row, cls_to_col[label]] = 1.0

        # One calibrator per model class
        self.calibrators_ = []
        for j in range(len(model_classes)):
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(raw_proba[:, j], y_onehot[:, j])
            self.calibrators_.append(iso)
        return self

    def predict_proba(self, X) -> np.ndarray:
        raw_proba = self.base_clf.predict_proba(X)
        model_classes = list(self.base_clf.classes_)

        # Build output with the full n_classes columns, mapping model classes
        # to their canonical integer indices (model class values are ints from
        # the LabelEncoder, so they directly index the output columns).
        n_samples = raw_proba.shape[0]
        out = np.zeros((n_samples, self.n_classes), dtype=float)
        for j, cls_idx in enumerate(model_classes):
            out[:, int(cls_idx)] = self.calibrators_[j].predict(raw_proba[:, j])

        row_sum = out.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        return out / row_sum
