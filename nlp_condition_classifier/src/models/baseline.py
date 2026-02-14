"""
Baseline: TF-IDF + Logistic Regression for sleeve and media condition.

- Two separate classifiers (sleeve, media) or one multi-output setup
- Outputs predicted label and confidence scores per class
"""
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

from ..data.ingest import CONDITION_GRADES
from ..features.tfidf_features import build_tfidf_vectorizer, tfidf_transform


class BaselineConditionClassifier:
    """
    Predicts sleeve and media condition from text using TF-IDF + Logistic Regression.
    Holds vectorizer + two LR models (sleeve, media) and label encoders.
    """

    def __init__(
        self,
        tfidf_config: dict[str, Any] | None = None,
        logistic_config: dict[str, Any] | None = None,
        random_state: int = 42,
    ):
        self.tfidf_config = tfidf_config or {}
        self.logistic_config = logistic_config or {}
        self.random_state = random_state
        self.classes_ = CONDITION_GRADES
        self.vectorizer_: TfidfVectorizer | None = None
        self.sleeve_clf_: LogisticRegression | None = None
        self.media_clf_: LogisticRegression | None = None

    def fit(
        self,
        X_text: list[str],
        y_sleeve: np.ndarray | list[str],
        y_media: np.ndarray | list[str],
    ) -> "BaselineConditionClassifier":
        """Fit TF-IDF on X_text and two Logistic Regression models."""
        y_sleeve = np.asarray(y_sleeve)
        y_media = np.asarray(y_media)
        self.vectorizer_ = build_tfidf_vectorizer(**self.tfidf_config)
        X = self.vectorizer_.fit_transform(X_text)
        self.sleeve_clf_ = LogisticRegression(
            random_state=self.random_state,
            **self.logistic_config,
        ).fit(X, y_sleeve)
        self.media_clf_ = LogisticRegression(
            random_state=self.random_state,
            **self.logistic_config,
        ).fit(X, y_media)
        return self

    def predict_sleeve(self, X_text: list[str]) -> np.ndarray:
        """Predict sleeve condition labels."""
        X = self.vectorizer_.transform(X_text)
        return self.sleeve_clf_.predict(X)

    def predict_media(self, X_text: list[str]) -> np.ndarray:
        """Predict media condition labels."""
        X = self.vectorizer_.transform(X_text)
        return self.media_clf_.predict(X)

    def predict_proba_sleeve(self, X_text: list[str]) -> np.ndarray:
        """Predict sleeve condition probabilities (order = self.classes_)."""
        X = self.vectorizer_.transform(X_text)
        return self.sleeve_clf_.predict_proba(X)

    def predict_proba_media(self, X_text: list[str]) -> np.ndarray:
        """Predict media condition probabilities (order = self.classes_)."""
        X = self.vectorizer_.transform(X_text)
        return self.media_clf_.predict_proba(X)

    def predict_item(
        self,
        item_id: str,
        seller_notes: str,
    ) -> dict[str, Any]:
        """
        Return single-item prediction in the user-story JSON format.
        """
        texts = [seller_notes]
        sleeve_pred = self.predict_sleeve(texts)[0]
        media_pred = self.predict_media(texts)[0]
        sleeve_proba = self.predict_proba_sleeve(texts)[0]
        # Ensure order matches self.classes_ (same as sklearn's clf.classes_ if aligned)
        sleeve_scores = {
            self.sleeve_clf_.classes_[i]: float(sleeve_proba[i])
            for i in range(len(self.sleeve_clf_.classes_))
        }
        return {
            "item_id": item_id,
            "predicted_sleeve_condition": str(sleeve_pred),
            "predicted_media_condition": str(media_pred),
            "confidence_scores": sleeve_scores,
        }

    def save(self, path: Path | str) -> None:
        """Save vectorizer and both classifiers."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vectorizer_, path / "vectorizer.joblib")
        joblib.dump(self.sleeve_clf_, path / "sleeve_clf.joblib")
        joblib.dump(self.media_clf_, path / "media_clf.joblib")
        joblib.dump(self.classes_, path / "classes.joblib")

    @classmethod
    def load(cls, path: Path | str) -> "BaselineConditionClassifier":
        """Load from artifact directory."""
        path = Path(path)
        self = cls()
        self.vectorizer_ = joblib.load(path / "vectorizer.joblib")
        self.sleeve_clf_ = joblib.load(path / "sleeve_clf.joblib")
        self.media_clf_ = joblib.load(path / "media_clf.joblib")
        self.classes_ = joblib.load(path / "classes.joblib")
        return self


def train_baseline(
    X_train: list[str],
    y_sleeve_train: np.ndarray | list[str],
    y_media_train: np.ndarray | list[str],
    tfidf_config: dict[str, Any] | None = None,
    logistic_config: dict[str, Any] | None = None,
    random_state: int = 42,
) -> BaselineConditionClassifier:
    """Train and return a BaselineConditionClassifier."""
    clf = BaselineConditionClassifier(
        tfidf_config=tfidf_config,
        logistic_config=logistic_config,
        random_state=random_state,
    )
    return clf.fit(X_train, y_sleeve_train, y_media_train)
