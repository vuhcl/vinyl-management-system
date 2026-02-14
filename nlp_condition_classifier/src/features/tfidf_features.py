"""
TF-IDF feature extraction for baseline model.

- Build and fit vectorizer on training texts
- Transform train/val/test for Logistic Regression
"""
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def build_tfidf_vectorizer(
    max_features: int = 10000,
    min_df: int = 2,
    max_df: float = 0.95,
    ngram_range: tuple[int, int] = (1, 2),
    **kwargs: Any,
) -> TfidfVectorizer:
    """Build a TfidfVectorizer with given params."""
    return TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        ngram_range=ngram_range,
        **kwargs,
    )


def tfidf_transform(
    vectorizer: TfidfVectorizer,
    train_texts: list[str],
    val_texts: list[str] | None = None,
    test_texts: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """
    Fit vectorizer on train_texts and transform train (and optionally val, test).
    Returns (X_train, X_val, X_test); val/test are None if not provided.
    """
    X_train = vectorizer.fit_transform(train_texts)
    X_val = vectorizer.transform(val_texts) if val_texts else None
    X_test = vectorizer.transform(test_texts) if test_texts else None
    return X_train, X_val, X_test
