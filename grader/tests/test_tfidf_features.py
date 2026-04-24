"""
grader/tests/test_tfidf_features.py

Tests for tfidf_features.py — fit-on-train-only enforcement,
output shapes, label encoding, and artifact persistence.
"""

import pickle
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from grader.src.features.tfidf_features import TFIDFFeatureBuilder


@pytest.fixture
def builder(test_config):
    return TFIDFFeatureBuilder(config_path=test_config)


class TestExtractTextStrayDigits:
    def test_extract_texts_strips_us_shipping_tail_on_raw_text(self, builder):
        """Parity with preprocess: promo noise runs before leading-digit strip."""
        raw = (
            " / $6.40 unlimited us-shipping / free on $100 orders of 3+ items read "
            "seller terms before paying"
        )
        records = [
            {
                "text_clean": "",
                "text": raw,
                "sleeve_label": "Mint",
                "media_label": "Mint",
            }
        ]
        texts = builder.extract_texts(records)
        assert texts[0].strip() == ""

    def test_strips_boilerplate_digit_in_text_clean(self, builder):
        records = [
            {
                "text_clean": "6 sealed nm hype sticker",
                "sleeve_label": "Mint",
                "media_label": "Mint",
            }
        ]
        texts = builder.extract_texts(records)
        assert "6" not in texts[0].split()
        assert "sealed" in texts[0]

    def test_strips_when_falling_back_to_raw_text(self, builder):
        records = [
            {
                "text_clean": "",
                "text": "6 sealed new",
                "sleeve_label": "Mint",
                "media_label": "Mint",
            }
        ]
        texts = builder.extract_texts(records)
        assert "6" not in texts[0].split()
        assert "sealed" in texts[0]


class TestVectorizerFitting:
    def test_vectorizer_fits_on_train(
        self, builder, split_jsonl_paths
    ):
        """Vectorizer must only be fitted using train data."""
        train_records = builder.load_split("train")
        train_texts   = builder.extract_texts(train_records)
        vec = builder.build_vectorizer(train_texts, target="sleeve")
        assert vec.vocabulary_ is not None
        assert len(vec.vocabulary_) > 0

    def test_vocabulary_size_bounded(
        self, builder, split_jsonl_paths
    ):
        train_records = builder.load_split("train")
        train_texts   = builder.extract_texts(train_records)
        vec = builder.build_vectorizer(train_texts, target="sleeve")
        # vocab should not exceed max_features from config (500 for tests)
        assert len(vec.vocabulary_) <= 500

    def test_transform_returns_sparse(
        self, builder, fitted_vectorizers, split_jsonl_paths
    ):
        val_records = builder.load_split("val")
        val_texts   = builder.extract_texts(val_records)
        X = builder.transform(fitted_vectorizers["sleeve"], val_texts, "val", "sleeve")
        assert sp.issparse(X)

    def test_transform_shape_correct(
        self, builder, fitted_vectorizers, split_jsonl_paths
    ):
        val_records = builder.load_split("val")
        val_texts   = builder.extract_texts(val_records)
        X = builder.transform(fitted_vectorizers["sleeve"], val_texts, "val", "sleeve")
        assert X.shape[0] == len(val_records)


class TestLabelEncoder:
    def test_encoder_fitted_on_train_only(
        self, builder, split_jsonl_paths, fitted_encoders
    ):
        train_records = builder.load_split("train")
        train_labels  = builder.extract_labels(train_records, "sleeve")
        encoder = builder.build_encoder(train_labels, target="sleeve")
        assert len(encoder.classes_) > 0

    def test_encoder_includes_guidelines_grades_absent_from_train(
        self, builder
    ):
        """Sleeve schema includes e.g. Excellent even if train is Discogs-only."""
        train_labels = ["Very Good", "Near Mint", "Mint"]
        encoder = builder.build_encoder(train_labels, target="sleeve")
        assert "Excellent" in list(encoder.classes_)
        assert "Generic" in list(encoder.classes_)

    def test_encoder_covers_all_seen_classes(
        self, builder, split_jsonl_paths
    ):
        train_records = builder.load_split("train")
        train_labels  = builder.extract_labels(train_records, "sleeve")
        encoder       = builder.build_encoder(train_labels, "sleeve")
        for label in train_labels:
            assert label in encoder.classes_


class TestArtifactPersistence:
    def test_vectorizer_saved_and_loaded(
        self, builder, fitted_vectorizers, tmp_dirs
    ):
        path = tmp_dirs["artifacts"] / "tfidf_vectorizer_sleeve.pkl"
        builder.save_vectorizer(fitted_vectorizers["sleeve"], "sleeve")
        loaded = TFIDFFeatureBuilder.load_vectorizer(str(path))
        assert loaded.vocabulary_ == fitted_vectorizers["sleeve"].vocabulary_

    def test_encoder_saved_and_loaded(
        self, builder, fitted_encoders, tmp_dirs
    ):
        path = tmp_dirs["artifacts"] / "label_encoder_sleeve.pkl"
        builder.save_encoder(fitted_encoders["sleeve"], "sleeve")
        loaded = TFIDFFeatureBuilder.load_encoder(str(path))
        assert list(loaded.classes_) == list(fitted_encoders["sleeve"].classes_)

    def test_features_saved_and_loaded(
        self, builder, sample_feature_matrices, tmp_dirs
    ):
        X_orig = sample_feature_matrices["train"]["sleeve"]["X"]
        y_orig = sample_feature_matrices["train"]["sleeve"]["y"]
        builder.save_features(X_orig, y_orig, "train", "sleeve")

        X_loaded, y_loaded = TFIDFFeatureBuilder.load_features(
            str(tmp_dirs["artifacts"] / "features"), "train", "sleeve"
        )
        assert X_loaded.shape == X_orig.shape
        assert np.array_equal(y_loaded, y_orig)


class TestTopTerms:
    def test_top_terms_per_grade(
        self, builder, fitted_vectorizers, fitted_encoders,
        sample_feature_matrices
    ):
        X_train = sample_feature_matrices["train"]["sleeve"]["X"]
        y_train = sample_feature_matrices["train"]["sleeve"]["y"]
        terms = builder.get_top_terms(
            fitted_vectorizers["sleeve"],
            fitted_encoders["sleeve"],
            X_train,
            y_train,
            n_terms=5,
        )
        assert len(terms) == len(fitted_encoders["sleeve"].classes_)
        for grade, grade_terms in terms.items():
            assert isinstance(grade_terms, list)
