"""
grader/tests/conftest.py

Shared pytest fixtures for vinyl condition grader tests.
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

SLEEVE_GRADES = [
    "Mint",
    "Near Mint",
    "Excellent",
    "Very Good Plus",
    "Very Good",
    "Good",
    "Poor",
    "Generic",
]
MEDIA_GRADES = [
    "Mint",
    "Near Mint",
    "Excellent",
    "Very Good Plus",
    "Very Good",
    "Good",
    "Poor",
]

GRADE_TEXTS = {
    "Mint": "factory sealed, still in shrink wrap, never opened",
    "Near Mint": "never played, no marks whatsoever, barely played once",
    "Excellent": "minor scuff on cover, well cared for, excellent condition",
    "Very Good Plus": "plays perfectly, very light scratch, small seam split",
    "Very Good": "surface noise on quiet passages, light scratches on vinyl",
    "Good": "heavy scratches, crackling throughout, seam split at spine",
    "Poor": "badly warped, skipping on side two, won't play properly",
    "Generic": "generic white sleeve, die-cut inner sleeve only",
}

GRADE_TEXT_VARIANTS = {
    "Mint": [
        "still sealed in original shrink",
        "factory sealed, unplayed",
        "sealed copy, mint condition",
        "new and sealed",
        "shrink intact, unplayed",
    ],
    "Near Mint": [
        "one play only, no marks at all",
        "like new, no visible wear",
        "mint minus, barely used",
        "no defects, excellent shape",
        "barely played, pristine",
    ],
    "Excellent": [
        "slight scuff on cover only",
        "very minor wear, carefully handled",
        "light marks on sleeve, plays great",
        "well cared for, minor cosmetic wear",
        "excellent shape, minor blemish",
    ],
    "Very Good Plus": [
        "plays perfectly, minor cosmetic wear only",
        "light scuff on cover, sounds great",
        "very light scratch, plays fine",
        "cosmetic wear only, no audio issues",
        "plays well, turned up corners",
    ],
    "Very Good": [
        "some surface noise, visible scratches",
        "audible noise on quiet passages",
        "plays with some noise, worn",
        "groove wear evident, noisy",
        "light scratches affect sound",
    ],
    "Good": [
        "significant crackling, seam split",
        "heavy wear throughout, tape on cover",
        "lots of surface noise, crackle",
        "writing on label, heavy scratches",
        "plays through, heavy wear",
    ],
    "Poor": [
        "skipping repeatedly, unplayable",
        "cracked, badly warped record",
        "won't play through without skipping",
        "heavily damaged, deep gouges",
        "groove damage, won't play",
    ],
    "Generic": [
        "plain white sleeve, no original cover",
        "company sleeve only, no original",
        "promo copy, generic sleeve",
        "die cut sleeve, missing original cover",
        "blank sleeve, no artwork",
    ],
}


@pytest.fixture(scope="session")
def tmp_dirs(tmp_path_factory):
    base = tmp_path_factory.mktemp("grader_test")
    mlflow_dir = base / "mlflow"
    dirs = {
        "raw": base / "data" / "raw",
        "processed": base / "data" / "processed",
        "splits": base / "data" / "splits",
        "artifacts": base / "artifacts",
        "reports": base / "reports",
        "mlflow_dir": mlflow_dir,
        # SQLite file — parent dir only is created (do not mkdir this path).
        "mlflow_db": mlflow_dir / "tracking.db",
    }
    for key, d in dirs.items():
        if key == "mlflow_db":
            continue
        d.mkdir(parents=True, exist_ok=True)
    (dirs["artifacts"] / "features").mkdir(parents=True, exist_ok=True)
    return dirs


@pytest.fixture(scope="session")
def guidelines_path():
    path = Path("grader/configs/grading_guidelines.yaml")
    if not path.exists():
        pytest.skip("grading_guidelines.yaml not found — run from repo root")
    return str(path)


@pytest.fixture(scope="session")
def test_config(tmp_dirs, guidelines_path):
    config = {
        "guidelines_path": guidelines_path,
        "paths": {
            "raw": str(tmp_dirs["raw"]),
            "processed": str(tmp_dirs["processed"]),
            "splits": str(tmp_dirs["splits"]),
            "artifacts": str(tmp_dirs["artifacts"]),
            "reports": str(tmp_dirs["reports"]),
        },
        "discogs": {
            "base_url": "https://api.discogs.com",
            "token": "TEST_TOKEN",
            "rate_limit_per_minute": 60,
        },
        "ebay": {
            "base_url": "https://api.ebay.com/buy/browse/v1",
            "token_url": "https://api.ebay.com/identity/v1/oauth2/token",
            "client_id": "TEST_CLIENT_ID",
            "client_secret": "TEST_CLIENT_SECRET",
            "rate_limit_per_minute": 45,
        },
        "data": {
            "sources": ["discogs", "ebay_jp"],
            "discogs": {
                "format_filter": "Vinyl",
                "target_per_grade": 10,
                "max_public_inventory_pages": 100,
                "inventory_sellers": ["fixture_seller"],
                "generic_note_filter": {
                    "enabled": False,
                    "strip_boilerplate": False,
                },
            },
            "ebay": {
                "min_text_length": 3,
                "trusted_sellers": {
                    "facerecords": {
                        "sleeve_field": "Sleeve Grading",
                        "media_field": "Record Grading",
                        "obi_field": "OBI_Grading",
                        "grade_format": "clean",
                    },
                    "ellarecords2005": {
                        "sleeve_field": "Cover Condition",
                        "media_field": "Vinyl Condition",
                        "obi_field": "OBI Condition",
                        "grade_format": "annotated",
                    },
                },
            },
            "splits": {
                "train": 0.70,
                "val": 0.15,
                "test": 0.15,
                "stratify_by": ["sleeve_label", "media_label"],
                "random_seed": 42,
            },
            "harmonization": {
                "min_samples_per_class": 2,
                "output_path": str(tmp_dirs["processed"] / "unified.jsonl"),
                "report_path": str(
                    tmp_dirs["reports"] / "class_distribution.txt"
                ),
            },
        },
        "preprocessing": {
            "lowercase": True,
            "normalize_whitespace": True,
            "abbreviation_map": {
                "m-": "mint minus",
                "nm": "near mint",
                "n.m.": "near mint",
                "vg++": "very good plus",
                "vg+": "very good plus",
                "vg": "very good",
                "g+": "good plus",
                "ex+": "excellent plus",
                "ex-": "excellent minus",
                "ex": "excellent",
                "e+": "excellent plus",
                "e-": "excellent minus",
            },
            "min_text_length_discogs": 10,
            "unverified_media_signals": [
                "untested",
                "unplayed",
                "sold as seen",
                "haven't played",
                "not played",
                "unable to test",
                "no turntable",
            ],
            "description_adequacy": {
                "enabled": True,
                "drop_insufficient_from_training": False,
                "require_both_for_training": True,
                "min_chars_sleeve_fallback": 56,
                "user_prompt_sleeve": "Add sleeve detail.",
                "user_prompt_media": "Add media detail.",
                "sleeve_evidence_terms": ["jacket", "cover", "corner"],
            },
        },
        "rules": {
            "guidelines_path": guidelines_path,
            "confidence_threshold": 0.85,
            "flag_contradictions": True,
            "contradiction_action": "return_model_prediction",
        },
        "models": {
            "baseline": {
                "name": "tfidf_logreg",
                "tfidf": {
                    "max_features": 500,
                    "ngram_range": [1, 2],
                    "sublinear_tf": True,
                    "min_df": 1,
                },
                "logistic_regression": {
                    "C": 1.0,
                    "max_iter": 200,
                    "class_weight": "balanced",
                    "solver": "lbfgs",
                    "random_state": 42,
                },
            },
            "transformer": {
                "name": "distilbert_two_head",
                "base_model": "distilbert-base-uncased",
                "freeze_encoder": True,
                "max_length": 128,
                "dropout": 0.3,
                "learning_rate": 2.0e-4,
                "batch_size": 4,
                "epochs": 2,
                "early_stopping_patience": 1,
                "class_weight": "balanced",
                "random_state": 42,
            },
        },
        "evaluation": {
            "primary_metric": "macro_f1",
            "metrics": ["macro_f1", "accuracy", "calibration_error"],
            "calibration": {
                "method": "isotonic",
                "cv_folds": 2,
                "n_bins": 5,
            },
        },
        "mlflow": {
            "experiment_name": "test_vinyl_grader",
            "tracking_uri": (
                "sqlite:///" + tmp_dirs["mlflow_db"].resolve().as_posix()
            ),
            "tags": {"project": "vinyl_collector_ai", "module": "grader"},
        },
        "export": {
            "onnx_path": str(tmp_dirs["artifacts"] / "model.onnx"),
            "coreml_path": str(
                tmp_dirs["artifacts"] / "VinylGraderTransformer.mlpackage"
            ),
            "preprocessing_pipeline_path": str(
                tmp_dirs["artifacts"] / "preprocessor.pkl"
            ),
            "label_encoder_path": str(
                tmp_dirs["artifacts"] / "label_encoder.pkl"
            ),
            "tfidf_vectorizer_path": str(
                tmp_dirs["artifacts"] / "tfidf_vectorizer.pkl"
            ),
        },
        "inference": {"model": "baseline"},
    }

    config_path = tmp_dirs["artifacts"] / "test_grader.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    return str(config_path)


@pytest.fixture(scope="session")
def sample_unified_records():
    """
    80 synthetic records — 5 texts per grade per source.
    Enough for stratified splitting across 7-8 classes.
    """
    records = []
    record_id = 0
    for source in ["discogs", "ebay_jp"]:
        for sleeve_grade in SLEEVE_GRADES:
            media_grade = (
                "Near Mint"
                if sleeve_grade == "Generic"
                else (
                    sleeve_grade
                    if sleeve_grade in MEDIA_GRADES
                    else "Very Good Plus"
                )
            )
            texts = [GRADE_TEXTS[sleeve_grade]] + GRADE_TEXT_VARIANTS[
                sleeve_grade
            ]
            for text in texts:
                records.append(
                    {
                        "item_id": str(record_id),
                        "source": source,
                        "text": text,
                        "sleeve_label": sleeve_grade,
                        "media_label": media_grade,
                        "label_confidence": (
                            1.0 if source == "discogs" else 0.90
                        ),
                        "media_verifiable": sleeve_grade != "Mint",
                        "obi_condition": "VG+" if source == "ebay_jp" else None,
                        "raw_sleeve": sleeve_grade,
                        "raw_media": media_grade,
                        "artist": "Test Artist",
                        "title": f"Test Album {record_id}",
                        "year": 1975,
                        "country": "JP" if source == "ebay_jp" else "US",
                    }
                )
                record_id += 1
    return records


@pytest.fixture(scope="session")
def sample_discogs_listing():
    return {
        "id": "12345678",
        "condition": "Near Mint (NM or M-)",
        "sleeve_condition": "Very Good Plus (VG+)",
        "comments": "plays perfectly, very light scuff on cover only",
        "release": {
            "artist": "Miles Davis",
            "title": "Kind of Blue",
            "year": 1959,
            "country": "US",
        },
    }


@pytest.fixture(scope="session")
def sample_ebay_item_clean():
    return {
        "itemId": "987654321",
        "title": "Miles Davis Kind of Blue CBS Sony Japan OBI",
        "shortDescription": "OBI",
        "localizedAspects": [
            {"name": "Sleeve Grading", "value": "VG+"},
            {"name": "Record Grading", "value": "E-"},
            {"name": "OBI_Grading", "value": "VG+"},
            {"name": "Artist", "value": "Miles Davis"},
            {"name": "Country of Origin", "value": "Japan"},
        ],
    }


@pytest.fixture(scope="session")
def sample_ebay_item_annotated():
    return {
        "itemId": "111222333",
        "title": "Madonna Sire P-11394 Japan VINYL LP OBI",
        "shortDescription": "",
        "localizedAspects": [
            {
                "name": "Cover Condition",
                "value": "E (Excellent) S (Stain) cornerbump",
            },
            {"name": "Vinyl Condition", "value": "E+ (Excellent Plus)"},
            {"name": "OBI Condition", "value": "E (Excellent) OS (OBI Stain)"},
            {"name": "Artist", "value": "Madonna"},
            {"name": "Country of Origin", "value": "Japan"},
        ],
    }


@pytest.fixture(scope="session")
def fitted_encoders():
    encoders = {}
    for grades, target in [(SLEEVE_GRADES, "sleeve"), (MEDIA_GRADES, "media")]:
        enc = LabelEncoder()
        enc.fit(grades)
        encoders[target] = enc
    return encoders


@pytest.fixture(scope="session")
def fitted_vectorizers(sample_unified_records):
    texts = [r["text"] for r in sample_unified_records]
    vectorizers = {}
    for target in ["sleeve", "media"]:
        vec = TfidfVectorizer(max_features=200, ngram_range=(1, 2), min_df=1)
        vec.fit(texts)
        vectorizers[target] = vec
    return vectorizers


@pytest.fixture(scope="session")
def sample_feature_matrices(
    fitted_vectorizers, fitted_encoders, sample_unified_records
):
    texts = [r["text"] for r in sample_unified_records]
    n = len(texts)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    idx = {
        "train": list(range(0, train_end)),
        "val": list(range(train_end, val_end)),
        "test": list(range(val_end, n)),
    }

    features = {}
    for split, split_idx in idx.items():
        features[split] = {}
        split_texts = [texts[i] for i in split_idx]
        for target in ["sleeve", "media"]:
            labels = [
                sample_unified_records[i][f"{target}_label"] for i in split_idx
            ]
            X = fitted_vectorizers[target].transform(split_texts)
            y = fitted_encoders[target].transform(labels)
            features[split][target] = {"X": X, "y": y}
    return features


@pytest.fixture(scope="session")
def fitted_baseline(sample_feature_matrices, fitted_encoders):
    """
    LR models fitted on full training data.
    No multi_class parameter — removed in sklearn 1.5+.
    """
    models = {}
    for target in ["sleeve", "media"]:
        X_train = sample_feature_matrices["train"][target]["X"]
        y_train = sample_feature_matrices["train"][target]["y"]
        lr = LogisticRegression(
            max_iter=200,
            random_state=42,
            class_weight="balanced",
            solver="lbfgs",
        )
        lr.fit(X_train, y_train)
        models[target] = lr
    return models


@pytest.fixture(scope="session")
def fitted_calibrated_models(fitted_baseline):
    """
    For tests, use raw LR models as the 'calibrated' models.

    Rationale: CalibratedClassifierCV without cv= does internal
    cross-validation which may not see all classes in small test
    splits, causing IndexError when predicting unseen class indices.
    Raw LR is fitted on all classes and is sufficient for testing
    prediction schema and pipeline wiring. Calibration quality is
    tested separately in test_baseline.py::TestCalibration.
    """
    return fitted_baseline


@pytest.fixture(scope="session")
def sample_prediction():
    return {
        "item_id": "test_001",
        "predicted_sleeve_condition": "Very Good Plus",
        "predicted_media_condition": "Very Good Plus",
        "confidence_scores": {
            "sleeve": {
                "Mint": 0.02,
                "Near Mint": 0.10,
                "Excellent": 0.08,
                "Very Good Plus": 0.55,
                "Very Good": 0.15,
                "Good": 0.05,
                "Poor": 0.02,
                "Generic": 0.03,
            },
            "media": {
                "Mint": 0.03,
                "Near Mint": 0.12,
                "Excellent": 0.10,
                "Very Good Plus": 0.50,
                "Very Good": 0.15,
                "Good": 0.06,
                "Poor": 0.04,
            },
        },
        "metadata": {
            "source": "discogs",
            "media_verifiable": True,
            "rule_override_applied": False,
            "rule_override_target": None,
            "contradiction_detected": False,
        },
    }


@pytest.fixture(scope="session")
def sample_predictions(sample_prediction):
    predictions = []
    for i in range(5):
        pred = {**sample_prediction, "item_id": f"test_{i:03d}"}
        pred["confidence_scores"] = {
            "sleeve": dict(sample_prediction["confidence_scores"]["sleeve"]),
            "media": dict(sample_prediction["confidence_scores"]["media"]),
        }
        pred["metadata"] = dict(sample_prediction["metadata"])
        predictions.append(pred)
    return predictions


@pytest.fixture(scope="session")
def unified_jsonl_path(tmp_dirs, sample_unified_records):
    path = tmp_dirs["processed"] / "unified.jsonl"
    with open(path, "w") as f:
        for record in sample_unified_records:
            f.write(json.dumps(record) + "\n")
    return path


@pytest.fixture(scope="session")
def split_jsonl_paths(tmp_dirs, sample_unified_records):
    n = len(sample_unified_records)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    splits = {
        "train": sample_unified_records[:train_end],
        "val": sample_unified_records[train_end:val_end],
        "test": sample_unified_records[val_end:],
    }

    paths = {}
    for split_name, records in splits.items():
        path = tmp_dirs["splits"] / f"{split_name}.jsonl"
        with open(path, "w") as f:
            for record in records:
                f.write(
                    json.dumps(
                        {
                            **record,
                            "text_clean": record["text"].lower(),
                            "split": split_name,
                        }
                    )
                    + "\n"
                )
        paths[split_name] = path
    return paths


@pytest.fixture(scope="session")
def saved_encoder_paths(tmp_dirs, fitted_encoders):
    paths = {}
    for target, encoder in fitted_encoders.items():
        path = tmp_dirs["artifacts"] / f"label_encoder_{target}.pkl"
        with open(path, "wb") as f:
            pickle.dump(encoder, f)
        paths[target] = path
    return paths


@pytest.fixture(scope="session")
def saved_vectorizer_paths(tmp_dirs, fitted_vectorizers):
    paths = {}
    for target, vec in fitted_vectorizers.items():
        path = tmp_dirs["artifacts"] / f"tfidf_vectorizer_{target}.pkl"
        with open(path, "wb") as f:
            pickle.dump(vec, f)
        paths[target] = path
    return paths


@pytest.fixture(scope="session")
def saved_feature_paths(tmp_dirs, sample_feature_matrices):
    features_dir = tmp_dirs["artifacts"] / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    for split, target_data in sample_feature_matrices.items():
        for target, data in target_data.items():
            sp.save_npz(
                str(features_dir / f"{split}_{target}_X.npz"), data["X"]
            )
            np.save(str(features_dir / f"{split}_{target}_y.npy"), data["y"])
    return features_dir


@pytest.fixture(scope="session")
def saved_calibrated_model_paths(tmp_dirs, fitted_baseline):
    """Save raw LR models as both 'raw' and 'calibrated' artifacts."""
    for target in ["sleeve", "media"]:
        for suffix in ["", "_calibrated"]:
            path = tmp_dirs["artifacts"] / f"baseline_{target}{suffix}.pkl"
            with open(path, "wb") as f:
                pickle.dump(fitted_baseline[target], f)
    return tmp_dirs["artifacts"]
