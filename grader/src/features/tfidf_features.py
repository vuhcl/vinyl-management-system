"""
grader/src/features/tfidf_features.py

TF-IDF feature extraction for the vinyl condition grader baseline.

Fits separate TF-IDF vectorizers and label encoders for sleeve
and media targets on the train split only — strictly enforced to
prevent data leakage. Transforms all splits and saves sparse
feature matrices, fitted vectorizers, and label encoders to disk.

Output artifacts:
  grader/artifacts/tfidf_vectorizer_sleeve.pkl
  grader/artifacts/tfidf_vectorizer_media.pkl
  grader/artifacts/label_encoder_sleeve.pkl
  grader/artifacts/label_encoder_media.pkl
  grader/artifacts/features/{split}_{target}_X.npz  (sparse)
  grader/artifacts/features/{split}_{target}_y.npy  (dense int array)

Usage:
    python -m grader.src.features.tfidf_features
    python -m grader.src.features.tfidf_features --dry-run
"""

import copy
import json
import logging
import pickle
import re
from pathlib import Path
from typing import Any, Optional

import mlflow
import numpy as np
import scipy.sparse as sp
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

from grader.src.data.preprocess import (
    Preprocessor,
    build_protected_term_token_patterns,
)
from grader.src.mlflow_tracking import (
    mlflow_log_artifacts_enabled,
    mlflow_pipeline_step_run_ctx,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Targets — sleeve and media are always processed together
TARGETS = ["sleeve", "media"]


# ---------------------------------------------------------------------------
# TFIDFFeatureBuilder
# ---------------------------------------------------------------------------
class TFIDFFeatureBuilder:
    """
    Fits TF-IDF vectorizers and label encoders on the train split,
    transforms all splits, and saves artifacts for downstream use
    by the baseline model and evaluation pipeline.

    One vectorizer and one label encoder are created per target
    (sleeve, media) to keep feature pipelines fully independent.

    Config keys read from grader.yaml:
        models.baseline.tfidf.*     — vectorizer hyperparameters
        paths.splits                — train/val/test JSONL files
        paths.artifacts             — output directory for artifacts
        mlflow.tracking_uri
        mlflow.experiment_name
    """

    def __init__(
        self,
        config_path: str,
        config: Optional[dict[str, Any]] = None,
    ) -> None:
        if config is not None:
            self.config = copy.deepcopy(config)
        else:
            self.config = self._load_yaml(config_path)

        tfidf_cfg = self.config["models"]["baseline"]["tfidf"]
        self.max_features: int = tfidf_cfg["max_features"]
        self.ngram_range: tuple = tuple(tfidf_cfg["ngram_range"])
        self.sublinear_tf: bool = tfidf_cfg["sublinear_tf"]
        self.min_df: int = tfidf_cfg["min_df"]
        eng_cfg = self.config["models"]["baseline"].get(
            "engineered_features", {}
        )
        self.engineered_enabled: bool = bool(eng_cfg.get("enabled", False))

        # Paths
        splits_dir = Path(self.config["paths"]["splits"])
        artifacts_dir = Path(self.config["paths"]["artifacts"])
        self.features_dir = artifacts_dir / "features"

        self.split_paths = {
            "train": splits_dir / "train.jsonl",
            "val": splits_dir / "val.jsonl",
            "test": splits_dir / "test.jsonl",
            "test_thin": splits_dir / "test_thin.jsonl",
        }

        self.vectorizer_paths = {
            target: artifacts_dir / f"tfidf_vectorizer_{target}.pkl"
            for target in TARGETS
        }
        self.encoder_paths = {
            target: artifacts_dir / f"label_encoder_{target}.pkl"
            for target in TARGETS
        }

        artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.features_dir.mkdir(parents=True, exist_ok=True)

        # Fitted objects — populated during run()
        self.vectorizers: dict[str, TfidfVectorizer] = {}
        self.encoders: dict[str, LabelEncoder] = {}
        self._mild_cues: tuple[str, ...] = (
            "light",
            "slight",
            "tiny",
            "minor",
        )
        self._strong_cues: tuple[str, ...] = (
            "heavy",
            "significant",
            "many",
            "multiple",
            "deep",
            "foggy",
            "cloudy",
            "hazy",
            "warp",
            "noise",
            "split",
        )
        self._defect_terms: tuple[str, ...] = (
            "scratch",
            "scratches",
            "scuff",
            "scuffs",
            "hairline",
            "hairlines",
            "noise",
            "surface noise",
            "crackle",
            "warp",
            "split",
            "seam split",
            "ringwear",
            "ring wear",
            "foxing",
            "stain",
            "stains",
            "foggy",
            "cloudy",
            "haze",
            "hazy",
        )
        self._positive_negation_patterns: tuple[re.Pattern, ...] = (
            re.compile(r"\bno\s+skip(?:ping)?\b", re.IGNORECASE),
            re.compile(r"\bplays?\s+perfectly\b", re.IGNORECASE),
            re.compile(r"\bplays?\s+well\b", re.IGNORECASE),
            re.compile(r"\bplays?\s+through\b", re.IGNORECASE),
        )
        self._self_grade_patterns: dict[str, re.Pattern] = {
            "self_grade_mint": re.compile(
                r"(?<!\w)(mint|m-?|factory sealed|sealed)(?!\w)",
                re.IGNORECASE,
            ),
            "self_grade_nm": re.compile(
                r"(?<!\w)(near mint|nm|m-)(?!\w)",
                re.IGNORECASE,
            ),
            "self_grade_ex": re.compile(
                r"(?<!\w)(excellent|ex)(?!\w)",
                re.IGNORECASE,
            ),
            "self_grade_vgp": re.compile(
                r"(?<!\w)(very good plus|vg\+)\b",
                re.IGNORECASE,
            ),
            "self_grade_vg": re.compile(
                r"(?<!\w)(very good|vg)\b",
                re.IGNORECASE,
            ),
            "self_grade_g": re.compile(
                r"(?<!\w)(good plus|good|g\+|g)(?!\w)",
                re.IGNORECASE,
            ),
            "self_grade_close_vgp": re.compile(
                r"(?<!\w)("
                r"close\s+vg\+|"
                r"borderline\s+vg\+|"
                r"close\s+very\s+good\s+plus"
                r")(?!\w)",
                re.IGNORECASE,
            ),
            "self_grade_vgp_nm": re.compile(
                r"(?<!\w)(vg\+\s*/\s*nm|very good plus\s*/\s*near mint)(?!\w)",
                re.IGNORECASE,
            ),
        }

        # MLflow: ``run()`` uses ``mlflow_pipeline_step_run_ctx`` — configure only
        # when a nested step run is actually opened (``log_pipeline_step_runs``).

    # -----------------------------------------------------------------------
    # Config loading
    # -----------------------------------------------------------------------
    @staticmethod
    def _load_yaml(path: str) -> dict:
        with open(path, "r") as f:
            return yaml.safe_load(f)

    # -----------------------------------------------------------------------
    # Data loading
    # -----------------------------------------------------------------------
    def load_split(self, split: str) -> list[dict]:
        """Load a single split JSONL file."""
        path = self.split_paths[split]
        if not path.exists():
            raise FileNotFoundError(
                f"Split file not found: {path}. "
                "Run preprocess.py first."
            )
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        logger.info("Loaded %d records from %s split.", len(records), split)
        return records

    def load_test_thin_optional(self) -> list[dict]:
        """Thin-note eval split from preprocess; empty if missing or unused."""
        path = self.split_paths["test_thin"]
        if not path.exists():
            return []
        records: list[dict] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        if records:
            logger.info(
                "Loaded %d records from test_thin split.", len(records)
            )
        return records

    @staticmethod
    def remove_test_thin_feature_files(features_dir: Path) -> None:
        """Drop stale matrices when preprocess did not emit test_thin.jsonl."""
        for target in TARGETS:
            for suffix in (f"test_thin_{target}_X.npz", f"test_thin_{target}_y.npy"):
                p = features_dir / suffix
                if p.exists():
                    p.unlink()
                    logger.info("Removed stale feature file %s", p.name)

    def extract_texts(self, records: list[dict]) -> list[str]:
        """
        Extract text_clean field from records.
        Falls back to text field if text_clean is absent —
        handles edge cases where preprocess.py was not run.
        """
        pp_cfg = self.config.get("preprocessing", {})
        protected_term_patterns: dict[str, re.Pattern[str]] | None = None
        gp = self.config.get("rules", {}).get("guidelines_path")
        if gp:
            try:
                with open(gp, "r", encoding="utf-8") as gf:
                    guidelines = yaml.safe_load(gf)
                if isinstance(guidelines, dict):
                    protected_term_patterns = (
                        build_protected_term_token_patterns(guidelines)
                    )
            except OSError as exc:
                logger.warning(
                    "Could not load guidelines for TF-IDF promo gating (%s): %s",
                    gp,
                    exc,
                )
        texts = []
        for record in records:
            text = record.get("text_clean") or record.get("text", "")
            texts.append(
                Preprocessor.normalize_text_for_tfidf(
                    text,
                    preprocessing_cfg=pp_cfg,
                    protected_term_patterns=protected_term_patterns,
                )
            )
        return texts

    def extract_labels(
        self, records: list[dict], target: str
    ) -> list[str]:
        """Extract label strings for a given target (sleeve or media)."""
        field = f"{target}_label"
        return [record[field] for record in records]

    # -----------------------------------------------------------------------
    # Vectorizer — fit on train only
    # -----------------------------------------------------------------------
    def build_vectorizer(
        self, train_texts: list[str], target: str
    ) -> TfidfVectorizer:
        """
        Fit a TF-IDF vectorizer on train texts only.
        Fitting on any other split would constitute data leakage.

        Args:
            train_texts: texts from the train split only
            target:      "sleeve" or "media" — for logging clarity

        Returns:
            Fitted TfidfVectorizer.
        """
        logger.info(
            "Fitting TF-IDF vectorizer for target=%s on %d train samples ...",
            target,
            len(train_texts),
        )

        vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            sublinear_tf=self.sublinear_tf,
            min_df=self.min_df,
            strip_accents="unicode",
            analyzer="word",
            token_pattern=r"(?u)\b\w+\b",  # include single-char tokens
        )
        vectorizer.fit(train_texts)

        vocab_size = len(vectorizer.vocabulary_)
        logger.info(
            "Vectorizer fitted — target=%s | vocab size: %d",
            target,
            vocab_size,
        )
        return vectorizer

    def transform(
        self,
        vectorizer: TfidfVectorizer,
        texts: list[str],
        split: str,
        target: str,
    ) -> sp.csr_matrix:
        """
        Apply a fitted vectorizer to a list of texts.
        Returns a sparse CSR matrix.
        """
        logger.info(
            "Transforming %s split for target=%s (%d samples) ...",
            split,
            target,
            len(texts),
        )
        return vectorizer.transform(texts)

    @staticmethod
    def _count_any(text: str, terms: tuple[str, ...]) -> int:
        tl = text.lower()
        return sum(1 for t in terms if t in tl)

    def _engineered_feature_names(self, target: str) -> list[str]:
        names = [
            "cue_mild_count",
            "cue_strong_count",
            "defect_count",
            "positive_negation_count",
            "self_grade_mint",
            "self_grade_nm",
            "self_grade_ex",
            "self_grade_vgp",
            "self_grade_vg",
            "self_grade_g",
            "self_grade_close_vgp",
            "self_grade_vgp_nm",
            "media_evidence_none",
            "media_evidence_weak",
            "media_evidence_strong",
            "has_media_subject",
            "has_media_condition",
            "is_media_target",
            "is_sleeve_target",
        ]
        return names + [f"{target}_engineered_bias"]

    def _build_engineered_matrix(
        self,
        records: list[dict],
        target: str,
    ) -> sp.csr_matrix:
        if not records:
            return sp.csr_matrix((0, 0), dtype=np.float32)

        rows: list[list[float]] = []
        for record in records:
            text = (
                record.get("text_clean") or record.get("text") or ""
            ).lower()
            media_strength = str(
                record.get("media_evidence_strength", "none")
            ).lower()
            has_media_subject = float(
                any(
                    t in text for t in ("vinyl", "record", "disc", "lp", "wax")
                )
            )
            has_media_condition = float(
                any(
                    t in text
                    for t in (
                        "scratch",
                        "scuff",
                        "hairline",
                        "noise",
                        "warp",
                        "foggy",
                        "cloudy",
                    )
                )
            )
            pos_neg_count = float(
                sum(
                    1
                    for pat in self._positive_negation_patterns
                    if pat.search(text)
                )
            )
            features = [
                float(self._count_any(text, self._mild_cues)),
                float(self._count_any(text, self._strong_cues)),
                float(self._count_any(text, self._defect_terms)),
                pos_neg_count,
                float(
                    bool(self._self_grade_patterns["self_grade_mint"].search(text))
                ),
                float(
                    bool(self._self_grade_patterns["self_grade_nm"].search(text))
                ),
                float(
                    bool(self._self_grade_patterns["self_grade_ex"].search(text))
                ),
                float(
                    bool(self._self_grade_patterns["self_grade_vgp"].search(text))
                ),
                float(
                    bool(self._self_grade_patterns["self_grade_vg"].search(text))
                ),
                float(
                    bool(self._self_grade_patterns["self_grade_g"].search(text))
                ),
                float(
                    bool(
                        self._self_grade_patterns["self_grade_close_vgp"].search(
                            text
                        )
                    )
                ),
                float(
                    bool(
                        self._self_grade_patterns["self_grade_vgp_nm"].search(text)
                    )
                ),
                float(media_strength == "none"),
                float(media_strength == "weak"),
                float(media_strength == "strong"),
                has_media_subject,
                has_media_condition,
                float(target == "media"),
                float(target == "sleeve"),
                1.0,
            ]
            rows.append(features)
        return sp.csr_matrix(np.asarray(rows, dtype=np.float32))

    def transform_records(
        self,
        vectorizer: TfidfVectorizer,
        records: list[dict],
        target: str,
        split: str,
    ) -> sp.csr_matrix:
        texts = self.extract_texts(records)
        X_text = self.transform(vectorizer, texts, split, target)
        if not self.engineered_enabled:
            return X_text
        X_eng = self._build_engineered_matrix(records, target)
        if X_eng.shape[1] == 0:
            return X_text
        return sp.hstack([X_text, X_eng], format="csr")

    # -----------------------------------------------------------------------
    # Label encoder — canonical schema ∪ train labels
    # -----------------------------------------------------------------------
    def _guidelines_grade_list(self, target: str) -> list[str]:
        """All canonical grades for target from grading_guidelines.yaml."""
        gp = self.config.get(
            "guidelines_path",
            "grader/configs/grading_guidelines.yaml",
        )
        gl = self._load_yaml(gp)
        key = "sleeve_grades" if target == "sleeve" else "media_grades"
        if key not in gl:
            raise KeyError(f"{gp} missing '{key}' for label encoder")
        return [str(x) for x in gl[key]]

    def build_encoder(
        self, train_labels: list[str], target: str
    ) -> LabelEncoder:
        """
        Fit a LabelEncoder on the **full canonical grade list** from guidelines
        plus any train labels not in that list.

        Train-only fitting would drop rare grades (e.g. sleeve ``Excellent``
        when the train split is Discogs-heavy). The rule engine and eBay
        harmonization still emit those strings — encoders must cover them.

        After changing this, re-run feature extraction and retrain models so
        ``y`` indices and classifier heads match the expanded class set.

        Args:
            train_labels: grade strings from the train split only
            target:       "sleeve" or "media"

        Returns:
            Fitted LabelEncoder.
        """
        canonical = self._guidelines_grade_list(target)
        train_set = {str(l) for l in train_labels}
        extra = train_set - set(canonical)
        if extra:
            logger.warning(
                "Train labels not listed in guidelines %s_grades: %s",
                target,
                sorted(extra),
            )
        combined = sorted(set(canonical) | train_set)
        encoder = LabelEncoder()
        encoder.fit(combined)
        logger.info(
            "Label encoder fitted — target=%s | n_classes=%d | classes: %s",
            target,
            len(encoder.classes_),
            list(encoder.classes_),
        )
        return encoder

    # -----------------------------------------------------------------------
    # Artifact persistence
    # -----------------------------------------------------------------------
    def save_vectorizer(
        self, vectorizer: TfidfVectorizer, target: str
    ) -> None:
        path = self.vectorizer_paths[target]
        with open(path, "wb") as f:
            pickle.dump(vectorizer, f)
        logger.info("Saved vectorizer for target=%s to %s", target, path)

    def save_encoder(self, encoder: LabelEncoder, target: str) -> None:
        path = self.encoder_paths[target]
        with open(path, "wb") as f:
            pickle.dump(encoder, f)
        logger.info("Saved label encoder for target=%s to %s", target, path)

    def save_features(
        self,
        X: sp.csr_matrix,
        y: np.ndarray,
        split: str,
        target: str,
    ) -> None:
        """Save sparse feature matrix and label array for one split/target."""
        X_path = self.features_dir / f"{split}_{target}_X.npz"
        y_path = self.features_dir / f"{split}_{target}_y.npy"
        sp.save_npz(str(X_path), X)
        np.save(str(y_path), y)
        logger.debug(
            "Saved features — split=%s target=%s shape=%s",
            split,
            target,
            X.shape,
        )

    @staticmethod
    def load_vectorizer(path: str) -> TfidfVectorizer:
        with open(path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def load_encoder(path: str) -> LabelEncoder:
        with open(path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def load_features(
        features_dir: str, split: str, target: str
    ) -> tuple[sp.csr_matrix, np.ndarray]:
        """Load feature matrix and label array for a split/target pair."""
        features_dir = Path(features_dir)
        X = sp.load_npz(str(features_dir / f"{split}_{target}_X.npz"))
        y = np.load(str(features_dir / f"{split}_{target}_y.npy"))
        return X, y

    # -----------------------------------------------------------------------
    # Top terms per grade — for MLflow logging and interpretability
    # -----------------------------------------------------------------------
    def get_top_terms(
        self,
        vectorizer: TfidfVectorizer,
        encoder: LabelEncoder,
        X_train: sp.csr_matrix,
        y_train: np.ndarray,
        n_terms: int = 10,
    ) -> dict[str, list[str]]:
        """
        Compute the top N TF-IDF terms per grade class.
        Useful for verifying that the vectorizer has learned
        meaningful grading vocabulary.

        Method: for each class, compute mean TF-IDF weight across
        all samples of that class, return top N feature names.
        """
        feature_names = vectorizer.get_feature_names_out()
        top_terms: dict[str, list[str]] = {}

        for class_idx, class_label in enumerate(encoder.classes_):
            # Mask for samples belonging to this class
            mask = y_train == class_idx
            if not np.any(mask):
                top_terms[class_label] = []
                continue

            # Mean TF-IDF weight across class samples
            class_X = X_train[mask]
            mean_weights = np.asarray(class_X.mean(axis=0)).flatten()
            # If engineered features are appended, keep top-terms reporting
            # scoped to original TF-IDF vocabulary only.
            if mean_weights.shape[0] > len(feature_names):
                mean_weights = mean_weights[: len(feature_names)]

            # Top N indices by weight
            top_indices = mean_weights.argsort()[-n_terms:][::-1]
            top_terms[class_label] = [
                feature_names[i] for i in top_indices
            ]

        return top_terms

    def save_top_terms_report(
        self,
        top_terms_sleeve: dict[str, list[str]],
        top_terms_media: dict[str, list[str]],
    ) -> Path:
        """
        Write a human-readable top terms report to artifacts/.
        Logged to MLflow as an artifact.
        """
        report_path = (
            Path(self.config["paths"]["artifacts"]) / "top_tfidf_terms.txt"
        )
        lines = [
            "=" * 60,
            "TOP TF-IDF TERMS PER GRADE",
            "=" * 60,
            "",
            "SLEEVE TARGET",
            "-" * 40,
        ]
        for grade, terms in top_terms_sleeve.items():
            lines.append(f"  {grade:<20} {', '.join(terms)}")

        lines += [
            "",
            "MEDIA TARGET",
            "-" * 40,
        ]
        for grade, terms in top_terms_media.items():
            lines.append(f"  {grade:<20} {', '.join(terms)}")

        lines.append("")
        with open(report_path, "w") as f:
            f.write("\n".join(lines))

        logger.info("Top terms report saved to %s", report_path)
        return report_path

    # -----------------------------------------------------------------------
    # MLflow logging
    # -----------------------------------------------------------------------
    def _log_mlflow(
        self,
        split_sizes: dict[str, int],
        vocab_sizes: dict[str, int],
        top_terms_report_path: Path,
    ) -> None:
        mlflow.log_params(
            {
                "tfidf_max_features": self.max_features,
                "tfidf_ngram_range":  str(self.ngram_range),
                "tfidf_sublinear_tf": self.sublinear_tf,
                "tfidf_min_df":       self.min_df,
                "engineered_features_enabled": self.engineered_enabled,
            }
        )
        metrics_tf = {
            "n_train": split_sizes["train"],
            "n_val": split_sizes["val"],
            "n_test": split_sizes["test"],
            "vocab_size_sleeve": vocab_sizes["sleeve"],
            "vocab_size_media": vocab_sizes["media"],
        }
        if split_sizes.get("test_thin", 0):
            metrics_tf["n_test_thin"] = split_sizes["test_thin"]
        mlflow.log_metrics(metrics_tf)
        if mlflow_log_artifacts_enabled(self.config):
            mlflow.log_artifact(str(top_terms_report_path))
            for target in TARGETS:
                mlflow.log_artifact(str(self.vectorizer_paths[target]))
                mlflow.log_artifact(str(self.encoder_paths[target]))

    # -----------------------------------------------------------------------
    # Orchestration
    # -----------------------------------------------------------------------
    def run(self, dry_run: bool = False) -> dict:
        """
        Full TF-IDF feature extraction pipeline:
          1. Load train/val/test splits (and test_thin when present)
          2. For each target (sleeve, media):
             a. Fit vectorizer on train texts only
             b. Fit label encoder on train labels only
             c. Transform all splits
             d. Encode labels for all splits
             e. Save feature matrices and label arrays
             f. Save fitted vectorizer and encoder
          3. Generate top terms report
          4. Log to MLflow

        Args:
            dry_run: fit and transform but do not save artifacts
                     or log to MLflow.

        Returns:
            Dict with fitted vectorizers, encoders, and feature
            matrices for inspection or direct use by baseline.py.
        """
        with mlflow_pipeline_step_run_ctx(
            self.config, "tfidf_features"
        ) as mlf:

            # Load all splits
            split_data: dict[str, list[dict]] = {}
            for split in ["train", "val", "test"]:
                split_data[split] = self.load_split(split)

            thin_records = self.load_test_thin_optional()

            split_sizes = {
                split: len(records)
                for split, records in split_data.items()
            }
            split_sizes["test_thin"] = len(thin_records)

            # Extract texts per split — same text regardless of target
            split_texts: dict[str, list[str]] = {
                split: self.extract_texts(records)
                for split, records in split_data.items()
            }

            feature_split_keys = ["train", "val", "test"]
            if thin_records:
                feature_split_keys.append("test_thin")

            results: dict = {
                "vectorizers": {},
                "encoders":    {},
                "features": {split: {} for split in feature_split_keys},
            }
            vocab_sizes: dict[str, int] = {}

            for target in TARGETS:
                logger.info("--- Processing target: %s ---", target.upper())

                # Train labels for fitting encoder
                train_labels = self.extract_labels(
                    split_data["train"], target
                )

                # Fit vectorizer and encoder on train only
                vectorizer = self.build_vectorizer(
                    split_texts["train"], target
                )
                encoder = self.build_encoder(train_labels, target)

                self.vectorizers[target] = vectorizer
                self.encoders[target]    = encoder
                vocab_sizes[target]      = len(vectorizer.vocabulary_)

                # Transform all splits and encode labels
                for split in ["train", "val", "test"]:
                    labels = self.extract_labels(split_data[split], target)
                    X = self.transform_records(
                        vectorizer=vectorizer,
                        records=split_data[split],
                        target=target,
                        split=split,
                    )
                    y = encoder.transform(labels)

                    results["features"][split][target] = {"X": X, "y": y}

                    if not dry_run:
                        self.save_features(X, y, split, target)

                if thin_records:
                    labels_thin = self.extract_labels(thin_records, target)
                    X_thin = self.transform_records(
                        vectorizer=vectorizer,
                        records=thin_records,
                        target=target,
                        split="test_thin",
                    )
                    y_thin = encoder.transform(labels_thin)
                    results["features"]["test_thin"][target] = {
                        "X": X_thin,
                        "y": y_thin,
                    }
                    if not dry_run:
                        self.save_features(X_thin, y_thin, "test_thin", target)

                if not dry_run:
                    self.save_vectorizer(vectorizer, target)
                    self.save_encoder(encoder, target)

            # Top terms report
            train_sleeve = results["features"]["train"]["sleeve"]
            train_media  = results["features"]["train"]["media"]

            top_terms_sleeve = self.get_top_terms(
                self.vectorizers["sleeve"],
                self.encoders["sleeve"],
                train_sleeve["X"],
                train_sleeve["y"],
            )
            top_terms_media = self.get_top_terms(
                self.vectorizers["media"],
                self.encoders["media"],
                train_media["X"],
                train_media["y"],
            )

            # Log top terms to console regardless of dry_run
            logger.info("Top TF-IDF terms per grade (sleeve):")
            for grade, terms in top_terms_sleeve.items():
                logger.info("  %-20s %s", grade, ", ".join(terms))

            logger.info("Top TF-IDF terms per grade (media):")
            for grade, terms in top_terms_media.items():
                logger.info("  %-20s %s", grade, ", ".join(terms))

            if dry_run:
                logger.info(
                    "Dry run — skipping artifact saves and MLflow logging."
                )
                results["vectorizers"] = self.vectorizers
                results["encoders"]    = self.encoders
                return results

            if not thin_records:
                self.remove_test_thin_feature_files(self.features_dir)

            top_terms_path = self.save_top_terms_report(
                top_terms_sleeve, top_terms_media
            )
            if mlf:
                self._log_mlflow(split_sizes, vocab_sizes, top_terms_path)

            results["vectorizers"] = self.vectorizers
            results["encoders"]    = self.encoders
            return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Build TF-IDF features for vinyl grader baseline"
    )
    parser.add_argument(
        "--config",
        default="grader/configs/grader.yaml",
        help="Path to grader config YAML",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fit and transform without saving artifacts",
    )
    args = parser.parse_args()

    builder = TFIDFFeatureBuilder(config_path=args.config)
    builder.run(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
