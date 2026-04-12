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

import json
import logging
import pickle
from pathlib import Path

import mlflow
import numpy as np
import scipy.sparse as sp
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

from grader.src.data.preprocess import strip_stray_numeric_tokens_from_text
from grader.src.mlflow_tracking import (
    configure_mlflow_from_config,
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
        mlflow (URI from MLFLOW_TRACKING_URI / tracking_uri_fallback)
        mlflow.experiment_name
    """

    def __init__(self, config_path: str) -> None:
        self.config = self._load_yaml(config_path)

        tfidf_cfg = self.config["models"]["baseline"]["tfidf"]
        self.max_features: int        = tfidf_cfg["max_features"]
        self.ngram_range: tuple       = tuple(tfidf_cfg["ngram_range"])
        self.sublinear_tf: bool       = tfidf_cfg["sublinear_tf"]
        self.min_df: int              = tfidf_cfg["min_df"]

        # Paths
        splits_dir        = Path(self.config["paths"]["splits"])
        artifacts_dir     = Path(self.config["paths"]["artifacts"])
        self.features_dir = artifacts_dir / "features"

        self.split_paths = {
            "train": splits_dir / "train.jsonl",
            "val":   splits_dir / "val.jsonl",
            "test":  splits_dir / "test.jsonl",
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

        pp_cfg = self.config.get("preprocessing") or {}
        self._strip_stray_numeric_tokens: bool = pp_cfg.get(
            "strip_stray_numeric_tokens", True
        )
        self._normalize_whitespace_for_strip: bool = pp_cfg.get(
            "normalize_whitespace", True
        )

        # Fitted objects — populated during run()
        self.vectorizers: dict[str, TfidfVectorizer] = {}
        self.encoders: dict[str, LabelEncoder]       = {}

        # MLflow — resolve tracking URI (env / fallback / legacy key)
        configure_mlflow_from_config(self.config)

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

    def extract_texts(self, records: list[dict]) -> list[str]:
        """
        Extract text_clean field from records.
        Falls back to ``text`` if ``text_clean`` is missing or blank —
        in that case other cleaning may be absent, so we re-apply stray
        digit stripping here (same rule as ``Preprocessor``) so TF-IDF
        top terms are not polluted by boilerplate ``6`` tokens.
        """
        texts = []
        for record in records:
            raw = record.get("text_clean")
            if raw is not None and str(raw).strip() != "":
                text = str(raw)
            else:
                text = record.get("text", "") or ""
            if self._strip_stray_numeric_tokens:
                text = strip_stray_numeric_tokens_from_text(
                    text,
                    normalize_whitespace=self._normalize_whitespace_for_strip,
                )
            texts.append(text)
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

    def transform_records(
        self,
        *,
        vectorizer: TfidfVectorizer,
        records: list[dict],
        target: str,
        split: str,
    ) -> sp.csr_matrix:
        """
        Vectorize JSONL-shaped ``records`` (``text_clean`` / ``text``) for
        inference or offline evaluation. ``split`` is for logging only.
        """
        texts = self.extract_texts(records)
        return self.transform(vectorizer, texts, split=split, target=target)

    # -----------------------------------------------------------------------
    # Label encoder — fit on train only
    # -----------------------------------------------------------------------
    def build_encoder(
        self, train_labels: list[str], target: str
    ) -> LabelEncoder:
        """
        Fit a LabelEncoder on train labels only.
        Encodes canonical grade strings to integer indices.

        Args:
            train_labels: grade strings from the train split only
            target:       "sleeve" or "media"

        Returns:
            Fitted LabelEncoder.
        """
        encoder = LabelEncoder()
        encoder.fit(train_labels)
        logger.info(
            "Label encoder fitted — target=%s | classes: %s",
            target,
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
        report_path = Path(self.config["paths"]["artifacts"]) / "top_tfidf_terms.txt"
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
            }
        )
        mlflow.log_metrics(
            {
                "n_train":                split_sizes["train"],
                "n_val":                  split_sizes["val"],
                "n_test":                 split_sizes["test"],
                "vocab_size_sleeve":      vocab_sizes["sleeve"],
                "vocab_size_media":       vocab_sizes["media"],
            }
        )
        mlflow.log_artifact(str(top_terms_report_path))

        # Log fitted vectorizer and encoder artifacts
        for target in TARGETS:
            mlflow.log_artifact(str(self.vectorizer_paths[target]))
            mlflow.log_artifact(str(self.encoder_paths[target]))

    # -----------------------------------------------------------------------
    # Orchestration
    # -----------------------------------------------------------------------
    def run(self, dry_run: bool = False) -> dict:
        """
        Full TF-IDF feature extraction pipeline:
          1. Load train/val/test splits
          2. For each target (sleeve, media):
             a. Fit vectorizer on train texts only
             b. Fit label encoder on train labels only
             c. Transform all splits
             d. Encode labels for all splits
             e. Save feature matrices and label arrays
             f. Save fitted vectorizer and encoder
          3. Generate top terms report
          4. Optionally log to MLflow (``mlflow.log_pipeline_step_runs``)

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

            split_sizes = {
                split: len(records)
                for split, records in split_data.items()
            }

            # Extract texts per split — same text regardless of target
            split_texts: dict[str, list[str]] = {
                split: self.extract_texts(records)
                for split, records in split_data.items()
            }

            results: dict = {
                "vectorizers": {},
                "encoders":    {},
                "features":    {split: {} for split in ["train", "val", "test"]},
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
                    texts  = split_texts[split]
                    labels = self.extract_labels(split_data[split], target)

                    X = self.transform(vectorizer, texts, split, target)
                    y = encoder.transform(labels)

                    results["features"][split][target] = {"X": X, "y": y}

                    if not dry_run:
                        self.save_features(X, y, split, target)

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
