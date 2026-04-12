"""
grader/src/export/coreml_export.py

Exports trained vinyl condition grader models to CoreML format
for on-device inference in the iOS app.

Exports:
  VinylGraderBaseline.mlpackage    — TF-IDF + LR, text input
  VinylGraderTransformer.mlpackage — DistilBERT two-head, token ID input
  tokenizer/                       — vocab.txt, tokenizer_config.json
  label_map.json                   — integer → grade string mapping

Tokenization strategy:
  Transformer takes raw token ID tensors as input.
  The Swift app performs WordPiece tokenization using the bundled
  vocab.txt before calling the CoreML model. This keeps the app
  fully offline and tokenization identical to training.

Validation:
  After each export, a round-trip test compares Python model output
  to CoreML model output on a set of test sentences. Predictions
  must match within floating point tolerance.

Usage:
    python -m grader.src.export.coreml_export
    python -m grader.src.export.coreml_export --model baseline
    python -m grader.src.export.coreml_export --model transformer
    python -m grader.src.export.coreml_export --skip-validation
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Optional

import coremltools as ct
import mlflow
import numpy as np
import torch
import yaml
from transformers import DistilBertTokenizerFast

from grader.src.data.preprocess import Preprocessor
from grader.src.features.tfidf_features import TFIDFFeatureBuilder
from grader.src.models.baseline import BaselineModel
from grader.src.mlflow_tracking import configure_mlflow_from_config
from grader.src.models.transformer import TwoHeadClassifier

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Validation test sentences — cover a range of grades and edge cases
VALIDATION_SENTENCES = [
    "factory sealed, still in shrink",
    "never played, no marks, near mint condition",
    "plays perfectly, very light scuff on sleeve",
    "surface noise on quiet passages, visible scratches",
    "heavy wear, crackling throughout, seam split at bottom",
    "generic white sleeve, record plays fine",
    "slight scuff on cover, excellent condition",
    "unplayed, mint minus",
    "good condition, plays through, some crackle",
    "badly warped, skipping on side two",
]


# ---------------------------------------------------------------------------
# CoreMLExporter
# ---------------------------------------------------------------------------
class CoreMLExporter:
    """
    Exports trained models to CoreML format for iOS deployment.

    Handles both baseline (TF-IDF + LR) and transformer (DistilBERT)
    export paths, tokenizer bundling, label map generation, and
    round-trip validation.

    Config keys read from grader.yaml:
        export.coreml_path              — transformer output path
        export.preprocessing_pipeline_path
        export.label_encoder_path
        paths.artifacts                 — artifact directory
        models.transformer.*            — transformer architecture config
        mlflow (URI from MLFLOW_TRACKING_URI / tracking_uri_fallback)
        mlflow.experiment_name
    """

    def __init__(self, config_path: str) -> None:
        self.config = self._load_yaml(config_path)
        self.config_path = config_path

        # Paths
        self.artifacts_dir = Path(self.config["paths"]["artifacts"])
        export_cfg         = self.config.get("export", {})

        self.baseline_coreml_path = (
            self.artifacts_dir / "VinylGraderBaseline.mlpackage"
        )
        self.transformer_coreml_path = Path(
            export_cfg.get(
                "coreml_path",
                str(self.artifacts_dir / "VinylGraderTransformer.mlpackage"),
            )
        )
        self.tokenizer_export_dir = self.artifacts_dir / "tokenizer"
        self.label_map_path       = self.artifacts_dir / "label_map.json"

        self.tokenizer_export_dir.mkdir(parents=True, exist_ok=True)

        # Transformer config
        t_cfg = self.config["models"]["transformer"]
        self.base_model  = t_cfg["base_model"]
        self.max_length  = t_cfg["max_length"]
        self.dropout     = t_cfg["dropout"]

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
    # Label encoder loading
    # -----------------------------------------------------------------------
    def _load_encoders(self) -> dict:
        encoders = {}
        for target in ["sleeve", "media"]:
            path = self.artifacts_dir / f"label_encoder_{target}.pkl"
            with open(path, "rb") as f:
                encoders[target] = pickle.load(f)
        return encoders

    # -----------------------------------------------------------------------
    # Label map export
    # -----------------------------------------------------------------------
    def export_label_map(self, encoders: dict) -> Path:
        """
        Export integer → grade string mapping for both targets.
        Used by the Swift app to decode CoreML integer outputs.

        Format:
            {
                "sleeve": {"0": "Good", "1": "Excellent", ...},
                "media":  {"0": "Good", ...}
            }

        Note: LabelEncoder sorts classes alphabetically, so index 0
        is not necessarily "Mint". The label map encodes the exact
        mapping used at training time.
        """
        label_map = {}
        for target, encoder in encoders.items():
            label_map[target] = {
                str(i): cls
                for i, cls in enumerate(encoder.classes_)
            }

        with open(self.label_map_path, "w") as f:
            json.dump(label_map, f, indent=2)

        logger.info("Label map saved to %s", self.label_map_path)
        logger.info("Sleeve class order: %s", list(encoders["sleeve"].classes_))
        logger.info("Media class order:  %s", list(encoders["media"].classes_))

        return self.label_map_path

    # -----------------------------------------------------------------------
    # Tokenizer bundling
    # -----------------------------------------------------------------------
    def export_tokenizer(self) -> Path:
        """
        Save tokenizer vocab and config for on-device Swift tokenization.

        Files saved:
          tokenizer/vocab.txt             — WordPiece vocabulary (30,522 tokens)
          tokenizer/tokenizer_config.json — special tokens, settings
          tokenizer/tokenizer.json        — full tokenizer spec (HuggingFace)

        The Swift app uses vocab.txt and tokenizer_config.json to
        implement WordPiece tokenization matching the training tokenizer.

        Returns:
            Path to tokenizer directory.
        """
        tokenizer = DistilBertTokenizerFast.from_pretrained(self.base_model)
        tokenizer.save_pretrained(str(self.tokenizer_export_dir))

        # Write a Swift-friendly summary of key tokenizer settings
        swift_config = {
            "model_max_length":    self.max_length,
            "do_lower_case":       True,
            "pad_token_id":        tokenizer.pad_token_id,
            "cls_token_id":        tokenizer.cls_token_id,
            "sep_token_id":        tokenizer.sep_token_id,
            "unk_token_id":        tokenizer.unk_token_id,
            "mask_token_id":       tokenizer.mask_token_id,
            "padding_side":        "right",
            "vocab_size":          tokenizer.vocab_size,
            "notes": (
                "Use vocab.txt with WordPiece tokenization. "
                "Add [CLS] at start, [SEP] at end. "
                "Pad or truncate to model_max_length. "
                "All text should be lowercased before tokenization."
            ),
        }

        swift_config_path = self.tokenizer_export_dir / "swift_tokenizer_config.json"
        with open(swift_config_path, "w") as f:
            json.dump(swift_config, f, indent=2)

        logger.info(
            "Tokenizer exported to %s (vocab size: %d)",
            self.tokenizer_export_dir,
            tokenizer.vocab_size,
        )

        return self.tokenizer_export_dir

    # -----------------------------------------------------------------------
    # Baseline CoreML export
    # -----------------------------------------------------------------------
    def export_baseline(self, encoders: dict) -> Path:
        """
        Export TF-IDF + Logistic Regression baseline to CoreML.

        Creates a CoreML Pipeline model with:
          - Input: String (seller notes)
          - Output: sleeve_grade (String), sleeve_probs (MultiArray)
                    media_grade  (String), media_probs  (MultiArray)

        Uses coremltools sklearn converter for TF-IDF and LR components.
        Both heads share the same text input but use independent
        vectorizers and classifiers.

        Returns:
            Path to saved .mlpackage.
        """
        logger.info("Exporting baseline model to CoreML ...")

        # Load fitted vectorizers and calibrated models
        sleeve_vectorizer = TFIDFFeatureBuilder.load_vectorizer(
            str(self.artifacts_dir / "tfidf_vectorizer_sleeve.pkl")
        )
        media_vectorizer = TFIDFFeatureBuilder.load_vectorizer(
            str(self.artifacts_dir / "tfidf_vectorizer_media.pkl")
        )
        sleeve_model = BaselineModel.load_model(
            str(self.artifacts_dir / "baseline_sleeve_calibrated.pkl")
        )
        media_model = BaselineModel.load_model(
            str(self.artifacts_dir / "baseline_media_calibrated.pkl")
        )

        sleeve_classes = list(encoders["sleeve"].classes_)
        media_classes  = list(encoders["media"].classes_)

        # Define a wrapper class that CoreML can trace
        # Combines vectorization + prediction into a single callable
        class BaselinePipeline:
            def __init__(self, vectorizer, classifier):
                self.vectorizer  = vectorizer
                self.classifier  = classifier

            def predict(self, text: str) -> tuple[str, np.ndarray]:
                X      = self.vectorizer.transform([text])
                grade  = self.classifier.predict(X)[0]
                probas = self.classifier.predict_proba(X)[0]
                return grade, probas.astype(np.float32)

        sleeve_pipeline = BaselinePipeline(sleeve_vectorizer, sleeve_model)
        media_pipeline  = BaselinePipeline(media_vectorizer,  media_model)

        # Convert sleeve pipeline
        sleeve_coreml = ct.converters.sklearn.convert(
            spec=[sleeve_vectorizer, sleeve_model],
            input_features=[("text", ct.converters.sklearn.STR_FEATURE)],
            output_feature_names=["sleeve_grade", "sleeve_probs"],
        )

        # Convert media pipeline
        media_coreml = ct.converters.sklearn.convert(
            spec=[media_vectorizer, media_model],
            input_features=[("text", ct.converters.sklearn.STR_FEATURE)],
            output_feature_names=["media_grade", "media_probs"],
        )

        # Update model metadata
        for model, target, classes in [
            (sleeve_coreml, "sleeve", sleeve_classes),
            (media_coreml,  "media",  media_classes),
        ]:
            model.short_description = (
                f"Vinyl condition grader — {target} target (baseline)"
            )
            spec = model.get_spec()
            spec.description.metadata.author     = "vinyl_collector_ai"
            spec.description.metadata.shortDescription = (
                f"TF-IDF + Logistic Regression baseline — {target}"
            )
            spec.description.metadata.versionString = "1.0.0"

        # Save both models
        sleeve_path = self.artifacts_dir / "VinylGraderBaseline_Sleeve.mlpackage"
        media_path  = self.artifacts_dir / "VinylGraderBaseline_Media.mlpackage"

        sleeve_coreml.save(str(sleeve_path))
        media_coreml.save(str(media_path))

        logger.info(
            "Baseline CoreML models saved:\n  %s\n  %s",
            sleeve_path,
            media_path,
        )

        # Return sleeve path as primary (media path logged separately)
        return sleeve_path

    # -----------------------------------------------------------------------
    # Transformer CoreML export
    # -----------------------------------------------------------------------
    def export_transformer(self, encoders: dict) -> Path:
        """
        Export DistilBERT two-head classifier to CoreML.

        Input:  input_ids      (Int32 MultiArray, shape [1, max_length])
                attention_mask (Int32 MultiArray, shape [1, max_length])
        Output: sleeve_logits  (Float32 MultiArray, shape [1, n_sleeve])
                media_logits   (Float32 MultiArray, shape [1, n_media])

        The Swift app:
          1. Tokenizes text using bundled vocab.txt
          2. Passes token ID tensors to this model
          3. Applies softmax to logits
          4. Uses label_map.json to decode argmax to grade string

        Returns:
            Path to saved .mlpackage.
        """
        logger.info("Exporting transformer model to CoreML ...")

        n_sleeve = len(encoders["sleeve"].classes_)
        n_media  = len(encoders["media"].classes_)

        # Load trained model weights
        model = TwoHeadClassifier(
            n_sleeve_classes=n_sleeve,
            n_media_classes=n_media,
            dropout=0.0,           # disable dropout at export time
            freeze_encoder=True,
            base_model=self.base_model,
        )

        weights_path = self.artifacts_dir / "transformer_weights.pt"
        model.load_state_dict(
            torch.load(weights_path, map_location="cpu")
        )
        model.eval()

        logger.info("Transformer weights loaded from %s", weights_path)

        # Create dummy inputs for tracing
        # Shape: [batch_size=1, seq_len=max_length]
        dummy_input_ids = torch.zeros(
            (1, self.max_length), dtype=torch.long
        )
        dummy_attention_mask = torch.ones(
            (1, self.max_length), dtype=torch.long
        )

        # Trace the model — captures the computation graph
        with torch.no_grad():
            traced_model = torch.jit.trace(
                model,
                (dummy_input_ids, dummy_attention_mask),
            )

        logger.info("Model traced successfully.")

        # Convert traced model to CoreML
        coreml_model = ct.convert(
            traced_model,
            inputs=[
                ct.TensorType(
                    name="input_ids",
                    shape=(1, self.max_length),
                    dtype=np.int32,
                ),
                ct.TensorType(
                    name="attention_mask",
                    shape=(1, self.max_length),
                    dtype=np.int32,
                ),
            ],
            outputs=[
                ct.TensorType(name="sleeve_logits", dtype=np.float32),
                ct.TensorType(name="media_logits",  dtype=np.float32),
            ],
            minimum_deployment_target=ct.target.iOS16,
            compute_units=ct.ComputeUnit.ALL,  # use ANE + GPU + CPU
        )

        # Add metadata
        coreml_model.short_description = (
            "Vinyl condition grader — DistilBERT two-head classifier"
        )
        spec = coreml_model.get_spec()
        spec.description.metadata.author     = "vinyl_collector_ai"
        spec.description.metadata.shortDescription = (
            "DistilBERT encoder with sleeve and media classification heads. "
            "Input: tokenized seller notes. "
            "Output: logits for sleeve and media condition grades. "
            "Apply softmax then argmax, decode with label_map.json."
        )
        spec.description.metadata.versionString = "1.0.0"
        spec.description.metadata.userDefined.update(
            {
                "max_length":      str(self.max_length),
                "n_sleeve_classes": str(n_sleeve),
                "n_media_classes":  str(n_media),
                "base_model":      self.base_model,
                "label_map_file":  "label_map.json",
                "tokenizer_dir":   "tokenizer/",
            }
        )

        coreml_model.save(str(self.transformer_coreml_path))
        logger.info(
            "Transformer CoreML model saved to %s",
            self.transformer_coreml_path,
        )

        return self.transformer_coreml_path

    # -----------------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------------
    def validate_transformer(
        self,
        encoders: dict,
        n_sentences: int = 5,
    ) -> bool:
        """
        Validate transformer CoreML export by comparing Python model
        output to CoreML model output on test sentences.

        Tokenizes sentences using the HuggingFace tokenizer (same as
        training), passes token IDs to both models, and asserts that
        predicted grade indices match.

        Args:
            encoders:    fitted label encoders
            n_sentences: number of validation sentences to test

        Returns:
            True if all predictions match, False otherwise.
        """
        logger.info("Validating transformer CoreML export ...")

        tokenizer = DistilBertTokenizerFast.from_pretrained(self.base_model)
        n_sleeve  = len(encoders["sleeve"].classes_)
        n_media   = len(encoders["media"].classes_)

        # Load PyTorch model
        pytorch_model = TwoHeadClassifier(
            n_sleeve_classes=n_sleeve,
            n_media_classes=n_media,
            dropout=0.0,
            freeze_encoder=True,
            base_model=self.base_model,
        )
        pytorch_model.load_state_dict(
            torch.load(
                self.artifacts_dir / "transformer_weights.pt",
                map_location="cpu",
            )
        )
        pytorch_model.eval()

        # Load CoreML model
        coreml_model = ct.models.MLModel(
            str(self.transformer_coreml_path)
        )

        sentences = VALIDATION_SENTENCES[:n_sentences]
        all_match = True

        for sentence in sentences:
            # Tokenize
            encoding = tokenizer(
                sentence,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids      = encoding["input_ids"]
            attention_mask = encoding["attention_mask"]

            # PyTorch prediction
            with torch.no_grad():
                sleeve_logits, media_logits = pytorch_model(
                    input_ids, attention_mask
                )
            pt_sleeve_pred = sleeve_logits.argmax(dim=-1).item()
            pt_media_pred  = media_logits.argmax(dim=-1).item()

            # CoreML prediction
            coreml_output = coreml_model.predict(
                {
                    "input_ids":      input_ids.numpy().astype(np.int32),
                    "attention_mask": attention_mask.numpy().astype(np.int32),
                }
            )
            cm_sleeve_pred = np.argmax(coreml_output["sleeve_logits"])
            cm_media_pred  = np.argmax(coreml_output["media_logits"])

            match = (
                pt_sleeve_pred == cm_sleeve_pred
                and pt_media_pred == cm_media_pred
            )

            sleeve_label = encoders["sleeve"].classes_[pt_sleeve_pred]
            media_label  = encoders["media"].classes_[pt_media_pred]

            status = "✓" if match else "✗"
            logger.info(
                "%s  %-45s  sleeve: %-18s  media: %-18s",
                status,
                f'"{sentence[:43]}"',
                sleeve_label,
                media_label,
            )

            if not match:
                all_match = False
                logger.error(
                    "MISMATCH — PyTorch: sleeve=%d media=%d | "
                    "CoreML: sleeve=%d media=%d",
                    pt_sleeve_pred, pt_media_pred,
                    cm_sleeve_pred, cm_media_pred,
                )

        if all_match:
            logger.info(
                "Transformer validation passed — "
                "all %d sentences match.", n_sentences
            )
        else:
            logger.error(
                "Transformer validation FAILED — "
                "CoreML output does not match PyTorch."
            )

        return all_match

    def validate_baseline(
        self,
        encoders: dict,
        n_sentences: int = 5,
    ) -> bool:
        """
        Validate baseline CoreML export by comparing sklearn model
        output to CoreML model output on test sentences.

        Args:
            encoders:    fitted label encoders
            n_sentences: number of validation sentences to test

        Returns:
            True if all predictions match, False otherwise.
        """
        logger.info("Validating baseline CoreML export ...")

        sleeve_vectorizer = TFIDFFeatureBuilder.load_vectorizer(
            str(self.artifacts_dir / "tfidf_vectorizer_sleeve.pkl")
        )
        sleeve_model = BaselineModel.load_model(
            str(self.artifacts_dir / "baseline_sleeve_calibrated.pkl")
        )

        sleeve_coreml = ct.models.MLModel(
            str(self.artifacts_dir / "VinylGraderBaseline_Sleeve.mlpackage")
        )

        sentences = VALIDATION_SENTENCES[:n_sentences]
        all_match = True

        for sentence in sentences:
            # sklearn prediction
            X = sleeve_vectorizer.transform([sentence])
            sk_pred    = sleeve_model.predict(X)[0]

            # CoreML prediction
            cm_output  = sleeve_coreml.predict({"text": sentence})
            cm_pred    = cm_output.get("sleeve_grade", "")

            # Decode sk_pred integer to string if needed
            if isinstance(sk_pred, (int, np.integer)):
                sk_pred = encoders["sleeve"].classes_[sk_pred]

            match  = sk_pred == cm_pred
            status = "✓" if match else "✗"

            logger.info(
                "%s  %-45s  sklearn: %-18s  CoreML: %-18s",
                status,
                f'"{sentence[:43]}"',
                sk_pred,
                cm_pred,
            )

            if not match:
                all_match = False

        if all_match:
            logger.info(
                "Baseline validation passed — "
                "all %d sentences match.", n_sentences
            )
        else:
            logger.error("Baseline validation FAILED.")

        return all_match

    # -----------------------------------------------------------------------
    # MLflow logging
    # -----------------------------------------------------------------------
    def _log_mlflow(
        self,
        exported_models: list[str],
        validation_results: dict[str, bool],
    ) -> None:
        mlflow.log_params(
            {
                "exported_models":  exported_models,
                "max_length":       self.max_length,
                "base_model":       self.base_model,
                "coreml_min_target": "iOS16",
            }
        )
        for model_name, passed in validation_results.items():
            mlflow.log_metric(
                f"validation_passed_{model_name}",
                1.0 if passed else 0.0,
            )

        # Log CoreML artifacts
        if "transformer" in exported_models:
            mlflow.log_artifact(str(self.transformer_coreml_path))
        if "baseline" in exported_models:
            mlflow.log_artifact(
                str(self.artifacts_dir / "VinylGraderBaseline_Sleeve.mlpackage")
            )
        mlflow.log_artifact(str(self.label_map_path))
        mlflow.log_artifact(str(self.tokenizer_export_dir))

    # -----------------------------------------------------------------------
    # Orchestration
    # -----------------------------------------------------------------------
    def run(
        self,
        export_baseline: bool = True,
        export_transformer: bool = True,
        skip_validation: bool = False,
    ) -> dict:
        """
        Export models to CoreML and validate outputs.

        Steps:
          1. Load label encoders
          2. Export label map (used by Swift app to decode predictions)
          3. Export tokenizer vocabulary bundle
          4. Export baseline model(s) to CoreML
          5. Export transformer model to CoreML
          6. Validate each exported model
          7. Log artifacts and results to MLflow

        Args:
            export_baseline:     export TF-IDF + LR baseline
            export_transformer:  export DistilBERT transformer
            skip_validation:     skip round-trip validation step

        Returns:
            Dict with export paths and validation results.
        """
        results: dict = {
            "exported":   [],
            "validated":  {},
            "paths":      {},
        }

        with mlflow.start_run(run_name="coreml_export"):

            # Step 1 — Load encoders
            encoders = self._load_encoders()

            # Step 2 — Export label map
            label_map_path = self.export_label_map(encoders)
            results["paths"]["label_map"] = str(label_map_path)

            # Step 3 — Export tokenizer
            if export_transformer:
                tokenizer_path = self.export_tokenizer()
                results["paths"]["tokenizer"] = str(tokenizer_path)

            # Step 4 — Export baseline
            if export_baseline:
                logger.info("=" * 50)
                logger.info("EXPORTING BASELINE TO COREML")
                logger.info("=" * 50)

                try:
                    sleeve_path = self.export_baseline(encoders)
                    results["exported"].append("baseline")
                    results["paths"]["baseline"] = str(sleeve_path)

                    if not skip_validation:
                        passed = self.validate_baseline(encoders)
                        results["validated"]["baseline"] = passed
                    else:
                        logger.info("Skipping baseline validation.")

                except Exception as e:
                    logger.error("Baseline export failed: %s", e)
                    results["validated"]["baseline"] = False

            # Step 5 — Export transformer
            if export_transformer:
                logger.info("=" * 50)
                logger.info("EXPORTING TRANSFORMER TO COREML")
                logger.info("=" * 50)

                try:
                    transformer_path = self.export_transformer(encoders)
                    results["exported"].append("transformer")
                    results["paths"]["transformer"] = str(transformer_path)

                    if not skip_validation:
                        passed = self.validate_transformer(encoders)
                        results["validated"]["transformer"] = passed
                    else:
                        logger.info("Skipping transformer validation.")

                except Exception as e:
                    logger.error("Transformer export failed: %s", e)
                    results["validated"]["transformer"] = False

            # Summary
            logger.info("=" * 50)
            logger.info("COREML EXPORT SUMMARY")
            logger.info("=" * 50)
            for model_name in results["exported"]:
                passed = results["validated"].get(model_name, "skipped")
                status = "✓ PASSED" if passed is True else (
                    "✗ FAILED" if passed is False else "— SKIPPED"
                )
                logger.info(
                    "  %-20s validation: %s",
                    model_name,
                    status,
                )

            self._log_mlflow(
                exported_models=results["exported"],
                validation_results=results["validated"],
            )

        return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Export vinyl grader models to CoreML"
    )
    parser.add_argument(
        "--config",
        default="grader/configs/grader.yaml",
        help="Path to grader config YAML",
    )
    parser.add_argument(
        "--model",
        choices=["baseline", "transformer", "both"],
        default="both",
        help="Which model(s) to export",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip round-trip validation after export",
    )
    args = parser.parse_args()

    exporter = CoreMLExporter(config_path=args.config)
    exporter.run(
        export_baseline=(args.model in ["baseline", "both"]),
        export_transformer=(args.model in ["transformer", "both"]),
        skip_validation=args.skip_validation,
    )


if __name__ == "__main__":
    main()
