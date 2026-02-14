# NLP Vinyl Condition Classifier

Supervised NLP system that predicts **sleeve and media condition** of a vinyl record from unstructured seller notes. Supports consistent grading, fair pricing, and integration with downstream tools (e.g. price estimation).

## Product vision

- **Input**: Free-text seller notes (e.g. from Discogs), optional metadata.
- **Output**: Per-item JSON with `predicted_sleeve_condition`, `predicted_media_condition`, and `confidence_scores` for each grade (Mint, Near Mint, Very Good Plus, Very Good, Good).
- **Non-goals**: Replacing human graders; fine-grained audio/physical inspection; real-time inference (offline batch is sufficient).

## Repository structure

```
nlp_condition_classifier/
├── data/
│   ├── raw/              # condition_labeled.csv or Discogs-sourced data
│   └── processed/         # cleaned text and labels
├── src/
│   ├── data/
│   │   ├── ingest.py      # Load from CSV or Discogs API
│   │   └── preprocess.py  # Text cleaning
│   ├── features/
│   │   ├── tfidf_features.py
│   │   └── embeddings.py  # Optional transformer embeddings
│   ├── models/
│   │   ├── baseline.py    # TF-IDF + Logistic Regression
│   │   └── transformer.py # Phase 2: DistilBERT fine-tuning
│   ├── evaluation/
│   │   ├── metrics.py    # Macro-F1, accuracy
│   │   └── calibration.py
│   └── pipeline.py       # End-to-end run
├── configs/
│   └── base.yaml
├── notebooks/
└── README.md
```

## Setup

From the **vinyl_management_system** project root:

```bash
# Install main project deps (includes scikit-learn, mlflow, pyyaml)
pip install -r requirements.txt

# Optional for Phase 2 (transformer)
pip install transformers torch datasets
```

## Data

Place labeled data in `nlp_condition_classifier/data/raw/`:

- **condition_labeled.csv** with columns:
  - `item_id`
  - `seller_notes` (or `notes` / `description`)
  - `sleeve_condition`
  - `media_condition`
  - Optional: `artist`, `genre`, `release_year`

Condition values are normalized to: **Mint**, **Near Mint**, **Very Good Plus**, **Very Good**, **Good**.

Alternatively, enable Discogs in `configs/base.yaml` and set `DISCOGS_USER_TOKEN` to fetch listings (seller notes + condition) for training.

## Running the pipeline

From project root:

```bash
# Resolve imports from repo root
export PYTHONPATH=/Users/nowaki027/vinyl_management_system

# Run baseline (TF-IDF + Logistic Regression)
python -m nlp_condition_classifier.src.pipeline --phase baseline

# Skip re-ingest (use existing processed data)
python -m nlp_condition_classifier.src.pipeline --skip-ingest --phase baseline

# Disable MLflow logging
python -m nlp_condition_classifier.src.pipeline --no-mlflow --phase baseline
```

Pipeline steps:

1. Ingest labeled data (CSV or Discogs).
2. Preprocess text (lowercase, strip URLs, min tokens).
3. Stratified train/val/test split.
4. Train baseline (or Phase 2 transformer stub).
5. Evaluate: **macro-F1**, **accuracy**, **ECE** (calibration).
6. Save model artifacts under `artifacts/baseline/` (or `artifacts/transformer/`).

## Evaluation

- **Primary metric**: Macro-F1 (sleeve and media separately).
- **Secondary**: Accuracy, calibration (ECE and calibration curves).
- **Success criteria**: Outperform baseline macro-F1 by 10–15% with transformer; well-calibrated confidence outputs.

## Output format

Per-item prediction (e.g. from `model.predict_item(item_id, seller_notes)`):

```json
{
  "item_id": "12345",
  "predicted_sleeve_condition": "Near Mint",
  "predicted_media_condition": "Very Good Plus",
  "confidence_scores": {
    "Mint": 0.05,
    "Near Mint": 0.75,
    "Very Good Plus": 0.15,
    "Very Good": 0.05,
    "Good": 0.0
  }
}
```

## Reproducibility

- Fixed random seed in `configs/base.yaml` and pipeline.
- Preprocessing and model config in YAML; artifact dir stores vectorizer + classifiers.
- Optional MLflow experiment logging (experiment name: `nlp_condition_classifier`).

## Definition of done

- [x] Preprocessed text dataset ready (pipeline writes `data/processed/condition_processed.csv`)
- [x] Baseline TF-IDF + Logistic Regression trained
- [x] Macro-F1 (and accuracy) reported
- [ ] Transformer fine-tuning (Phase 2 stub in place)
- [x] Model artifacts saved with reproducible pipeline
- [x] Confidence calibration (ECE) validated

## Resume framing goal

> Built an end-to-end NLP pipeline predicting vinyl condition from seller notes, improving macro-F1 from 0.62 → 0.79 with transformer fine-tuning, including calibrated probabilities and reproducible preprocessing pipeline.
