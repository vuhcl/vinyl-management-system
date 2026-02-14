# 💰 Vinyl Price Estimation Model

Regression-based system that predicts the **fair-market value of vinyl records** using historical sales (e.g. from Discogs), predicted condition from the NLP classifier, and metadata.

## Product Vision

- **Inputs**: Album metadata, historical sale prices (1–3 years), condition features (sleeve/media), time-decay signals.
- **Output**: For a given `item_id`, a predicted price and prediction interval in the format:

```json
{
  "item_id": "12345",
  "predicted_price": 34.50,
  "prediction_interval": [30.20, 38.80],
  "model_version": "v1.0"
}
```

## Repository Structure

```
price_estimation/
├── data/
│   ├── raw/          # sales.csv, metadata.csv
│   └── processed/
├── src/
│   ├── data/         # ingest, preprocess
│   ├── features/     # historical_price, condition_features, embeddings
│   ├── models/       # baseline (linear), gradient_boosting (LightGBM)
│   ├── evaluation/   # metrics (MAE, MAPE), prediction_interval
│   └── pipeline.py
├── configs/
│   └── base.yaml
└── README.md
```

## Data

- **Historical sales**: Discogs does not expose historical sale prices via the public API. Use:
  - CSV in `data/raw/sales.csv` with columns: `item_id`, `sale_price`, `sale_date` (optional: `sleeve_condition`, `media_condition`).
  - Or export/dumps from marketplace tools.
- **Metadata**: `data/raw/metadata.csv` with `item_id`, `artist`, `genre`, `release_year` (or use Discogs API when `discogs.use_api: true` in config).

Sample files `data/raw/sales_sample.csv` and `metadata_sample.csv` are provided; copy them to `sales.csv` and `metadata.csv` for a quick run (or use the included `sales.csv` / `metadata.csv` if present).

## Setup

From project root:

```bash
pip install -r requirements.txt   # includes lightgbm for gradient_boosting phase
```

## Running the Pipeline

From project root:

```bash
# Baseline (linear regression)
PYTHONPATH=. python -m price_estimation.src.pipeline --phase baseline

# Advanced (LightGBM + prediction intervals)
PYTHONPATH=. python -m price_estimation.src.pipeline --phase gradient_boosting

# Options
#   --config path/to/config.yaml
#   --skip-ingest   use existing data/processed/sales_processed.csv
#   --no-mlflow     disable MLflow logging
```

From `price_estimation/`:

```bash
PYTHONPATH=.. python -m src.pipeline --phase baseline
```

## Config

- `configs/base.yaml`: seed, paths, feature settings, baseline/gradient_boosting hyperparameters, split fractions, MLflow experiment name.
- Time-based train/val/test split uses each item’s latest sale date so that test set is temporally after train.

## Evaluation

- **Metrics**: MAE (primary), MAPE; for gradient_boosting also interval coverage and mean interval width.
- **Success criteria**: Lower MAE than baseline; prediction intervals with reasonable coverage (e.g. 80%).

## Reproducibility

- Fixed random seed in config.
- Versioned config and saved preprocessing/feature pipeline state with the model artifact.
- Optional MLflow experiment tracking for params and metrics.

## Definition of Done

- [x] Historical price features built reproducibly
- [x] Baseline linear regression trained
- [x] Advanced gradient boosting trained with feature set
- [x] MAE/MAPE reported
- [x] Prediction intervals validated (coverage metric)
- [x] Model artifacts saved with reproducible pipeline
