# Price Estimator (subproject)

Estimates fair market value for vinyl releases using historical sales (e.g. CSV from Discogs dumps), optional condition from the **vinyl condition grader**, and metadata.

## Inputs / Output

- **Inputs**: Release/item ID, historical sale prices and dates, optional sleeve/media condition, artist, genre, year.
- **Output**: Point estimate and prediction interval, e.g.:

```json
{
  "item_id": "12345",
  "predicted_price": 34.50,
  "prediction_interval": [30.20, 38.80],
  "model_version": "v1.0"
}
```

Or via `estimate(release_id, ...)`: `estimate_usd`, `interval_low`, `interval_high`, `status`.

## Data

- **Sales**: `data/raw/sales.csv` with columns `item_id`, `sale_price`, `sale_date` (optional: `sleeve_condition`, `media_condition`).
- **Metadata**: `data/raw/metadata.csv` with `item_id`, `artist`, `genre`, `release_year`.
- Sample files: `sales_sample.csv` / `metadata_sample.csv`; copy to `sales.csv` and `metadata.csv` for a quick run.

## Structure

```
price_estimator/
├── data/raw/
├── data/processed/
├── src/
│   ├── data/       # ingest, preprocess
│   ├── features/   # historical_price, condition_features, embeddings
│   ├── models/     # baseline (linear), gradient_boosting (LightGBM)
│   ├── evaluation/ # metrics, prediction_interval
│   └── pipeline.py # run_pipeline(), estimate()
├── configs/base.yaml
└── README.md
```

## Running

From project root:

```bash
# Baseline (linear regression)
PYTHONPATH=. python -m price_estimator.src.pipeline --phase baseline

# Gradient boosting + prediction intervals
PYTHONPATH=. python -m price_estimator.src.pipeline --phase gradient_boosting

# Options: --config PATH --skip-ingest --no-mlflow
```

## Estimate API

After training, use `estimate(release_id, sleeve_condition=..., media_condition=..., artifacts_dir=..., features_row=df_one_row)`.
If `features_row` is provided (one row with the model’s feature columns), returns `estimate_usd` and interval; otherwise returns stub with `status="no_features"`.
