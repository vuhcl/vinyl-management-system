# Price Estimator (subproject)

Estimates fair market value for vinyl releases using marketplace data (e.g. Discogs sales history) and optional condition from the **NLP condition classifier**.

## Status

Stub: pipeline entrypoint and config are in place; model and features are to be implemented.

## Integration

- **Inputs**: Release ID (Discogs), optional sleeve/media condition (from `nlp_condition_classifier`), optional historical prices.
- **Output**: Point estimate and optional interval (e.g. low/median/high).
- **Data**: Discogs API (price suggestions, marketplace listings) or scraped sales data.

## Repository structure

```
price_estimator/
├── src/
│   ├── data/        # ingest (Discogs price data, condition from NLP)
│   ├── features/    # price features
│   ├── models/      # regressor (e.g. quantile regression)
│   └── pipeline.py  # train / predict
├── configs/
│   └── base.yaml
└── README.md
```

## Running (stub)

From project root:

```bash
python -m price_estimator.pipeline --config configs/price_estimator.yaml
```

See `configs/price_estimator.yaml` for paths and model settings.
