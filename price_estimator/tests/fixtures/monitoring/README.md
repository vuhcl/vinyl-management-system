# Monitoring fixtures

Synthetic **releases_features** rows are generated in
[`test_monitoring_vinyliq.py`](../../test_monitoring_vinyliq.py) with a fixed RNG seed (`make_releases_features_df`), so drift statistics are reproducible without committing large Parquet files.

**Challenge / drift axis** (`test_monitoring_detects_drift_challenge_fixture`):

- **`year` / `decade`**: shift years by **+14** years (clamped), then recompute `decade`.
- **`genre`**: set every row to **`ObscureGenre`** to force a strong categorical shift versus the reference mix.

See [`thresholds.yaml`](../../../src/monitoring/thresholds.yaml) for PSI / KS / chi-square gates.
