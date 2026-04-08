# VinylIQ / `price_estimator`

Dedicated **price microservice** and training pipeline: Discogs `marketplace/stats` labels, catalog feature store (SQLite), gradient-boosting regressors (default **residual** target vs Discogs median), FastAPI (`POST /estimate`), and a Chrome extension under [`vinyliq-extension/`](../vinyliq-extension/).

## Layout

| Path | Role |
|------|------|
| `src/api/main.py` | FastAPI: `/health`, `/estimate`, `/collection/value` |
| `src/api/schemas.py` | Pydantic models |
| `src/inference/service.py` | Stats + feature store + model orchestration |
| `src/features/vinyliq_features.py` | Condition ordinals, feature row builder |
| `src/models/xgb_vinyliq.py` | XGBoost artifact load/save |
| `src/storage/` | SQLite: marketplace cache/labels + `releases_features` |
| `src/training/train_vinyliq.py` | Train booster (requires joined labels + features) |
| `scripts/collect_marketplace_stats.py` | Rate-limited stats collector (`--resume`, `--max`) |
| `scripts/ingest_discogs_dump.py` | Monthly `releases.xml(.gz)` → feature store + optional ID list |
| `scripts/export_release_ids.py` | `feature_store.sqlite` → IDs (`--sort-by release_id`, `have`, or `want`) |
| `scripts/build_stats_collection_queue.py` | Merge catalog-proxy (or community) + stratified IDs → queue for stats collector |
| `scripts/ingest_from_discogs.py` | Live API → feature store + marketplace DB |
| `src/ingest/discogs_dump.py` | Streaming XML parser for dump rows |
| `scripts/build_feature_store.py` | CSV → feature store |
| `scripts/seed_demo_data.py` | Synthetic DBs for local dev |
| `configs/base.yaml` | Paths under `vinyliq.paths` |

## Quick local demo

```bash
# From monorepo root (uv workspace)
uv sync
PYTHONPATH=. python price_estimator/scripts/seed_demo_data.py
PYTHONPATH=. python -m price_estimator.src.training.train_vinyliq
PYTHONPATH=. uvicorn price_estimator.src.api.main:app --host 127.0.0.1 --port 8801
```

Optional: `DISCOGS_USER_TOKEN` or `DISCOGS_TOKEN` for live `marketplace/stats` (and set `VINYLIQ_API_KEY` to require the extension/web proxy to send `X-API-Key`).

## MLflow (same as grader)

- Set **`MLFLOW_TRACKING_URI`** to the same remote tracking server URL you use for the grader (e.g. `http://your-mlflow-host:5000`). Repo-root **`.env`** is loaded automatically (via `shared.project_env`), same as grader.
- The tracking server should use the same **`--default-artifact-root`** (e.g. `gs://your-bucket/mlflow-artifacts`) as grader training so runs land in one bucket.
- Use **`GOOGLE_APPLICATION_CREDENTIALS`** or Application Default Credentials when uploading to GCS.
- If `MLFLOW_TRACKING_URI` is unset, **`tracking_uri_fallback`** in `configs/base.yaml` defaults to **`sqlite:///grader/experiments/mlflow.db`** (shared local metadata DB with grader; experiment name is `vinyl_price_estimator`).
- Training logs metrics, hyperparameters, tags (`project`, `module`), the YAML config, and the full **`artifacts/vinyliq`** directory under the run artifact path **`vinyliq_model/`**.

## Environment

| Variable | Purpose |
|----------|---------|
| `DISCOGS_USER_TOKEN` or `DISCOGS_TOKEN` | Personal access token for live stats (either name) |
| `VINYLIQ_CONFIG` | Override path to YAML config |
| `VINYLIQ_API_KEY` | If set, API requires header `X-API-Key` |
| `PRICE_SERVICE_URL` | Web monolith proxies `/api/price/...` here when set |

## Ingesting real Discogs data

### A. Monthly dump → feature store (catalog, no prices)

Download **`discogs_*_releases.xml.gz`** from [Discogs Data](https://data.discogs.com/) (CC0).

**Where to put it:** Prefer **`price_estimator/data/dumps/`** (created for this; contents are gitignored) so paths stay obvious, e.g. `price_estimator/data/dumps/discogs_20240201_releases.xml.gz`. You can also keep the file anywhere (external disk, `~/Downloads`) and pass an absolute path to **`--dump`** — the tooling does not require it to live inside the repo.

If you already ingested the dump without **`--ids-out`**, export IDs from SQLite (no re-parse): **`scripts/export_release_ids.py`** with **`--out`**. Use **`--sort-by have`** or **`--sort-by want`** to order by community **`have_count`** or **`want_count`**. Optional **`--min-have`** / **`--min-want`**. Or use the **`sqlite3`** one-liner in that script’s docstring for plain ID order.

```bash
# Full ingest into SQLite (streaming; use --limit 5000 for a smoke test)
PYTHONPATH=. python price_estimator/scripts/ingest_discogs_dump.py \
  --dump price_estimator/data/dumps/discogs_20240201_releases.xml.gz

# Same run: also write all release IDs for the stats collector
PYTHONPATH=. python price_estimator/scripts/ingest_discogs_dump.py \
  --dump /path/to/discogs_20240201_releases.xml.gz \
  --ids-out price_estimator/data/raw/dump_release_ids.txt

# IDs only (no feature DB writes)
PYTHONPATH=. python price_estimator/scripts/ingest_discogs_dump.py \
  --dump /path/to/releases.xml.gz \
  --ids-only \
  --ids-out price_estimator/data/raw/dump_release_ids.txt
```

Deleted releases (`status="Deleted"`) are skipped by default; use **`--include-deleted`** to keep them. **`label_tier`** stays `0` in dump mode (no label-frequency tiering yet). If Discogs changes XML shape and parsing fails, use an external tool such as [`discogs-xml2db`](https://github.com/philipmat/discogs-xml2db) and export a CSV for **`build_feature_store.py`**.

### B. Marketplace labels (`/marketplace/stats`)

The collector loads repo-root **`.env`** automatically. Use either:

- **Personal token:** **`DISCOGS_TOKEN`** or **`DISCOGS_USER_TOKEN`**, or  
- **OAuth 1.0a:** **`DISCOGS_CONSUMER_KEY`**, **`DISCOGS_CONSUMER_SECRET`**, **`DISCOGS_OAUTH_TOKEN`**, **`DISCOGS_OAUTH_TOKEN_SECRET`** (obtain once with **`--oauth-login`**; register the same callback URL in your Discogs app as **`DISCOGS_OAUTH_CALLBACK`**).

```bash
# Optional one-time: print OAuth lines for .env
PYTHONPATH=. python price_estimator/scripts/collect_marketplace_stats.py --oauth-login

PYTHONPATH=. python price_estimator/scripts/collect_marketplace_stats.py \
  --release-ids price_estimator/data/raw/dump_release_ids.txt \
  --delay 2.5 \
  --resume \
  --max 10000
```

After adding **`requests-oauthlib`** to **`shared`**, run **`uv sync`** once at the repo root.

**`--resume`** skips IDs already in `marketplace_stats.sqlite`. Use **`--resume-mode query`** for very large DBs (per-ID SQLite check, low RAM) instead of default **`memory`** (loads all keys). **`--max N`** caps successful new upserts.

**Parallelism & limits:** **`--workers`** (default 8), **`--req-per-minute`** global sliding window (default 55), **`--max-retries`**, **`--backoff-base`**, **`--backoff-max`**, **`--http-timeout`**. IDs are read in a **streaming** fashion. One Discogs token still obeys Discogs’ per-app rate limit; use separate runs with different tokens and split ID files to scale further.

**Shard order:** Contiguous splits (e.g. **`split -l 50000`**) keep file order: **shard 1 = head of the file**. Lists from **`export_release_ids.py`** default **`--sort-by release_id`** are **lexicographic ID order**. For **catalog-based “head” first** (master + artist mass), use **`build_stats_collection_queue.py`** (default **`--rank-by proxy`**) without **`--shuffle-final`**, or **`export_release_ids.py --sort-by catalog_proxy`**. For **community want/have** ordering you need non-zero counts in SQLite; then **`--rank-by combined`** or **`export_release_ids.py --sort-by combined`**, then split.

### Data collection strategy (how many labels, which releases)

You **do not** need marketplace stats for the full Discogs catalog—only for `release_id`s that become training rows after joining to `releases_features`. Aim for **tens of thousands to low hundreds of thousands** of labels for a strong first model; validate with learning curves (MAE vs. training size) once you have ~20k+.

**What to prioritize**

1. **Popularity (have / want)** — Head releases usually have more liquid markets; `median_price` is more stable when `num_for_sale` is not tiny. Export with **`export_release_ids.py --sort-by have`** and/or **`--sort-by want`** (see §A).
2. **Stratified coverage** — Popularity alone under-represents old decades, niche genres, and low-have tail. Sample randomly **per bucket** (e.g. decade × `genre`) so the feature space is covered.
3. **Pure random** over the whole feature store wastes quota on cold listings; use it **inside** stratification, not as the only source.

**Practical recipe:** Build a merged ID list—e.g. a **proxy** head (default **catalog score**: master fan-out + primary-artist catalog mass), plus optional **stratified** slice—and run **`collect_marketplace_stats.py`** with **`--resume`** / **`--max`**. If **`have_count` / `want_count`** are populated (API ingest), you can use **`--rank-by combined`** instead. Training uses a **release_id-level holdout**, so diversity of pressings matters more than raw row count alone.

**Scripted merge:** **`build_stats_collection_queue.py`** reads `feature_store.sqlite`. Head size is **`--primary-limit`**, ordered by **`--rank-by`** (**`proxy`** default, or **`combined` / `have` / `want`** for community sorts). **`--extra-limit`** adds more IDs (same proxy order skipping duplicates, or complement community sort). Stratified sampling adds up to **`--stratify-per-bucket`** per **`decade_genre`** or **`decade`**; use **`--stratify-order proxy`** for catalog score per bucket, or **`community`** (alias **`popularity`**) for have+want. **Various-artist** rows (Discogs Various id **194**, or primary artist name containing ``various``) are **omitted** from the queue and from proxy fan-out counts. **`--max-per-primary-artist`** (default **5**, **0** = off) limits repeats of the same primary artist in the proxy head/extra blocks and in **proxy** stratified buckets. Optional **`--max-total`** caps the file; **`--shuffle-final`** permutes the merged list before writing.

```bash
PYTHONPATH=. python price_estimator/scripts/build_stats_collection_queue.py \
  --db price_estimator/data/feature_store.sqlite \
  --out price_estimator/data/raw/collection_queue.txt \
  --rank-by proxy \
  --primary-limit 350000 \
  --extra-limit 350000 \
  --stratify-per-bucket 40 \
  --stratify-by decade_genre \
  --stratify-order proxy \
  --seed 42 \
  --max-total 500000

PYTHONPATH=. python price_estimator/scripts/collect_marketplace_stats.py \
  --release-ids price_estimator/data/raw/collection_queue.txt \
  --delay 2.5 --resume --max 10000
```

Stratified sampling uses SQLite **window functions** (3.25+). Optional later refinement: cap releases per **`master_id`** when building the candidate set so alternate pressings do not dominate.

### C. Live API only (small lists or your collection)

```bash
# IDs file: one release ID per line (see data/raw/release_ids.example.txt)
PYTHONPATH=. python price_estimator/scripts/ingest_from_discogs.py \
  --release-ids price_estimator/data/raw/my_release_ids.txt

# Or pull IDs from your collection + wantlist (folder 0 + wants)
PYTHONPATH=. python price_estimator/scripts/ingest_from_discogs.py \
  --username YourDiscogsUsername
```

Defaults write to `data/feature_store.sqlite` and `data/cache/marketplace_stats.sqlite` (under `price_estimator/`). Use **`--delay`** (~2s) and jitter to stay under Discogs rate limits; add **`--fetch-master`** if you want `is_original_pressing` from the master release. **`--no-stats`** / **`--no-features`** split work if you already have one side filled.

Then train: `PYTHONPATH=. python -m price_estimator.src.training.train_vinyliq`.

## Targets, leakage, and train/serve alignment

- **Default target (`vinyliq.training_target.kind: residual_log_median`)**  
  The model predicts **`z = log1p(y_label) - log1p(median_price)`** where `y_label` comes from `training_label` (e.g. `spread_signal` in [`label_synthesis`](src/training/label_synthesis.py)) and `median_price` is from the same `marketplace_stats` row. **Features** omit same-snapshot Discogs price and liquidity (`baseline_median`, `log1p_baseline_median`, `num_for_sale`, `log_num_for_sale`). At **inference** and in the **MLflow pyfunc** bundle, the service adds **`log1p(median_live)`** to the model output, then applies the usual **condition** adjustment and `expm1`. Batch/pyfunc inputs must include **`discogs_median_price`** when the saved manifest has `target_kind: residual_log_median`.

- **Legacy (`dollar_log1p`)**  
  Direct `log1p(y_label)` target; training rows still pass **zeroed** marketplace stats into `row_dict_for_inference` in [`train_vinyliq`](src/training/train_vinyliq.py), while inference may fill live stats—prefer **residual** mode to avoid that skew.

- **Why residual avoids a common leak**  
  If `y_label` is (almost) a function of the same snapshot’s median/lowest, putting those scalars in **X** lets the model fit the label without generalizing. Residual uses median only in **y** and at **score time**, not as train-time features.

- **Temporal stats**  
  Residual removes **feature/label** leakage from a single snapshot; it does not remove **distribution shift** if Discogs aggregates drift between training data collection and live API calls. Versioned stats (`fetched_at`) or external sold prices are future work.

- **Tuning metrics**  
  Validation MAE, WAPE, and median APE are computed in **dollar** space: predictions are mapped back to **log1p(dollar)** with the row’s training median anchor, then [`mae_dollars` / `wape_dollars` / `median_ape_dollars`](src/models/fitted_regressor.py) use `expm1` on both sides.

- **Optional** `training_target.residual_z_clip_abs` winsorizes \(z\) for heavy-tailed rows.

## Training notes

- Split is by `release_id` (holdout set of whole releases).
