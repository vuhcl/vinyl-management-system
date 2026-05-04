# VinylIQ / `price_estimator` (subproject)

Dedicated **price microservice** and training pipeline for the monorepo: Discogs marketplace labels (SQLite), catalog **feature store**, gradient-boosting regressors (default **residual** target vs Discogs median), **FastAPI** (`POST /estimate`), and a Chrome extension under [`vinyliq-extension/`](../vinyliq-extension/).

This package is part of **vinyl_management_system**. It uses **shared** utilities (`shared.project_env` for repo-root `.env`, Discogs helpers where applicable) and resolves **`vinyliq.paths.*`** in YAML against the **`price_estimator/`** directory.

---

## Repository structure

```
price_estimator/
├── configs/
│   └── base.yaml      # vinyliq.* paths, training_label, training_target, tuning, ensemble
├── src/
│   ├── api/           # FastAPI app
│   ├── training/      # train_vinyliq.py (single train + tune entrypoint)
│   ├── inference/     # service.py, service_factory.py, mlflow_bundle.py — stats + features + model
│   ├── storage/       # marketplace_stats, marketplace_projection, sqlite_util, feature_store, sale_history
│   ├── features/      # vinyliq_features + VinylIQFeatureSchema (column contract)
│   ├── models/        # fitted_regressor (facade) + regressor_{fitted,training,metrics}, sample_weights; pyfunc; overlays
│   ├── ingest/        # discogs_dump streaming parser
│   └── scrape/        # sale history parsing helpers
├── scripts/           # collectors, ingest, queues, audits, merges
├── data/              # raw, processed, cache (largely gitignored)
├── artifacts/         # default model_dir (see vinyliq.paths)
└── README.md
```

**Config:** default **`price_estimator/configs/base.yaml`**. Override with **`VINYLIQ_CONFIG`**. Collectors and training load the repo-root **`.env`** automatically where noted.

**Local `pipeline.estimate`:** uses the same YAML merge (`inherits`) and service factory as the API (`load_service_from_config` → `build_inference_service_from_merged_config`), so MLflow model paths, optional Postgres, and `DISCOGS_*` token resolution match `uvicorn` startup.

---

## Path conventions

Commands below assume the **shell cwd** is the **monorepo root** and imports work (`uv run` from the workspace root, or **`PYTHONPATH=.`** with `python price_estimator/scripts/...`).

| Anchor | Scripts / arguments |
|--------|---------------------|
| **Monorepo root** | Relative **`--db`** on **`collect_marketplace_stats.py`** and **`collect_sale_history_botasaurus.py`** (e.g. `price_estimator/data/cache/marketplace_stats.sqlite`). |
| **`price_estimator/` package root** | Relative **`--feature-db`** on **`ingest_discogs_dump.py`**; relative **`--db`** / **`--marketplace-db`** on **`build_stats_collection_queue.py`**; paths in **`vinyliq.paths`** in YAML; default **`--config`** for **`audit_training_db_joins.py`** (`price_estimator/configs/base.yaml`). |
| **Process cwd** | **`--dump`**, **`--ids-out`**, **`--release-ids`**, **`--out`** on queue/export scripts when not joined by the script — examples use repo-root-relative paths like **`price_estimator/data/raw/...`**. |

---

## Config summary (read before running)

- **`vinyliq.paths`**: `marketplace_db`, `feature_store_db`, `sale_history_db`, `model_dir`, optional **`mlflow_cache_dir`** — **relative to `price_estimator/`** unless absolute.
- **`vinyliq.mlflow_model_uri`**: optional MLflow URI for the Price API to download weights at startup (see **Serve API** and **MLflow**). Environment **`VINYLIQ_MLFLOW_MODEL_URI`** overrides this when set.
- **`vinyliq.training_label`**: e.g. **`sale_floor_blend`** / **`sale_floor`**; blend modes need **`paths.sale_history_db`** populated.
- **`vinyliq.training_target`**: default **`residual_log_median`**; optional **`residual_z_clip_abs`**.
- **`vinyliq.tuning`**: **`enabled`**, **`n_trials_per_family`**, **`model_families`**, **`search_spaces`**, **`constraints`**, **`cv_folds`**, **`cv_stratify`**, **`selection_metric`** (and related blocks in [`configs/base.yaml`](configs/base.yaml)).
- **`vinyliq.ensemble`**: two-head / blend options — **`ensemble.enabled` requires `tuning.enabled: true`** (ensemble uses champion hyperparameters from the tuning loop; training errors otherwise).
- **MLflow**: **`MLFLOW_TRACKING_URI`** in `.env`; fallbacks and **`mlflow.log_artifacts`** live under the same YAML / top-level keys as today.

---

## Setup

From **monorepo root**:

```bash
uv sync
# Optional: uv sync --extra test
```

Prefer **`uv run python ...`** for scripts and **`uv run python -m price_estimator.src.training.train_vinyliq`** for training so workspace dependencies resolve.

---

## Full pipeline (commands in order)

### 1) Catalog from monthly Discogs dump

Download **`discogs_*_releases.xml.gz`** from [Discogs Data](https://data.discogs.com/) (CC0).

Relative **`--dump`** / **`--ids-out`** are **cwd-relative**; relative **`--feature-db`** is joined under **`price_estimator/`** (see Path conventions).

```bash
# Streaming ingest (use --limit 5000 for a smoke test)
uv run python price_estimator/scripts/ingest_discogs_dump.py \
  --dump price_estimator/data/dumps/discogs_20240201_releases.xml.gz

# Also write release IDs for collectors
uv run python price_estimator/scripts/ingest_discogs_dump.py \
  --dump /path/to/releases.xml.gz \
  --ids-out price_estimator/data/raw/dump_release_ids.txt

# IDs only (no feature DB writes)
uv run python price_estimator/scripts/ingest_discogs_dump.py \
  --dump /path/to/releases.xml.gz \
  --ids-only \
  --ids-out price_estimator/data/raw/dump_release_ids.txt
```

Use **`--include-deleted`**, **`--probe-community N`**, **`--commit-every`** as needed. If XML shape drifts, use an external exporter + **`build_feature_store.py`**.

### 2) (Optional) Export IDs or build a collection queue

**`export_release_ids.py`**: dump IDs from **`feature_store.sqlite`**; community sorts (**`--sort-by have|want|combined`**) require **`--marketplace-db`** pointing at **`marketplace_stats.sqlite`**.

**`build_stats_collection_queue.py`**: merge proxy head + stratified tail. **`--db`** / **`--marketplace-db`** are **package-relative**.

```bash
uv run python price_estimator/scripts/build_stats_collection_queue.py \
  --db data/feature_store.sqlite \
  --out price_estimator/data/raw/collection_queue.txt \
  --rank-by proxy \
  --primary-limit 350000 \
  --extra-limit 350000 \
  --stratify-per-bucket 40 \
  --stratify-by decade_genre \
  --stratify-order proxy \
  --seed 42 \
  --max-total 500000
```

For **`--rank-by combined|have|want`**, add e.g. **`--marketplace-db data/cache/marketplace_stats.sqlite`** after marketplace data exists.

### 3) Marketplace labels (Discogs API)

**`collect_marketplace_stats.py`**: relative **`--db`** resolves from the **monorepo root** (default: **`price_estimator/data/cache/marketplace_stats.sqlite`**).

- **`--collect-mode full`** (default): **`GET /releases`** + **`GET /marketplace/price_suggestions`** (two requests per ID; each counts toward **`--req-per-minute`**).
- **`--collect-mode stats_only`**: **`GET /marketplace/stats`** only (legacy; use if you need **`blocked_from_sale`**).
- **`--resume`**, **`--max`**, **`--workers`**, **`--req-per-minute`**, **`--resume-mode query`** for very large DBs (per-ID resume check), **`--curr-abbr USD`**.

```bash
uv run python price_estimator/scripts/collect_marketplace_stats.py --oauth-login

uv run python price_estimator/scripts/collect_marketplace_stats.py \
  --release-ids price_estimator/data/raw/dump_release_ids.txt \
  --db price_estimator/data/cache/marketplace_stats.sqlite \
  --curr-abbr USD \
  --workers 8 \
  --req-per-minute 55 \
  --resume
```

Auth: **`DISCOGS_TOKEN`** / **`DISCOGS_USER_TOKEN`**, or OAuth env vars (**`--oauth-login`** once). See the script docstring for endpoint details and rate-limit behavior.

**Community source of truth:** **`marketplace_stats.community_have` / `community_want`** from **`full`** mode. Feature-store schema no longer carries `want_count` / `have_count` / `want_have_ratio`; sorting/queue APIs join MP directly.

### 4) Sale history (website; Botasaurus)

**`collect_sale_history_botasaurus.py`**: same **monorepo-root** **`--db`** rule as step 3 (default **`price_estimator/data/cache/sale_history.sqlite`**). Requires Botasaurus, a persistent Chrome **`--profile`** (or **`DISCOGS_SALE_HISTORY_BROWSER_PROFILE`**) with Discogs login, and compliance with Discogs terms — see the script module docstring.

```bash
uv run python price_estimator/scripts/collect_sale_history_botasaurus.py \
  --release-ids price_estimator/data/raw/dump_release_ids.txt \
  --db price_estimator/data/cache/sale_history.sqlite \
  --profile /path/to/discogs-chrome-profile \
  --resume
```

### 5) (Optional) QA and merge SQLite copies

Overlap / join audit (defaults read **`price_estimator/configs/base.yaml`** **`vinyliq.paths`**; override with **`--feature-store-db`** / **`--marketplace-db`** / **`--sale-history-db`**):

```bash
PYTHONPATH=. uv run python price_estimator/scripts/audit_training_db_joins.py
PYTHONPATH=. uv run python price_estimator/scripts/audit_training_db_joins.py --format-audit
```

Merge in-repo duplicates or teammate DBs into canonical paths:

```bash
PYTHONPATH=. uv run python price_estimator/scripts/merge_sqlite_sources.py \
  --discover-repo-cache --dry-run
PYTHONPATH=. uv run python price_estimator/scripts/merge_sqlite_sources.py \
  --discover-repo-cache --apply
```

See **`merge_sqlite_sources.py --help`** for **`--mp-merge`** / **`--sh-merge`** and explicit **`--out-mp`** / **`--out-sh`**.

### 6) (Optional) Normalize sale-history USD strings

One-shot migration for **`release_sale`** user-price text (**`finalize_sale_history_usd_strings.py`**). **`--db`** accepts **`price_estimator/...`** relative to the monorepo root or other rules documented in the script — use **`--dry-run`** first; back up the DB.

```bash
PYTHONPATH=. uv run python price_estimator/scripts/finalize_sale_history_usd_strings.py \
  --db price_estimator/data/cache/sale_history.sqlite --dry-run
```

### 7) Train + tune (single entrypoint)

There is **no** separate **`tune_*.py`**. Hyperparameter search is driven by **`vinyliq.tuning`**, **`model_families`**, and **`search_spaces`** in YAML (see **Tuning workflow** below).

```bash
uv run python -m price_estimator.src.training.train_vinyliq
```

CLI flags (parsed first; unknown args are ignored with a stderr notice):

| Flag | Purpose |
|------|---------|
| **`--no-mlflow`** | Disable MLflow (same as disabling it in config). |
| **`--mlflow-no-artifacts`** | Log params/metrics only; no model artifact upload. |
| **`--google-application-credentials PATH`** | Service account JSON for GCS artifact upload (sets env for this process). |
| **`--residual-sanity-only`** | Load frame, print residual **z** stats / baselines, exit without tuning or fit. |
| **`--sale-condition-policy`** | Override **`nm_substrings_only`** \| **`ordinal_cascade`** for **`sale_floor_blend`** (MLflow A/B). |

**Holdout:** split is by **`release_id`** (whole releases held out of train).

### 8) (Optional) Grade-delta scale JSON

**`fit_grade_delta_scale.py`** emits **`grade_delta_scale.json`** for overlays merged by **`load_params_with_grade_delta_overlays`** next to **`condition_params.json`** (scaler keys nest under **`grade_delta_scale`**; optional top-level **`alpha`** / **`beta`** from the fit overwrite serving coefficients).

Default DB paths are under **`price_estimator/data/`** (package-relative); **`--out`** is required.

```bash
PYTHONPATH=. uv run python price_estimator/scripts/fit_grade_delta_scale.py \
  --out price_estimator/artifacts/vinyliq/grade_delta_scale.json
```

**Defaults (sale-history DBs present):** jointly fits **`price_gamma`** / **`age_k`** and **α / β** from pooled cross-grade contrasts (symmetric VG+ vs NM plus asymmetric media/sleeve slices). Metadata **`fit_kind`** reports **`cross_grade_bin_median_v2_alpha_beta`** when triplet fitting runs.

| Flag | Purpose |
|------|---------|
| **`--no-fit-alpha-beta`** | Keep **`--base-alpha`** / **`--base-beta`** fixed; legacy grid on scalers only (**`fit_kind`** **`cross_grade_bin_median_v1`**). |
| **`--beta-per-alpha-fallback`** | When asymmetric strata are too sparse, split **`s = α+β`** with **`β = ratio·α`** (default: **`base_beta/base_alpha`**). |

Use **`--placeholder`** for a bootstrap JSON without DB reads.

### 9) Serve API

```bash
uv run uvicorn price_estimator.src.api.main:app --host 127.0.0.1 --port 8801
```

Example (**weights from registry**, cache under default **`artifacts/mlflow_model_cache`**):

```bash
export MLFLOW_TRACKING_URI=https://your-tracking-host/
export VINYLIQ_MLFLOW_MODEL_URI='models:/VinylIQPrice@production'
uv run uvicorn price_estimator.src.api.main:app --host 127.0.0.1 --port 8801
```

Host/port defaults also appear under **`api.*`** in YAML. Set **`VINYLIQ_API_KEY`** if the API should require **`X-API-Key`**.

**Model weights**

- **Bundled path (default):** **`vinyliq.paths.model_dir`** — same layout as after **`train_vinyliq`** (**`model_manifest.json`**, **`regressor.joblib`** or **`xgb_model.joblib`**, encoders, **`condition_params.json`**). Optional **`grade_delta_scale.json`** in that directory is merged for inference.
- **MLflow pull:** If **`VINYLIQ_MLFLOW_MODEL_URI`** or **`vinyliq.mlflow_model_uri`** is set, startup downloads the champion **artifact tree** logged as **`vinyliq_artifacts`** (training logs it beside the pyfunc **`vinyliq_model`** bundle). Supported URI forms include **`models:/VinylIQPrice@production`**, **`models:/VinylIQPrice/<version>`**, and **`runs:/<run_id>`** (normalized to **`…/vinyliq_artifacts`**). Requires a resolvable **`MLFLOW_TRACKING_URI`** (or YAML **`mlflow.tracking_uri_fallback`**). Downloads are cached under **`VINYLIQ_MLFLOW_CACHE_DIR`** or **`vinyliq.paths.mlflow_cache_dir`**; set **`VINYLIQ_MLFLOW_FORCE_REFRESH=true`** to ignore the cache.

**`GET /health`** returns **`model_loaded`** if **`model_manifest.json`**, **`regressor.joblib`**, or **`xgb_model.joblib`** exists under the effective model directory, and **`model_source`** **`local`** vs **`mlflow`**.

**`POST /estimate`** responses include **`num_for_sale`** and **`warnings`** (e.g. **`low_market_depth`** when **`num_for_sale < 3`**).

---

## Tuning workflow (YAML-driven)

Edit **`price_estimator/configs/base.yaml`** (or a merged override via **`VINYLIQ_CONFIG`**):

- Flip **`vinyliq.tuning.enabled`** and set **`n_trials_per_family`**, **`model_families`**, **`search_spaces`**, **`constraints`**, **`selection_metric`**, CV (**`cv_folds`**, **`cv_stratify`**, **`val_fraction`**, **`test_fraction`**), **`early_stopping_rounds`**, etc.
- The training run performs trial search, selects a champion, refits, and writes artifacts under **`vinyliq.paths.model_dir`**.
- **`vinyliq.ensemble.enabled: true`** requires **`vinyliq.tuning.enabled: true`**.
- MLflow behavior: set **`MLFLOW_TRACKING_URI`**; use CLI **`--no-mlflow`** / **`--mlflow-no-artifacts`** for local-only runs. If unset, **`tracking_uri_fallback`** in YAML may point at a local SQLite metadata store (shared patterns with grader — see **MLflow** below).

---

## MLflow (same as grader)

- Set **`MLFLOW_TRACKING_URI`** in repo-root **`.env`** to your tracking server.
- Align the server’s **`--default-artifact-root`** (e.g. GCS) with grader when sharing a bucket.
- Use **`GOOGLE_APPLICATION_CREDENTIALS`** or ADC for GCS uploads (training uploads and **Price API downloads** both honor **`mlflow.google_application_credentials`** in YAML when relevant).
- If **`MLFLOW_TRACKING_URI`** is unset, **`tracking_uri_fallback`** in **`configs/base.yaml`** can default to a local metadata DB (e.g. under **`grader/experiments/`**); experiment name **`vinyl_price_estimator`**.
- **Training** logs metrics, hyperparameters, tags, the YAML config, the flat champion directory under **`vinyliq_artifacts/`**, and a **`mlflow.pyfunc`** model under **`vinyliq_model/`** when artifact logging is enabled. Registry options (**`registry_model_name`**, aliases **`staging`** / **`production`**) live under **`mlflow.*`** in YAML.
- **Price API:** optional pull of **`vinyliq_artifacts`** via **`VINYLIQ_MLFLOW_MODEL_URI`** (or **`vinyliq.mlflow_model_uri`**). Registry URIs resolve the version’s **`source`** run and swap **`vinyliq_model` → `vinyliq_artifacts`** so inference loads the same files as local **`model_dir`**. See **Serve API** for cache and force-refresh env vars.

---

## Overnight / long runs (optional)

On macOS, prevent sleep during long collection or training:

```bash
caffeinate -dimsu uv run python -m price_estimator.src.training.train_vinyliq 2>&1 | tee price_estimator/artifacts/vinyliq_train.log
```

---

## Live API ingest (small lists or collection)

For a short list of IDs or your Discogs username (collection + wantlist), without the monthly dump:

```bash
uv run python price_estimator/scripts/ingest_from_discogs.py \
  --release-ids price_estimator/data/raw/my_release_ids.txt

uv run python price_estimator/scripts/ingest_from_discogs.py \
  --username YourDiscogsUsername
```

Defaults write feature + marketplace SQLite under **`price_estimator/data/`**. Use **`--delay`**, **`--fetch-master`**, **`--no-stats`** / **`--no-features`** as needed (**`--help`**).

---

## Training semantics (targets and leakage)

- **Default (`vinyliq.training_target.kind: residual_log_median`)** — The model predicts **`z = log1p(y_label) - log1p(anchor)`**; **`y_label`** comes from **`training_label`** (e.g. **`sale_floor_blend`** when **`sale_history.sqlite`** is configured). The **anchor** prefers **`release_lowest_price`**, then marketplace **`lowest_price`** / **`median_price`**. Training **features** omit same-snapshot listing-dollar scalars; community counts come from **`marketplace_stats`** only. At inference / pyfunc, the service adds **`log1p(anchor_live)`** back, then condition adjustment and **`expm1`**.
- **`price_suggestion`** targets are **Discogs suggested** prices, not observed sales — pseudo-labels; **`price_suggestion_fallback_lowest: true`** when ladders are often empty.
- **Legacy `dollar_log1p`** — direct **`log1p(price)`** with marketplace scalars in **X** at train time; prefer **residual** to avoid train/serve skew.
- **Why residual** — avoids fitting the label from the same snapshot’s median/lowest in **X**.
- **Temporal drift** — residual does not fix distribution shift between collection time and live API.
- **Validation metrics** — MAE / WAPE / median APE are computed in **dollar** space after mapping through the row’s training anchor (**`expm1`** on both sides).

---

## Environment

| Variable | Purpose |
|----------|---------|
| `DISCOGS_USER_TOKEN` or `DISCOGS_TOKEN` | Personal token for API collectors |
| `VINYLIQ_CONFIG` | Override path to YAML |
| `VINYLIQ_API_KEY` | If set, API requires `X-API-Key` |
| `PRICE_SERVICE_URL` | Web monolith proxies `/api/price/...` here when set |
| `MLFLOW_TRACKING_URI` | MLflow server URL (training + optional API artifact pull) |
| `VINYLIQ_MLFLOW_MODEL_URI` | If set, Price API downloads champion weights from MLflow (overrides **`vinyliq.mlflow_model_uri`**) |
| `VINYLIQ_MLFLOW_CACHE_DIR` | Local cache directory for downloaded **`vinyliq_artifacts`** (overrides **`vinyliq.paths.mlflow_cache_dir`**) |
| `VINYLIQ_MLFLOW_FORCE_REFRESH` | If `1` / `true` / `yes`, re-download even when cache stamp matches |
| `GOOGLE_APPLICATION_CREDENTIALS` | GCS artifact upload / download (optional) |
| `REDIS_HOST` | Optional Memorystore (or local Redis) host for the L1 stats cache; unset = SQLite-only |
| `REDIS_PORT` | Defaults to `6379` |
| `REDIS_DB` | Defaults to `0` |
| `REDIS_TTL_SECONDS` | Defaults to `2592000` (30 days); matches the demo system-design slide |

When `REDIS_HOST` is set, `InferenceService.fetch_stats` reads through
Redis -> SQLite -> Discogs (write-through on live fetches). The
[`RedisStatsCache`](src/storage/redis_stats_cache.py) gracefully
degrades to a no-op if Redis is unreachable or the `redis` package is
missing — local dev keeps working without any Redis at all.

---

## GKE / containerized deployment

For the demo deploy on GKE Autopilot (Memorystore Redis, GCS, MLflow,
Workload Identity, GitHub Actions Workload Identity Federation), see
[`k8s/demo/README.md`](../k8s/demo/README.md). Highlights specific to
this package:

- The price API loads weights from **`vinyliq.paths.model_dir`** by default, or from **MLflow** when **`VINYLIQ_MLFLOW_MODEL_URI`** (or **`vinyliq.mlflow_model_uri`**) is set—downloads land in **`mlflow_cache_dir`** (emptyDir or PVC) so the pod does not need the full bundle in the image.
- In the bundled demo, trained **`regressor.joblib`** / **`model_manifest.json`** (and friends) may live on a **`ReadWriteOnce`** PersistentVolumeClaim populated from [`price_estimator/artifacts/vinyliq/`](artifacts/vinyliq/) at bootstrap time when not using MLflow pull.
- A `ConfigMap` mounted at `/etc/vinyliq/config.yaml` overrides
  `vinyliq.paths.*` to point at the PVC mount; the Deployment sets
  `VINYLIQ_CONFIG=/etc/vinyliq/config.yaml`.
- The image is built by
  [`.github/workflows/demo-deploy.yml`](../.github/workflows/demo-deploy.yml)
  via [`price_estimator/Dockerfile`](Dockerfile) (two-stage,
  selective-COPY, `uv export` deps stage).

---

## Quick local demo

```bash
# Monorepo root
uv sync
PYTHONPATH=. uv run python price_estimator/scripts/seed_demo_data.py
uv run python -m price_estimator.src.training.train_vinyliq
uv run uvicorn price_estimator.src.api.main:app --host 127.0.0.1 --port 8801
```
