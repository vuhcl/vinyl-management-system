# Vinyl condition grader (`grader/`)

Supervised NLP + rule layer that predicts **sleeve** and **media** condition from seller notes (Discogs / eBay JP style), with optional CoreML export for on-device iOS use.

## Quick eval for resume / portfolio

**Synthetic benchmark** (same data recipe as `grader/tests/conftest.py` — fast, reproducible):

```bash
# From monorepo root, with venv activated
python scripts/grader_eval_resume.py
```

Writes `artifacts/grader_eval_resume.json` with **macro-F1**, **accuracy**, and **ECE** on train/val/test for both heads.

Use the `disclaimer` field in that JSON: cite these numbers as **synthetic pipeline validation**, not real marketplace accuracy.

## Discogs ingestion (real marketplace data)

From the **monorepo root** (paths in `grader.yaml` are relative to cwd):

1. Create a [Discogs personal access token](https://www.discogs.com/settings/developers).
2. Put it in the **repo-root** `.env` as `DISCOGS_TOKEN=...` (see `.env` template comments), **or** export it in the shell.

Ingest modules call `load_dotenv` on that file automatically (searching upward from the current working directory, then from `grader/src/`). Existing shell env vars are not overwritten.

```bash
# Optional if DISCOGS_TOKEN is already in .env at repo root:
# export DISCOGS_TOKEN="your_token_here"

# Validate auth + parsing without writing files or MLflow
python -m grader.src.data.ingest_discogs --dry-run

# Full pull (default: up to 500 raw listings per canonical grade — can take a while)
python -m grader.src.data.ingest_discogs

# Faster smoke ingest (override per-grade cap)
python -m grader.src.data.ingest_discogs --target-per-grade 80
```

If `data.discogs.format_filter` is missing or blank, ingestion defaults to **`Vinyl`** (`DEFAULT_DISCOGS_FORMAT_FILTER` in `ingest_discogs.py`). Use `--format NAME` on the CLI to override.

**How it works:** Discogs does **not** expose `GET /marketplace/search` (404). The ingester pages **`GET /users/{username}/inventory`** for each name in `data.discogs.inventory_sellers` in `grader.yaml` and keeps vinyl rows until each canonical **media** grade hits `target_per_grade`. Edit that list to add high-volume sellers you trust.

**Pagination cap:** For shops that aren’t yours, Discogs returns **403** after **page 100** (at most **10,000** items per seller at 100 rows/page).

**Page size (100 vs 250):** The **seller profile** on the website supports **`limit=250`** and **`format=Vinyl`** (e.g. `discogs.com/seller/Redscroll/profile?limit=250&format=Vinyl`). The ingester mirrors that on **`api.discogs.com/users/{user}/inventory`** with **`per_page`**, **`limit`**, and **`format`** (`data.discogs.inventory_per_page`, `inventory_send_limit_param`, `inventory_format_api_param`). If the JSON API still responds with **`pagination.per_page: 100`**, you’ll see a one-time warning — that’s a **website vs API** limitation, not something we can fix without scraping. Client-side vinyl filtering always applies as a backstop.

**Generic seller notes:** Listings whose comments look like shop boilerplate (configurable `data.discogs.generic_note_filter.patterns`) are **dropped** unless the note also contains **item-specific** language (`item_specific_hints`), or **either** sleeve/media is **Mint**, or **preservation** phrases appear (e.g. sealed, shrink, brand new). Tune lists in `grader.yaml`; set `generic_note_filter.enabled: false` to disable.

**Boilerplate stripping:** When `generic_note_filter.strip_boilerplate` is true (default), sentence/line chunks that match those patterns **without** item-specific or preservation cues are **removed** from the saved `text` before drop checks and training — mixed notes keep the useful parts only. Set `strip_boilerplate: false` to store full comments.

**Outputs**

- Raw API pages (resume-safe): `grader/data/raw/discogs/inventory/<seller>/per_<N>/page_*.json` (`N` = requested `inventory_per_page`, e.g. 250).
- Processed JSONL (overwritten each run): `grader/data/processed/discogs_processed.jsonl`

Raw pages are **not** deleted between runs; the ingester skips re-downloading existing `page_*.json`. To force a full re-fetch, remove `grader/data/raw/discogs/inventory/` (or a seller folder) before running again.

## Full pipeline (real data)

After ingestion + harmonization + feature extraction (see `grader/src/pipeline.py`):

```bash
# End-to-end with Discogs only (no eBay API keys); token from .env or env
python -m grader.src.pipeline train --skip-ebay-ingest --baseline-only

# Or: ingest Discogs manually, then train from disk
python -m grader.src.data.ingest_discogs
python -m grader.src.pipeline train --skip-ingest --baseline-only
```

`LabelHarmonizer` loads `ebay_processed.jsonl` if present; if you only ingested Discogs, it merges Discogs-only.

**Thin seller notes:** `preprocessing.description_adequacy` flags rows missing sleeve and/or playable-media cues. With `drop_insufficient_from_training: true`, those rows stay in `preprocessed.jsonl` but are **excluded from train/val/test splits** (so the model trains only on adequate notes). The same excluded rows are written to **`grader/data/splits/test_thin.jsonl`** for evaluation only. Baseline and transformer runs then report metrics on **`test`** (held-out adequate) and **`test_thin`** (thin notes) separately. See `grader/reports/description_adequacy_summary.txt` after preprocess. At inference, `predict` adds `needs_richer_note`, `description_quality_gaps`, and `description_quality_prompts` in each prediction’s `metadata` for UI copy.

Produces calibrated baselines, test metrics, calibration artifacts, and MLflow tracking. **Tracking metadata** is stored in **`grader/experiments/mlflow.db`** (SQLite — avoids the deprecated file-backed tracking backend; see MLflow issue #18534). **Run artifacts** may still appear under `grader/experiments/mlruns/` depending on MLflow’s default artifact root.

To migrate old runs from a legacy `./mlruns` file store, use [mlflow-export-import](https://github.com/mlflow/mlflow-export-import).

## DistilBERT hyperparameter tuning

Presets in `grader/configs/transformer_tune.yaml` merge onto `models.transformer` in `grader.yaml`. Each run writes weights under `grader/artifacts/tuning/<preset_key>/` and appends a row to `grader/reports/transformer_tune_results.csv`.

```bash
# Smoke (1 epoch, no saves)
python -m grader.src.models.transformer_tune --presets partial1_low_lr --dry-run

# Full sweep (long); skip MLflow if you only want the CSV
python -m grader.src.models.transformer_tune --presets all --skip-mlflow

# Promote a winner to the default inference path (overwrites root artifacts)
python -m grader.src.models.transformer_tune --promote partial1_low_lr
```

Then copy the winning numeric settings from the CSV / preset into `grader.yaml` under `models.transformer` for reproducible defaults.

Single training run (no sweep): `python -m grader.src.models.transformer` (optional `--skip-mlflow`).

Tunable in YAML: `learning_rate`, `batch_size`, `dropout`, `weight_decay`, `warmup_ratio`, `unfreeze_top_n_layers`, `early_stopping_patience`, `media_evidence_aux.{enabled,loss_weight}`.

### Misclassification sample for review

```bash
python -m grader.src.eval.export_mispredictions \
  --split test --n 120 --stratify \
  --output grader/reports/mispredictions_sample_test.csv
```

Uses transformer weights under `grader/artifacts/` unless `--artifact-subdir tuning/<preset>` is set. Columns include true vs pred, top-1 prob / top-2 gap, `media_evidence_strength`, and raw text.

## Tests

```bash
pytest grader/tests
```

## HTTP API (monorepo)

`POST /api/condition` on the FastAPI app calls the **baseline** grader (personal-use setup). See root `README.md` for the JSON contract.
