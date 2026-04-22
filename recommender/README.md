# Recommender (subproject)

Hybrid vinyl recommendation system: **Discogs** (shared API) + **AOTY scraped data** + ALS + content-based ranking.

This subproject is part of **vinyl_management_system**. It uses:

- **Shared Discogs API** (`shared.discogs_api`) for collection and wantlist when **`recommender/configs/base.yaml`** (or merged root config) has `discogs.use_api` and usernames/token.
- **AOTY scraped data** (`shared.aoty`) when `aoty_scraped.dir` points to your scraped data directory.
- **CSV fallback** in `data/raw/` when Discogs or AOTY is not configured.

---

## Repository structure

```
recommender/
├── configs/
│   └── base.yaml   # Pipeline YAML (inherits repo `configs/base.yaml`)
├── src/
│   ├── data/       # ingest.py (Discogs API + AOTY scraped + CSV), preprocess.py
│   ├── features/   # build_matrix.py, content_features.py
│   ├── models/     # als.py, content_model.py, hybrid.py, reranker.py
│   ├── evaluation/ # metrics.py, evaluate.py
│   ├── retrieval/  # candidates.py — metadata helpers for reranker features
│   └── pipeline.py # orchestration + recommend API
└── README.md
```

**Config:** `recommender/configs/base.yaml` holds ALS, evaluation, reranker, and Discogs ingest settings; it merges with **`configs/base.yaml`** (paths, tokens). Data dirs: `data/raw`, `data/processed`, `artifacts`.

---

## Data sources

| Data        | Primary source              | Fallback        |
|------------|-----------------------------|-----------------|
| Collection | Discogs API (shared client) | `data/raw/collection.csv` |
| Wantlist   | Discogs API (shared client) | `data/raw/wantlist.csv` |
| Ratings    | AOTY scraped data directory | `data/raw/ratings.csv` |
| Albums     | AOTY scraped data directory | `data/raw/albums.csv` |

Set in **`recommender/configs/base.yaml`** (or override `--config`):

- **Discogs**: `discogs.use_api: true`, `discogs.usernames: ["your_username"]`, and `DISCOGS_USER_TOKEN` in env (or `DISCOGS_TOKEN`), or put either key in the **repo-root `.env`** — `python -m recommender.pipeline` and `scripts/smoke_recommender_ingest.py` load `.env` automatically (via `shared.project_env`). You can also set `discogs.token` in YAML instead of env.
- **Discogs → AOTY (recommended)**: incremental Mongo-backed pipeline (candidate masters from your collection ∪ wantlist only — **no** full AOTY catalog Discogs search):
  1. `scripts/build_discogs_master_to_aoty_artifact.py` — release→master + master→AOTY (Mongo upserts + `artifacts/discogs_master_to_aoty.json`).
  2. `scripts/build_discogs_release_to_aoty_artifact.py` — compose `artifacts/discogs_release_to_aoty.json` + Mongo `discogs_release_aoty`.
  Then set `discogs.release_to_aoty_map_path` / `skip_live_discogs_aoty_mapping` in **`recommender/configs/base.yaml`**.
- **Legacy monolithic script**: `scripts/build_discogs_aoty_release_map.py` (full old flow; avoid for large catalogs).
- **AOTY**: `aoty_scraped.dir: "data/aoty_scraped"` (or path to your scraped output).

### Reranker (optional second stage)

Stage-1 recall is always **ALS over the full item matrix** — no metadata-gated candidate pool.
An optional learned reranker (second stage) takes the ALS top-N candidates and reorders
them using content-based features derived from `RetrievalMetadata`:

- **Features** (6, all personalized or bounded): `als_score_z` (per-user z-scored ALS),
  `als_rank_inv` (1 / (1 + rank)), `genre_jaccard`, `artist_match`, `year_distance`,
  `item_avg_rating`. Raw popularity and distinct-users counts are deliberately
  excluded — they fed a "popular ⇒ irrelevant" feedback loop with ALS-top hard negatives.
- **Models**: `linear` (StandardScaler + logistic regression, safer baseline) or
  `pointwise` (histogram gradient-boosted tree).
- **Hard negatives**: mined from a mid-rank ALS band, not the top, controlled by
  `reranker.hard_negative_skip_top_frac` (default 0.1). Skipping the very top prevents
  teaching the model that high-ALS items are negatives.

Enable in **`recommender/configs/base.yaml`** under `reranker.enabled: true`, then run the
full pipeline to train and save the reranker bundle (`artifacts/reranker.pkl`).

**Serving:** `artifacts/retrieval_serving.pkl` stores `RetrievalMetadata` needed by the
reranker at serving time. If this file or `reranker.pkl` was produced by a pre-cleanup run
(legacy feature names), `load_pipeline_artifacts` / `load_reranker_bundle` logs a clear
rebuild message and falls back gracefully to full-catalog ALS without the reranker.

Hybrid blending (`content_sim` + `alpha` < 1) also uses full-catalog ALS for stage-1.

### Discogs ↔ AOTY ID matching (API limits)

When Discogs collection/wantlist is loaded from the API, `ingest_all` remaps
Discogs **release** IDs to AOTY **album** IDs. That uses extra Discogs HTTP
calls (search + per-release lookups). To stay within rate limits:

- Responses are cached under **`{data_dir}/.discogs_cache/`** (gitignored).
- Requests are spaced by default (~**1.05s** between calls). Tune in code via
  `DiscogsMatchConfig.min_request_interval_s` in
  `recommender/src/data/discogs_aoty_id_matching.py` (or wire from config
  later).

---

## Setup and run

From **project root** (vinyl_management_system):

```bash
uv sync --extra test   # or: pip install -e . from repo root (workspace)
python -m recommender.pipeline --config recommender/configs/base.yaml --data-dir data/raw --processed-dir data/processed --artifacts-dir artifacts
```

Use `--skip-ingest` to reuse existing processed data.

---

## Training workflow

Run from project root in this order:

1. **Tune ALS hyperparameters** — pick best `als.*` by NDCG@10 on leave-one-out.
2. **Train final ALS** on full processed data (reranker disabled).
3. **Measure ALS hit rate** at candidate cutoffs to calibrate `reranker.candidate_top_n`.
4. **Sweep reranker hyperparameters** — pick `model_type` / `candidate_top_n` / `hard_negative_ratio`; write winner to `base.yaml`.
5. **Re-run final training with reranker enabled** once `hit_rate@candidate_top_n >= 0.8` and a sweep winner exists.

### 1) ALS tuning

Stage-1 is full-catalog ALS (exhaustive dot product over all items). Alpha is
fixed at 10; the sweep varies factors, regularization, and iterations
(3x3x3 = 27 runs).

```bash
uv run python -u scripts/tune_recommender_als.py \
  --config recommender/configs/base.yaml \
  --sample-n 3000000 \
  --factors "64,128,256" \
  --regularization "0.01,0.1,1.0" \
  --alpha "10" \
  --iterations "10,15,20" \
  --max-runs 27 \
  --random-state 42 2>&1 | tee artifacts/tune_als.log
```

Best run is written to `artifacts/als_tuning_best.json`. Update the `als.*`
block in `recommender/configs/base.yaml` with the winning values before
continuing.

### 2) Final training run (artifact build)

Keep `reranker.enabled: false` in the config for this pass so the ALS
hit-rate diagnostic in step 3 can be run on a clean ALS-only build.

```bash
uv run python -m recommender.pipeline \
  --config recommender/configs/base.yaml \
  --data-dir data/raw \
  --processed-dir data/processed \
  --artifacts-dir artifacts \
  --skip-ingest
```

Drop `--skip-ingest` when you want fresh ingest/preprocess from raw sources.

Set `MLFLOW_TRACKING_URI` in `.env` (see `.env.template`) to route runs to a remote or local tracking server. Pass `--no-mlflow` to disable tracking entirely.

### 3) Hit-rate diagnostic (calibrate `reranker.candidate_top_n`)

Measures `hit_rate@N` for full-catalog ALS top-N at several cutoffs so you
can pick the smallest N that clears the reranker's recall floor. The script
trains its own ALS on a leave-one-out train split (held-out item is unseen
during fit), so the measurement is unbiased.

```bash
uv run python -u scripts/measure_als_hit_rate.py \
  --config recommender/configs/base.yaml \
  --processed-dir data/processed \
  --sample-n 3000000 \
  --cutoffs "50,100,200,500,1000,2000,5000" \
  --random-state 42 \
  --artifacts-dir artifacts
```

Inspect `artifacts/als_hit_rate.json`. Pick the smallest N where
`hit_rate@N >= 0.8`, then set `reranker.candidate_top_n: <N>` in
`recommender/configs/base.yaml`.

### 4) Reranker sweep (pick `model_type` / `candidate_top_n` / `hard_negative_ratio`)

ALS is trained once and shared across all grid points, so this is cheap
relative to the ALS tuning sweep. Defaults are a 2 x 3 x 3 = 18-run grid.
If step 3 gave you a calibrated `N` very different from 200/500/1000, widen
the `--candidate-top-n` grid to straddle it.

```bash
uv run python -u scripts/sweep_reranker.py \
  --config recommender/configs/base.yaml \
  --processed-dir data/processed \
  --sample-n 3000000 \
  --k 10 \
  --model-types "linear,pointwise" \
  --candidate-top-n "200,500,1000" \
  --hard-negative-ratio "0.0,0.3,0.7" \
  --hard-negative-skip-top-frac 0.1 \
  --random-state 42 \
  --artifacts-dir artifacts 2>&1 | tee artifacts/sweep_reranker.log
```

Results land in `artifacts/sweep_reranker.json`. Each run records both the
reranked `ndcg@10`, `ndcg@10_pop_head`, `ndcg@10_pop_tail` **and** the
ALS-only baselines (`als_only_*`) from the same fitted ALS, so you can
compare them apples-to-apples.

**Winner rule:** lowest `|als_only_ndcg@10_pop_head - ndcg@10_pop_head|`
among runs that satisfy both `ndcg@10 > als_only_ndcg@10` and
`ndcg@10_pop_tail > als_only_ndcg@10_pop_tail`. If no config passes, keep
`reranker.enabled: false` — the reranker is not yet helping. Otherwise,
copy the winner's `rr_cfg` fields into the `reranker:` block in
`recommender/configs/base.yaml` before step 5.

Note: at `hard_negative_ratio=0.0` the `skip_top_frac` knob is a no-op; that
row is the intentional "easy negatives only" baseline.

### 5) Final training with reranker

Flip `reranker.enabled: true` in `recommender/configs/base.yaml` and re-run
the step 2 command. The pipeline trains and saves the reranker bundle
(`artifacts/reranker.pkl`) alongside the ALS artifacts.

### Overnight runs (optional)

Use `caffeinate` on macOS to prevent sleep:

```bash
caffeinate -dimsu uv run python -u scripts/tune_recommender_als.py ... 2>&1 | tee artifacts/tune_overnight.log
```

---

## Get recommendations

```python
from recommender.pipeline import recommend, run_pipeline

# After run_pipeline(), use returned dict as pipeline_artifacts
out = recommend("user_id", pipeline_artifacts, top_k=10, exclude_owned=True, alpha=0.7)
# {"user_id": "...", "recommendations": [{"album_id": "...", "score": 0.87, "rank": 1}, ...]}
```

---

## Config summary

- `interaction_weights`, `als`, `hybrid`, `evaluation`, `reranker`, `recommendation` (see **`recommender/configs/base.yaml`**).
- `discogs`: use_api, usernames, token.
- `aoty_scraped`: dir, ratings_file, albums_file.
