# Recommender (subproject)

Hybrid vinyl recommendation system: **Discogs** (shared API) + **AOTY scraped data** + ALS + content-based ranking.

This subproject is part of **vinyl_management_system**. It uses:

- **Shared Discogs API** (`shared.discogs_api`) for collection and wantlist when `configs/base.yaml` has `discogs.use_api` and usernames/token.
- **AOTY scraped data** (`shared.aoty`) when `aoty_scraped.dir` points to your scraped data directory.
- **CSV fallback** in `data/raw/` when Discogs or AOTY is not configured.

---

## Repository structure

```
recommender/
├── src/
│   ├── data/       # ingest.py (Discogs API + AOTY scraped + CSV), preprocess.py
│   ├── features/   # build_matrix.py, content_features.py
│   ├── models/     # als.py, content_model.py, hybrid.py
│   ├── evaluation/ # metrics.py, evaluate.py
│   ├── retrieval/  # candidates.py — two-stage metadata pools
│   └── pipeline.py # orchestration + recommend API
└── README.md
```

Config and paths live at project root: `configs/base.yaml`, `data/raw`, `data/processed`, `artifacts`.

---

## Data sources

| Data        | Primary source              | Fallback        |
|------------|-----------------------------|-----------------|
| Collection | Discogs API (shared client) | `data/raw/collection.csv` |
| Wantlist   | Discogs API (shared client) | `data/raw/wantlist.csv` |
| Ratings    | AOTY scraped data directory | `data/raw/ratings.csv` |
| Albums     | AOTY scraped data directory | `data/raw/albums.csv` |

Set in `configs/base.yaml`:

- **Discogs**: `discogs.use_api: true`, `discogs.usernames: ["your_username"]`, and `DISCOGS_USER_TOKEN` in env (or `discogs.token`).
- **AOTY**: `aoty_scraped.dir: "data/aoty_scraped"` (or path to your scraped output).

### Two-stage retrieval (optional)

Evaluation and smoke scripts can **restrict ALS scoring** to a candidate pool built from
`albums.parquet` metadata (no test leakage: pools use **train** interactions only):

- **Genre expansion**: albums sharing any genre with the user’s train albums.
- **Same-artist expansion**: other albums by those artists.
- **Quality floors**: `min_avg_rating`, `min_train_count`, `min_distinct_users`,
  `min_rating_rows` (rating-source rows only, needs `source` on interactions),
  optional `min_priority_score` (from album `priority_score`).
- **Year band**: quantiles of the user’s train-album `year` values, always expanded
  to include min/max train year; optional `year_window_years` slack. Albums with
  `year == 0` are not excluded by the band. **`release_date`** is stored on
  albums (Mongo/CSV) for traceability; filtering uses integer **`year`**.
- **Cap**: `max_candidates` (sort: train count, then priority, distinct users,
  rating rows, year).

Enable in YAML under `retrieval.enabled: true` (see `configs/smoke_pipeline.yaml`) or:

```bash
python scripts/smoke_recommender_als_eval_only.py --two-stage --processed-dir data/processed ...
```

Metrics include `candidate_relevant_hit_rate` (fraction of test users whose held-out
item appears in the candidate pool) and `two_stage_candidate_nonempty_rate`.

Set `retrieval.min_candidate_relevant_hit_rate` (e.g. `0.15`) to emit a **warning** when
the measured rate falls below that threshold; set `fail_on_low_candidate_hit_rate: true`
to **raise** instead. When below threshold, metrics include
`candidate_retrieval_hit_rate_below_min: 1.0`.

**Serving:** After training, `artifacts/retrieval_serving.pkl` stores the same
metadata used in eval (when `save_for_serving: true` and albums exist).
`recommend()` uses two-stage candidate ALS by default when that file is loaded
(`load_pipeline_artifacts`); set `use_candidate_retrieval=False` to force full-catalog
`recommend()`. Hybrid blending (`content_sim` + `alpha` < 1) still uses the full
catalog path.

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
pip install -r requirements.txt
python -m recommender.pipeline --config configs/base.yaml --data-dir data/raw --processed-dir data/processed --artifacts-dir artifacts
```

Use `--skip-ingest` to reuse existing processed data.

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

- `interaction_weights`, `als`, `hybrid`, `evaluation`, `recommendation` (see `configs/base.yaml`).
- `discogs`: use_api, usernames, token.
- `aoty_scraped`: dir, ratings_file, albums_file.
