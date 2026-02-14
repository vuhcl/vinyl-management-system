# Recommender (subproject)

Hybrid vinyl recommendation system: **Discogs** (shared API) + **AOTY scraped data** + ALS + content-based ranking.

This subproject is part of **vinyl_management_system**. It uses:

- **Shared Discogs API** (`discogs_api`) for collection and wantlist when `configs/base.yaml` has `discogs.use_api` and usernames/token.
- **AOTY scraped data** (`aoty` loader) when `aoty_scraped.dir` points to your scraped data directory.
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
