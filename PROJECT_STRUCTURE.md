# Vinyl Management System – Project Structure

This document describes the **master project structure** that houses and coordinates the three ML components and the web interface for Discogs integration and data ingest.

---

## High-level layout

```
vinyl_management_system/
├── README.md
├── PROJECT_STRUCTURE.md          # This file
├── pyproject.toml
├── requirements.txt
├── configs/
│   └── base.yaml                 # Shared config (paths, Discogs, AOTY, recommender params)
│
├── core/                          # Shared orchestration & contracts
│   ├── __init__.py
│   ├── config.py                  # Load/merge YAML; resolve paths
│   ├── auth.py                    # Discogs token storage (web sessions)
│   └── jobs.py                    # Ingest jobs: Discogs → data/raw
│
├── discogs_api/                   # Shared Discogs API client
│   ├── __init__.py
│   └── client.py
│
├── aoty/                          # AOTY scraped data loader
│   ├── __init__.py
│   └── loader.py
│
├── recommender/                   # ML component 1: hybrid recommendations
│   ├── src/
│   │   ├── data/                  # ingest (Discogs + AOTY + CSV), preprocess
│   │   ├── features/               # build_matrix, content_features
│   │   ├── models/                 # ALS, content_model, hybrid
│   │   └── evaluation/
│   ├── pipeline.py                 # run_pipeline, recommend, load_pipeline_artifacts
│   └── README.md
│
├── nlp_condition_classifier/       # ML component 2: sleeve/media condition from notes
│   ├── src/
│   │   ├── data/
│   │   ├── features/
│   │   ├── models/
│   │   └── evaluation/
│   ├── configs/base.yaml
│   ├── pipeline.py
│   └── README.md
│
├── price_estimator/               # ML component 3: price estimation (stub)
│   ├── src/
│   │   └── pipeline.py             # estimate(), run_pipeline() stub
│   ├── configs/base.yaml
│   └── README.md
│
├── web/                           # Web interface
│   ├── app/
│   │   ├── main.py                 # FastAPI app, middleware, routes
│   │   └── routers/
│   │       ├── auth.py             # Login (Discogs token), /auth/me
│   │       ├── ingest.py           # POST /ingest/sync, /ingest/full
│   │       └── ml.py               # /api/recommendations, /api/condition, /api/price
│   ├── requirements.txt
│   └── README.md
│
├── data/                           # Shared data (default)
│   ├── raw/                        # collection.csv, wantlist.csv, ratings.csv, albums.csv
│   └── processed/
├── artifacts/                      # ML artifacts (recommender, NLP, price_estimator)
└── scrapers/                       # Optional; scrapers live here or elsewhere
```

---

## Coordination

| Concern | Where it lives | How it’s used |
|--------|----------------|----------------|
| **Config** | `configs/base.yaml` + `core/config.py` | All components and web app load config via `core.config.load_config()`. Paths are resolved relative to project root. |
| **Discogs** | `discogs_api/` | Recommender ingest, web ingest, and (future) price_estimator use the same client. Token comes from env or from web login (stored in `core.auth`). |
| **AOTY data** | `aoty/` | Recommender reads ratings/albums from a scraped directory or CSV fallback. |
| **Ingest** | `core/jobs.py` | Web app calls `run_discogs_ingest(username, token)` or `run_full_ingest(...)` after login; writes to `data/raw/`. |
| **Auth** | `core/auth.py` + `web/.../auth.py` | User pastes Discogs token → app verifies with Discogs, stores token per username, sets cookie. Ingest and API use stored token. |

---

## Data flow

1. **User logs in** (web): submits Discogs token → `/auth/token` → token stored, cookie set.
2. **User triggers ingest** (web): POST `/ingest/sync` or `/ingest/full` → `core.jobs` fetches collection/wantlist (and optionally AOTY) → writes CSVs to `data/raw/`.
3. **Recommender**: Run `python -m recommender.pipeline` (reads `data/raw/`, writes `data/processed/` and `artifacts/`). Web API `/api/recommendations` loads artifacts and returns recommendations for the logged-in user.
4. **NLP classifier**: Run from project root (see `nlp_condition_classifier/README.md`). Web API `/api/condition` can call it for seller-notes → condition (stub in place).
5. **Price estimator**: Stub; pipeline and `/api/price/{release_id}` are in place for future implementation.

---

## Running the stack

- **Install**: From project root, `pip install -r requirements.txt` and `pip install -r web/requirements.txt`.
- **Web app**: `uvicorn web.app.main:app --reload` (from project root, so `core`, `discogs_api`, `recommender`, etc. are on `PYTHONPATH`).
- **Ingest**: Use the web UI (Login → Ingest) or call POST `/ingest/sync` with a logged-in session.
- **Recommender**: After ingest, run `python -m recommender.pipeline --config configs/base.yaml`.
- **NLP classifier**: `python -m nlp_condition_classifier.src.pipeline --phase baseline`.
- **Price estimator**: `python -m price_estimator.src.pipeline` (stub).

---

## Adding a new ML component

1. Add a top-level package (e.g. `my_component/`) with `src/`, `configs/`, and a `pipeline` or main entrypoint.
2. Use `core.config.load_config()` for paths and shared settings; use `discogs_api` for any Discogs data.
3. Expose an HTTP API under `web/app/routers/ml.py` (or a new router) and document in this file.
