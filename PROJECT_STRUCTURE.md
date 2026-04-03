# Vinyl Management System – Project Structure

This document describes the **master project structure** that houses and coordinates the three ML components and the web interface for Discogs integration and data ingest.

---

## High-level layout

```
vinyl_management_system/
├── README.md
├── PROJECT_STRUCTURE.md          # This file
├── pyproject.toml
├── uv.lock                    # Pinned deps (uv); optional for pip-only workflows
├── configs/
│   └── base.yaml                 # Shared config (paths, Discogs, AOTY, recommender params)
│
├── core/                          # Shared orchestration & contracts
│   ├── __init__.py
│   ├── config.py                  # Load/merge YAML; resolve paths
│   ├── auth.py                    # Discogs token storage (web sessions)
│   └── jobs.py                    # Ingest jobs: Discogs → data/raw
│
├── shared/discogs_api/           # Shared Discogs API client
│   ├── __init__.py
│   └── client.py
│
├── shared/aoty/                  # AOTY scraped data loader
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
├── grader/                        # ML component 2: sleeve/media condition grader from notes
│   ├── src/
│   │   ├── data/
│   │   ├── features/
│   │   ├── models/
│   │   └── evaluation/
│   ├── configs/base.yaml
│   ├── pipeline.py
│   └── README.md
│
├── price_estimator/               # ML component 3: price estimation
│   ├── data/raw/                   # sales.csv, metadata.csv
│   ├── src/
│   │   ├── data/                   # ingest, preprocess
│   │   ├── features/               # historical_price, condition_features, embeddings
│   │   ├── models/                  # baseline (linear), gradient_boosting (LightGBM)
│   │   ├── evaluation/             # metrics, prediction_interval
│   │   └── pipeline.py             # run_pipeline(), estimate()
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
│   ├── pyproject.toml              # vinyl-web (workspace member)
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
| **Discogs** | `shared/discogs_api/` | Recommender ingest, web ingest, and (future) price_estimator use the same client. Token comes from env or from web login (stored in `core.auth`). |
| **AOTY data** | `shared/aoty/` | Recommender reads ratings/albums from a scraped directory or CSV fallback. |
| **Ingest** | `core/jobs.py` | Web app calls `run_discogs_ingest(username, token)` or `run_full_ingest(...)` after login; writes to `data/raw/`. |
| **Auth** | `core/auth.py` + `web/.../auth.py` | User pastes Discogs token → app verifies with Discogs, stores token per username, sets cookie. Ingest and API use stored token. |

---

## Data flow

1. **User logs in** (web): submits Discogs token → `/auth/token` → token stored, cookie set.
2. **User triggers ingest** (web): POST `/ingest/sync` or `/ingest/full` → `core.jobs` fetches collection/wantlist (and optionally AOTY) → writes CSVs to `data/raw/`.
3. **Recommender**: Run `python -m recommender.pipeline` (reads `data/raw/`, writes `data/processed/` and `artifacts/`). Web API `/api/recommendations` loads artifacts and returns recommendations for the logged-in user.
4. **Vinyl condition grader**: Run from project root (see `grader/README.md`). Web API `/api/condition` calls it for seller-notes → condition.
5. **Price estimator**: Stub; pipeline and `/api/price/{release_id}` are in place for future implementation.

---

## Running the stack

- **Install**: From project root, `uv sync --extra test` (see root `README.md`).
- **Web app**: `uvicorn web.app.main:app --reload` (from project root, so `core`, `shared`, `recommender`, etc. are on `PYTHONPATH`).
- **Ingest**: Use the web UI (Login → Ingest) or call POST `/ingest/sync` with a logged-in session.
- **Recommender**: After ingest, run `python -m recommender.pipeline --config configs/base.yaml`.
- **Vinyl condition grader**: `python -m grader.src.pipeline train --baseline-only`.
- **Price estimator**: `python -m price_estimator.src.pipeline --phase baseline` (or `--phase gradient_boosting`).

---

## Adding a new ML component

1. Add a top-level package (e.g. `my_component/`) with `src/`, `configs/`, and a `pipeline` or main entrypoint.
2. Use `core.config.load_config()` for paths and shared settings; use `shared.discogs_api` for any Discogs data.
3. Expose an HTTP API under `web/app/routers/ml.py` (or a new router) and document in this file.
