# Vinyl Management System – Project Structure

This document describes the **master project structure** that houses and coordinates the three ML components and the web interface for Discogs integration and data ingest.

---

## Workspace packages (uv)

| Package | Path | Role |
|--------|------|------|
| **vinyl-core** | `core/` | Repo-root config (`core.config.load_config`), Discogs token storage, ingest jobs |
| **vinyl-shared** | `shared/` | Discogs client, AOTY loader |
| **vinyl-recommender** | `recommender/` | Hybrid recommender pipeline |
| **vinyl-grader** | `grader/` | Condition grader + optional FastAPI serving (`grader.serving`) |
| **vinyl-price-estimator** | `price_estimator/` | VinylIQ training + `/estimate` API + local `estimate()` |
| **vinyl-web** | `web/` | Dashboard FastAPI app; routers use `web.app.deps.get_current_username` (session middleware sets `request.state.username` from cookie) |

Root `pyproject.toml` lists these as `[tool.uv.workspace]` members.

### Config roots

- **Repository root:** `configs/base.yaml` — shared paths, Discogs token placeholder, MLflow. Loaded with `core.config.load_config()` (default path). Used by `core.jobs` and the web app for artifact paths.
- **Recommender:** `recommender/configs/base.yaml` — ALS, evaluation, two-stage `retrieval`, ingest-related `discogs` / `aoty_scraped`. Uses `inherits: configs/base.yaml` so `core.config.load_config("recommender/configs/base.yaml")` merges both. `recommender.pipeline` defaults to this file.
- **Price estimator:** `price_estimator/configs/base.yaml` is loaded by `price_estimator.src.pipeline` using the package root (not the repo root). Override via env `VINYLIQ_CONFIG` where supported.

---

## High-level layout

```
vinyl_management_system/
├── README.md
├── PROJECT_STRUCTURE.md          # This file
├── pyproject.toml
├── uv.lock                    # Pinned deps (uv); optional for pip-only workflows
├── configs/
│   └── base.yaml                 # Shared config (paths, Discogs, MLflow)
│
├── core/                          # Shared orchestration & contracts
│   ├── __init__.py
│   ├── config.py                  # Load/merge YAML; resolve paths
│   ├── auth.py                    # Discogs token storage (web sessions)
│   └── jobs.py                    # Ingest jobs: Discogs → data/raw (repo root)
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
│   ├── configs/
│   │   └── base.yaml              # Recommender YAML (inherits ../configs/base.yaml)
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
├── price_estimator/               # ML component 3: VinylIQ price API + training
│   ├── data/cache/, feature_store.sqlite
│   ├── src/
│   │   ├── api/                    # FastAPI microservice (/estimate, /health)
│   │   ├── inference/              # orchestration
│   │   ├── features/               # vinyliq_features
│   │   ├── models/                 # xgb_vinyliq, condition_adjustment
│   │   ├── storage/                # SQLite marketplace + feature store
│   │   ├── training/               # train_vinyliq
│   │   └── pipeline.py             # estimate(), run_pipeline → train
│   ├── scripts/                    # collect stats, build_feature_store, seed_demo
│   ├── configs/base.yaml
│   └── README.md
├── vinyliq-extension/             # Chrome extension (MV3) → price API
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
| **Config** | `configs/base.yaml`, `recommender/configs/base.yaml`, `price_estimator/configs/base.yaml` + `core/config.py` | `core.config.load_config()` supports `inherits` and resolves paths under the **repo root**. Recommender training uses **`recommender/configs/base.yaml`**. VinylIQ uses `price_estimator/configs/`. |
| **Discogs** | `shared/discogs_api/` | Recommender ingest, web ingest, and (future) price_estimator use the same client. Token comes from env or from web login (stored in `core.auth`). |
| **AOTY data** | `shared/aoty/` | Recommender reads ratings/albums from a scraped directory or CSV fallback. |
| **Ingest** | `core/jobs.py` | Web app calls `run_discogs_ingest(username, token)` or `run_full_ingest(...)` after login; writes to `data/raw/` (repo root). |
| **Auth** | `core/auth.py` + `web/.../auth.py` | User pastes Discogs token → app verifies with Discogs, stores token per username, sets cookie. Ingest and API use stored token. |

---

## Data flow

1. **User logs in** (web): submits Discogs token → `/auth/token` → token stored, cookie set.
2. **User triggers ingest** (web): POST `/ingest/sync` or `/ingest/full` → `core.jobs` fetches collection/wantlist (and optionally AOTY) → writes CSVs to `data/raw/`.
3. **Recommender**: Run `python -m recommender.pipeline` (reads `data/raw/`, writes `recommender/data/processed/` and `artifacts/`). Web API `/api/recommendations` loads artifacts and returns recommendations for the logged-in user.
4. **Vinyl condition grader**: Run from project root (see `grader/README.md`). Web API `/api/condition` calls it for seller-notes → condition.
5. **VinylIQ / price estimator**: FastAPI in `price_estimator/src/api`; web `/api/price/...` proxies when `PRICE_SERVICE_URL` is set, else in-process `estimate()`. Chrome: `vinyliq-extension/`.

---

## Running the stack

- **Install**: From project root, `uv sync --extra test` (see root `README.md`).
- **Web app**: `uvicorn web.app.main:app --reload` (from project root, so `core`, `shared`, `recommender`, etc. are on `PYTHONPATH`).
- **Ingest**: Use the web UI (Login → Ingest) or call POST `/ingest/sync` with a logged-in session.
- **Recommender**: After ingest, run `python -m recommender.pipeline --config recommender/configs/base.yaml`.
- **Vinyl condition grader**: `python -m grader.src.pipeline train --baseline-only`.
- **Price estimator**: `uvicorn price_estimator.src.api.main:app --port 8801` (see `price_estimator/README.md`); train: `python -m price_estimator.src.training.train_vinyliq`.

---

## Adding a new ML component

1. Add a top-level package (e.g. `my_component/`) with `src/`, `configs/`, and a `pipeline` or main entrypoint.
2. Use `core.config.load_config()` for paths and shared settings; use `shared.discogs_api` for any Discogs data.
3. Expose an HTTP API under `web/app/routers/ml.py` (or a new router) and document in this file.
