# Vinyl Management System

A monorepo for vinyl collection tooling: **Discogs integration**, **data ingest**, and **three ML components**—recommender, vinyl condition grader, and price estimator—plus a **web interface** to log in with Discogs and sync your data.

## Demo

[![VinylIQ seller demo on Discogs (~3 min) — watch on YouTube](demo/vinyliq_demo_playwright/demo_thumbnail.png)](https://youtu.be/04D_E4hLWyE)

End-to-end demo: a seller pastes condition notes into a Discogs sell-listing form, the VinylIQ Chrome extension calls the **grader API** to predict media + sleeve grades and updates the dropdowns, then on the matching release page the same extension calls the **price API** and renders an estimate that varies meaningfully with the predicted condition. Full deploy is on GKE Autopilot — see [`k8s/demo/README.md`](k8s/demo/README.md) for the runbook and [`demo/vinyliq_demo_playwright/RECORDING.md`](demo/vinyliq_demo_playwright/RECORDING.md) for the recording flow.

---

## Features

| Component | Description |
|-----------|-------------|
| **Web app** | Log in with your Discogs token, sync collection & wantlist to the app, and call ML APIs (recommendations, condition, price). |
| **Recommender** | Hybrid (ALS + content-based) recommendations using your Discogs collection/wantlist and optional AOTY ratings; optional learned reranker. |
| **Vinyl condition grader** | Predicts sleeve and media condition from seller notes (e.g. Discogs listings). |
| **Price estimator (VinylIQ)** | Marketplace stats, feature store, gradient-boosting stack; FastAPI microservice with Memorystore (Redis) cache in front of SQLite + Discogs. |
| **VinylIQ Chrome extension** | Two-surface client on Discogs: condition grading on the seller listing form (`/sell/post/*`) plus price estimates on release pages (`/release/*`). |

Shared infrastructure: **`shared.discogs_api`** for Discogs HTTP and **`shared.aoty`** for loading scraped Album of the Year CSVs when present.

---

## Quick start

```bash
# Clone and enter project
cd vinyl_management_system

# Install (from repo root; requires [uv](https://docs.astral.sh/uv/))
uv sync --extra test

# Run the web app (from project root)
uv run uvicorn web.app.main:app --reload
```

Open **http://127.0.0.1:8000** → [Log in with Discogs](http://127.0.0.1:8000/auth/login) (paste a [personal token](https://www.discogs.com/settings/developers)) → [Sync / Ingest](http://127.0.0.1:8000/ingest/) to pull your collection and wantlist into `data/raw/` (repo root).

---

## Project layout

```
vinyl_management_system/
├── configs/base.yaml              # Shared repo config (paths, Discogs, MLflow)
├── recommender/configs/base.yaml  # Recommender (inherits root; ALS, reranker, ingest)
├── core/                          # Config loader, auth, ingest jobs
├── shared/discogs_api/            # Shared Discogs API client
├── shared/aoty/                   # AOTY scraped CSV loader
├── recommender/                   # ML: hybrid recommendations
├── grader/                        # ML: sleeve/media condition from notes
├── price_estimator/               # ML: VinylIQ price API + training
├── web/                           # Web UI and API
├── scrapers/aoty/                 # Placeholder for personal AOTY scrapers (see README there)
├── vinyliq-extension/             # Chrome MV3 → price API
├── data/raw/                      # Discogs CSVs (web-friendly mount)
├── recommender/data/processed/    # Recommender parquet artifacts
└── artifacts/                     # Models, maps, tuning outputs
```

Full layout and how components coordinate: **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)**.

---

## Setup

### Requirements

- **Python 3.12+**
- Discogs [personal access token](https://www.discogs.com/settings/developers) (for API and web login)

### Install

This repo is a **uv workspace** (`[tool.uv.workspace]` in `pyproject.toml`). From the repo root:

```bash
uv sync --extra test
```

That installs `vinyl-shared`, `vinyl-core`, `vinyl-grader[serve]`, `vinyl-recommender`, `vinyl-price-estimator`, and `vinyl-web` together.

**Without uv:** from the repo root, install each workspace package in dependency order (shared first), e.g. `pip install -e ./shared -e ./core -e ./grader -e ./recommender -e ./price_estimator -e ./web -e .`, or use `pip install -e .` after ensuring the root `pyproject.toml` resolves workspace members (pip 23+ with PEP 660).

**Minimal environment** (e.g. grader API only): `uv sync --package vinyl-grader --extra serve` (see [`grader/Dockerfile`](grader/Dockerfile)).

**Gradient boosting (price estimator):** `uv sync --package vinyl-price-estimator --extra lgb` (optional `lightgbm`).

### Config

Edit **`configs/base.yaml`** for shared paths and tokens. For the **recommender** pipeline, edit **`recommender/configs/base.yaml`** (it **inherits** the root file via `inherits: configs/base.yaml`) and set:

- **Discogs**: `discogs.use_api`, `discogs.usernames`, token via env or YAML.
- **AOTY**: `aoty_scraped.dir` (e.g. `recommender/data/aoty_scraped`) or CSVs in `data/raw/`.

---

## Usage

### Web app

From the project root:

```bash
uv run uvicorn web.app.main:app --reload
```

- **Home**: http://127.0.0.1:8000  
- **Login**: http://127.0.0.1:8000/auth/login (paste Discogs token)  
- **Ingest**: http://127.0.0.1:8000/ingest/ (sync collection & wantlist; requires login)  
- **API docs**: http://127.0.0.1:8000/docs  

Routes and env vars: **[web/README.md](web/README.md)**.

After logging in and running ingest, you can call `GET /api/recommendations` once the recommender pipeline has produced artifacts (see below).

### Recommender

Stage-1 recall is **ALS over the full item catalog**; an optional **learned reranker** reorders ALS top-N candidates using content and optional Discogs-side features. Ingest data via the web or place CSVs under `data/raw/` (`collection.csv`, `wantlist.csv`, optional `ratings.csv`, `albums.csv`), then train:

```bash
uv run python -m recommender.pipeline \
  --config recommender/configs/base.yaml \
  --data-dir data/raw \
  --processed-dir recommender/data/processed \
  --artifacts-dir artifacts
```

The full workflow—ALS tuning, hit-rate calibration for `candidate_top_n`, reranker sweep, Discogs↔AOTY artifact scripts—is documented in **[recommender/README.md](recommender/README.md)**.

### Grader

Two surfaces:

- **Web `POST /api/condition`** — baseline (TF‑IDF) + rule engine for personal use; not hardened for public traffic. Train baseline, Discogs ingest, synthetic eval, and API details: **[grader/README.md](grader/README.md)**.
- **Standalone FastAPI (`grader.serving`)** — loads an **MLflow-registered** DistilBERT pyfunc, same preprocessor + rule engine, **`POST /predict`**. Docker image **`vinyl-grader-api:latest`**: **[grader/serving/README.md](grader/serving/README.md)**.

### Price estimator (VinylIQ)

Dedicated package: **SQLite** marketplace + feature store, **Discogs dump ingest**, tuning/training (`sale_floor_blend` / `sale_floor` labels, residual vs median target), **`POST /estimate`** FastAPI app, and optional **Chrome extension**. Botasaurus-based **Discogs** collectors live under **`price_estimator/scripts/`** (not shared with other packages).

1. Follow **[price_estimator/README.md](price_estimator/README.md)** for collectors, feature store, and training.  
2. Run the API: `PYTHONPATH=. uv run uvicorn price_estimator.src.api.main:app --port 8801`  
3. Web: set **`PRICE_SERVICE_URL`** to proxy `GET /api/price/{release_id}` to the microservice, or omit for in-process `estimate()`.  
4. Extension: load unpacked **`vinyliq-extension/`** — **[vinyliq-extension/README.md](vinyliq-extension/README.md)**.

---

## Discogs API (shared)

All components use the same **`shared.discogs_api`** package:

```python
from shared.discogs_api import (
    DiscogsClient,
    get_user_collection,
    get_user_wantlist,
)

# Uses env DISCOGS_USER_TOKEN
df = get_user_collection("username")
want = get_user_wantlist("username")

# Or with explicit token
client = DiscogsClient(user_token="...")
df = client.collection_to_dataframe("username")
```

---

## AOTY scraped data

The recommender can use **Album of the Year** user ratings and album metadata for content features. Scrapers are **not committed** here; the canonical place to keep personal Botasaurus (or other) scripts is **`scrapers/aoty/`**. Write `ratings.csv` and `albums.csv` into the directory configured as `aoty_scraped.dir` (often `recommender/data/aoty_scraped/`). Column contract and wiring: **[scrapers/aoty/README.md](scrapers/aoty/README.md)** and **[recommender/README.md](recommender/README.md)** (*Data sources*).

---

## Subproject docs

| Subproject | README |
|------------|--------|
| Recommender | [recommender/README.md](recommender/README.md) |
| Vinyl condition grader | [grader/README.md](grader/README.md) |
| Grader MLflow API | [grader/serving/README.md](grader/serving/README.md) |
| Price estimator (VinylIQ) | [price_estimator/README.md](price_estimator/README.md) |
| VinylIQ Chrome extension | [vinyliq-extension/README.md](vinyliq-extension/README.md) |
| Web app | [web/README.md](web/README.md) |
| AOTY scrapers (personal) | [scrapers/aoty/README.md](scrapers/aoty/README.md) |
| GKE demo deploy runbook | [k8s/demo/README.md](k8s/demo/README.md) |
| Demo automation (Playwright) | [demo/vinyliq_demo_playwright/README.md](demo/vinyliq_demo_playwright/README.md) |
| Demo golden file | [grader/demo/README.md](grader/demo/README.md) |
| Structure & coordination | [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) |

---

## License

See repository license file.
