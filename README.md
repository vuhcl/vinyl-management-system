# Vinyl Management System

A monorepo for vinyl collection tooling: **Discogs integration**, **data ingest**, and **three ML components**—recommender, vinyl condition grader, and price estimator—plus a **web interface** to log in with Discogs and sync your data.

---

## Features

| Component | Description |
|-----------|-------------|
| **Web app** | Log in with your Discogs token, sync collection & wantlist to the app, and call ML APIs (recommendations, condition, price). |
| **Recommender** | Hybrid (ALS + content-based) recommendations using your Discogs collection/wantlist and optional AOTY ratings. |
| **Vinyl condition grader** | Predicts sleeve and media condition from seller notes (e.g. Discogs listings). |
| **Price estimator (VinylIQ)** | XGBoost + Discogs stats; FastAPI microservice, optional Chrome extension. |

Shared infrastructure: **Discogs API** client and **AOTY** scraped-data loader used across components.

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

Open **http://127.0.0.1:8000** → [Log in with Discogs](http://127.0.0.1:8000/auth/login) (paste a [personal token](https://www.discogs.com/settings/developers)) → [Sync / Ingest](http://127.0.0.1:8000/ingest) to pull your collection and wantlist into `data/raw/` (repo root).

---

## Project layout

```
vinyl_management_system/
├── configs/base.yaml       # Shared repo config (paths, Discogs, MLflow)
├── recommender/configs/base.yaml  # Recommender pipeline (inherits root; ALS, retrieval)
├── core/                   # Config loader, auth, ingest jobs
├── shared/discogs_api/    # Shared Discogs API client
├── shared/aoty/           # AOTY scraped data loader
├── recommender/            # ML: hybrid recommendations
├── grader/                   # ML: sleeve/media condition grader from notes
├── price_estimator/       # ML: price estimation
├── web/                    # Web UI and API
├── data/raw (Discogs CSVs), recommender/data/processed
└── artifacts/
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
uvicorn web.app.main:app --reload
```

- **Home**: http://127.0.0.1:8000  
- **Login**: http://127.0.0.1:8000/auth/login (paste Discogs token)  
- **Ingest**: http://127.0.0.1:8000/ingest (sync collection & wantlist; requires login)  
- **API docs**: http://127.0.0.1:8000/docs  

After logging in and running ingest, you can call:

- `GET /api/recommendations` — recommendations for the logged-in user (requires recommender pipeline to have been run once).

### Recommender

1. Ingest data (via web or CSV in `data/raw/`: `collection.csv`, `wantlist.csv`, optional `ratings.csv`, `albums.csv`).
2. Train and save artifacts:

```bash
python -m recommender.pipeline --config recommender/configs/base.yaml --data-dir data/raw --processed-dir recommender/data/processed --artifacts-dir artifacts
```

3. Use in code or via web: `GET /api/recommendations` (when logged in).

See **`recommender/README.md`** for details.

### Grader API

Standalone **FastAPI** service for the vinyl condition grader: loads the **MLflow-registered** DistilBERT pyfunc, runs the same **preprocessor + rule engine** as pipeline inference, and serves **`POST /predict`** (JSON in/out). Docker build uses image tag **`vinyl-grader-api:latest`** (see `grader/Dockerfile`).

**Run locally, Docker build/run, GCS credentials, environment variables, `/predict` request/response format, and example curls**: **[grader/serving/README.md](grader/serving/README.md)**.

The monolith web app’s **`POST /api/condition`** uses the **baseline** (TF‑IDF) pipeline instead—see *Vinyl condition grader* below.

### Vinyl condition grader (personal use)

This component (and the `/api/condition` endpoint) is intended for **personal use**. It loads the **baseline** model on the server and applies the rule engine, but it is not hardened for public traffic yet.

#### Run the grader locally (baseline)

```bash
export PYTHONPATH=/path/to/vinyl_management_system
python -m grader.src.pipeline train --baseline-only
```

For inference from the CLI, use:

```bash
python -m grader.src.pipeline predict --text "factory sealed, never opened" --model baseline
```

For mobile/React Native use, call the existing FastAPI endpoint: `POST /api/condition` with JSON `{ "seller_notes": "..." }`.

**Quick synthetic eval (resume / portfolio):** run `python scripts/grader_eval_resume.py` → writes `artifacts/grader_eval_resume.json` (macro-F1, accuracy, ECE on a benchmark aligned with the grader test suite; see file `disclaimer`).

API contract (baseline + rule engine applied):

Request:
```json
{
  "seller_notes": "raw seller notes text",
  "item_id": "optional id echoed back",
  "metadata": { "optional": "free-form metadata for rule signals" }
}
```

Response:
```json
{
  "item_id": "optional id echoed back",
  "predicted_sleeve_condition": "Mint|Near Mint|Excellent|Very Good Plus|Very Good|Good|Poor|Generic",
  "predicted_media_condition":  "Mint|Near Mint|Excellent|Very Good Plus|Very Good|Good|Poor",
  "confidence_scores": {
    "sleeve": { "Mint": 0.1, "Near Mint": 0.4, "...": 0.5 },
    "media":  { "Mint": 0.2, "Near Mint": 0.3, "...": 0.5 }
  },
  "metadata": {
    "source": "unknown|discogs|ebay_jp|user_input",
    "media_verifiable": true,
    "rule_override_applied": true,
    "rule_override_target": "Mint|Poor|Generic|...",
    "contradiction_detected": false
  }
}
```

#### Changes required for a wider release

Before turning this into a public-facing product, I recommend addressing:
- **Auth & rate limiting** on `POST /api/condition` (currently option A / no per-user auth).
- **CORS + mobile-friendly deployment config** (base URL handling, HTTPS, allowed origins).
- **Model lifecycle management** (clear artifact versioning, warmup, and safe concurrent loading).
- **Operational monitoring** (request latency/failures, calibration drift, and rule override frequency).
- **Dataset and label governance** (document label mappings/contradictions; add regression tests for harmonization).
- **CI/CD and reproducible training** (locked dependencies, deterministic splits, artifact checks).

### Price estimator (VinylIQ)

1. Seed demo data and train (see **`price_estimator/README.md`**) or collect real `marketplace/stats` + feature store.
2. Run API: `PYTHONPATH=. uvicorn price_estimator.src.api.main:app --port 8801`
3. Web: set `PRICE_SERVICE_URL` to proxy `GET /api/price/{release_id}` to the microservice, or use in-process `estimate()` without it.
4. Chrome: load unpacked **`vinyliq-extension/`**.

See **`price_estimator/README.md`** and **`vinyliq-extension/README.md`**.

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

The recommender can use **Album of the Year** scraped data for ratings and album metadata.

- **Expected files** (in the dir set by `aoty_scraped.dir` in config):
  - `ratings.csv`: `user_id`, `album_id`, `rating`
  - `albums.csv`: `album_id`, `artist`, `genre`, `year`, `avg_rating`
- If `aoty_scraped.dir` is not set, the recommender falls back to CSVs in `data/raw/`.

Loader API:

```python
from shared.aoty import load_ratings_from_scraped, load_album_metadata_from_scraped
from pathlib import Path
ratings = load_ratings_from_scraped(Path("path/to/scraped"))
albums = load_album_metadata_from_scraped(Path("path/to/scraped"))
```

---

## Subproject docs

| Subproject | README |
|------------|--------|
| Recommender | [recommender/README.md](recommender/README.md) |
| Vinyl condition grader | [grader/README.md](grader/README.md) |
| Price estimator | [price_estimator/README.md](price_estimator/README.md) |
| Web app | [web/README.md](web/README.md) |
| Structure & coordination | [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) |

---

## License

See repository license file.
