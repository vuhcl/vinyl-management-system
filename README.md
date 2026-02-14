# Vinyl Management System

A monorepo for vinyl collection tooling: **Discogs integration**, **data ingest**, and **three ML components**—recommender, NLP condition classifier, and price estimator—plus a **web interface** to log in with Discogs and sync your data.

---

## Features

| Component | Description |
|-----------|-------------|
| **Web app** | Log in with your Discogs token, sync collection & wantlist to the app, and call ML APIs (recommendations, condition, price). |
| **Recommender** | Hybrid (ALS + content-based) recommendations using your Discogs collection/wantlist and optional AOTY ratings. |
| **NLP condition classifier** | Predicts sleeve and media condition from seller notes (e.g. Discogs listings). |
| **Price estimator** | Estimates fair market value for releases (stub; pipeline and API in place). |

Shared infrastructure: **Discogs API** client and **AOTY** scraped-data loader used across components.

---

## Quick start

```bash
# Clone and enter project
cd vinyl_management_system

# Create venv and install
python3.12 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -r web/requirements.txt

# Run the web app (from project root)
uvicorn web.app.main:app --reload
```

Open **http://127.0.0.1:8000** → [Log in with Discogs](http://127.0.0.1:8000/auth/login) (paste a [personal token](https://www.discogs.com/settings/developers)) → [Sync / Ingest](http://127.0.0.1:8000/ingest) to pull your collection and wantlist into `data/raw/`.

---

## Project layout

```
vinyl_management_system/
├── configs/base.yaml       # Shared config (paths, Discogs, AOTY, model params)
├── core/                   # Config loader, auth, ingest jobs
├── discogs_api/            # Shared Discogs API client
├── aoty/                   # AOTY scraped data loader
├── recommender/            # ML: hybrid recommendations
├── nlp_condition_classifier/  # ML: condition from seller notes
├── price_estimator/       # ML: price estimation (stub)
├── web/                    # Web UI and API
├── data/raw, data/processed
└── artifacts/
```

Full layout and how components coordinate: **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)**.

---

## Setup

### Requirements

- **Python 3.12+**
- Discogs [personal access token](https://www.discogs.com/settings/developers) (for API and web login)

### Install

```bash
pip install -r requirements.txt
pip install -r web/requirements.txt
```

Optional: install in editable mode so imports resolve from the repo root:

```bash
pip install -e .
```

### Config

Edit **`configs/base.yaml`** to:

- **Discogs**: set `discogs.use_api: true`, `discogs.usernames: ["your_username"]`, and either `discogs.token` or env `DISCOGS_USER_TOKEN`.
- **AOTY**: set `aoty_scraped.dir` to the path of your scraped data (e.g. `data/aoty_scraped`) if you use it; otherwise the recommender uses CSVs in `data/raw/`.

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
python -m recommender.pipeline --config configs/base.yaml --data-dir data/raw --processed-dir data/processed --artifacts-dir artifacts
```

3. Use in code or via web: `GET /api/recommendations` (when logged in).

See **`recommender/README.md`** for details.

### NLP condition classifier

```bash
export PYTHONPATH=/path/to/vinyl_management_system
python -m nlp_condition_classifier.src.pipeline --phase baseline
```

See **`nlp_condition_classifier/README.md`** for data format and options.

### Price estimator

Stub only; entrypoint:

```bash
python -m price_estimator.src.pipeline
```

See **`price_estimator/README.md`** for the planned interface.

---

## Discogs API (shared)

All components use the same **`discogs_api`** package:

```python
from discogs_api import DiscogsClient, get_user_collection, get_user_wantlist

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
from aoty import load_ratings_from_scraped, load_album_metadata_from_scraped
from pathlib import Path
ratings = load_ratings_from_scraped(Path("path/to/scraped"))
albums = load_album_metadata_from_scraped(Path("path/to/scraped"))
```

---

## Subproject docs

| Subproject | README |
|------------|--------|
| Recommender | [recommender/README.md](recommender/README.md) |
| NLP condition classifier | [nlp_condition_classifier/README.md](nlp_condition_classifier/README.md) |
| Price estimator | [price_estimator/README.md](price_estimator/README.md) |
| Web app | [web/README.md](web/README.md) |
| Structure & coordination | [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) |

---

## License

See repository license file.
