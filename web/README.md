# Web interface

FastAPI app for the Vinyl Management System: **Discogs login**, **data ingest**, and access to the three ML components (recommender, NLP condition classifier, price estimator).

## Features

- **Auth**: Submit Discogs personal token → app verifies with Discogs and stores session (username).
- **Ingest**: Trigger sync of collection and wantlist from Discogs; data is written to `data/raw/` for the recommender and other pipelines.
- **API**: Recommendations, condition prediction, price estimate (delegate to ML subprojects).

## Run

From **project root** (so `core`, `discogs_api`, `recommender`, etc. are on PYTHONPATH):

```bash
pip install -r requirements.txt
pip install -r web/requirements.txt
uvicorn web.app.main:app --reload --app-dir .
```

Or from `web/`:

```bash
cd web
PYTHONPATH=.. uvicorn app.main:app --reload
```

Then open http://127.0.0.1:8000 . Use the **Login** page to paste your Discogs token, then **Ingest** to sync your collection/wantlist.

## Environment

- `DISCOGS_USER_TOKEN`: Optional default token (e.g. for server-side single-user).
- For per-user login, users paste their token in the UI; the app stores it in session (in-memory by default; use Redis/DB in production).
