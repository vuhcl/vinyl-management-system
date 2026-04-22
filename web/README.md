# Web interface

FastAPI app for the Vinyl Management System: **Discogs login**, **data ingest**, and access to the three ML components (recommender, vinyl condition grader, price estimator).

## Features

- **Auth**: Submit Discogs personal token → app verifies with Discogs and stores session (username).
- **Ingest**: Trigger sync of collection and wantlist from Discogs; data is written to `data/raw/` (repo root) so the web image can mount that path without the recommender package tree.
- **API**: Recommendations, condition prediction, price estimate (delegate to ML subprojects).

## Run

From **project root** (so `core`, `shared`, `recommender`, etc. are on PYTHONPATH):

```bash
uv sync --extra test
uv run uvicorn web.app.main:app --reload
```

Or with a venv activated after `uv sync`:

```bash
uvicorn web.app.main:app --reload
```

Then open http://127.0.0.1:8000 . Use the **Login** page to paste your Discogs token, then **Ingest** to sync your collection/wantlist.

## Environment

- `DISCOGS_USER_TOKEN`: Optional default token (e.g. for server-side single-user).
- For per-user login, users paste their token in the UI; the app stores it in session (in-memory by default; use Redis/DB in production).
