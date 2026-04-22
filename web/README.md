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

## API routes

Routers are mounted in `web/app/main.py` (`/auth`, `/ingest`, `/api`).

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Home / dashboard (HTML) |
| `POST` | `/auth/token` | Submit Discogs token; stores session |
| `GET` | `/auth/me` | Current username (from session) |
| `GET` | `/auth/login` | Login page (HTML) |
| `POST` | `/ingest/sync` | Sync collection + wantlist → `data/raw/` |
| `POST` | `/ingest/full` | Full ingest (Discogs + AOTY if configured) |
| `GET` | `/ingest/` | Ingest page (HTML) |
| `GET` | `/api/recommendations` | Recommendations for logged-in user |
| `POST` | `/api/condition` | Condition prediction from seller notes |
| `GET` | `/api/price/{release_id}` | Price estimate (proxies to `PRICE_SERVICE_URL` if set, else in-process) |

OpenAPI: `/docs` when the app is running.

## Environment

- `DISCOGS_USER_TOKEN`: Optional default token (e.g. for server-side single-user).
- For per-user login, users paste their token in the UI; the app stores it in session (in-memory by default; use Redis/DB in production).
- `PRICE_SERVICE_URL`: Optional base URL for the VinylIQ price microservice (e.g. `http://127.0.0.1:8801`). When set, `GET /api/price/{release_id}` proxies with `POST {PRICE_SERVICE_URL}/estimate`; when unset, the web app calls in-process `estimate()` from `vinyl-price-estimator`.
- `VINYLIQ_API_KEY`: Optional; sent as `X-API-Key` when proxying to the price service if the API requires it.
