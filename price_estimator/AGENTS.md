# VinylIQ — agent brief

Short invariants for coding agents. Full pipeline detail: [README.md](README.md). Extension: [../vinyliq-extension/README.md](../vinyliq-extension/README.md).

## Session opener (paste at chat start)

```
VinylIQ monorepo package: price_estimator/ (config: price_estimator/configs/base.yaml).
Price API: POST /estimate (not /predict). Grader serving: POST /predict with { "text" }.
Feature columns: VinylIQFeatureSchema / default_feature_columns() — never duplicate column lists.
Discogs REST: shared/discogs_api/client.py (DiscogsClient) — do not invent endpoints.
Website scraping: Botasaurus under price_estimator/scripts/ only; see .notes/lessons_cursor.md.
Chrome MV3 host_permissions: www.discogs.com, localhost, *.nip.io (see vinyliq-extension/manifest.json).
Validate: uv run pytest price_estimator/tests --no-cov -m "not monitoring" -v
```

## API surfaces

| Surface | Route | Notes |
|---------|-------|-------|
| Price microservice | `POST /estimate` | [`src/api/main.py`](src/api/main.py), schemas in [`src/api/schemas.py`](src/api/schemas.py) |
| Grader serving | `POST /predict` | [`../grader/serving/main.py`](../grader/serving/main.py) |
| Web proxy | `GET /api/price/{release_id}` | Proxies to `/estimate`; [`../web/app/routers/ml.py`](../web/app/routers/ml.py) |

Optional auth: `X-API-Key` when `VINYLIQ_API_KEY` is set.

### Extension contract (must not break)

**`POST /estimate`** — request: `release_id`, `media_condition`, `sleeve_condition`, optional `marketplace_client`.  
Response fields used by extension: `estimated_price`, `confidence_interval[0]`, `confidence_interval[1]`, `model_version`, `status`.

**`POST /predict`** — request: `{ "text" }`.  
Response: `predictions[0].predicted_media_condition`, `predictions[0].predicted_sleeve_condition`, sleeve/media confidences.

Contract tests: [`tests/test_estimate_api_contract.py`](tests/test_estimate_api_contract.py), [`../grader/tests/test_serving_predict_contract.py`](../grader/tests/test_serving_predict_contract.py).

## Feature columns

Single source of truth: [`VinylIQFeatureSchema`](src/features/vinyliq_features.py) (`CONDITION_HEAD`, `COLD_START`, `MARKETPLACE_DEPTH_BODY`, `CATALOG_TAIL`).  
Training persists order in `feature_columns.joblib`. Import `default_feature_columns()` or `residual_training_feature_columns()` — do not maintain a parallel `FEATURE_COLS` list.

Fixture lock: [`tests/test_feature_pipeline_fixtures.py`](tests/test_feature_pipeline_fixtures.py).

## Discogs REST allowlist

Use methods on [`DiscogsClient`](../shared/discogs_api/client.py) — do not add undocumented API paths:

| Method | Endpoint |
|--------|----------|
| `get_release` / `get_release_with_retries` | `GET /releases/{id}` |
| `get_master` | `GET /masters/{id}` |
| `get_marketplace_stats` / `get_marketplace_stats_with_retries` | `GET /marketplace/stats/{id}` |
| `get_release_stats_with_retries` | `GET /releases/{id}/stats` |
| `get_price_suggestions_with_retries` | `GET /marketplace/price_suggestions/{id}` |
| `get_user_collection_releases` | user collection |
| `get_user_wantlist` | user wantlist |
| `database_search` | database search |

**Not in REST client:** transaction history, cross-site matching. **Website** sale-history and search HTML scraping live under `price_estimator/scripts/` (Botasaurus). Respect rate limits and Discogs terms (backlog H1).

## Chrome extension (MV3)

[`../vinyliq-extension/manifest.json`](../vinyliq-extension/manifest.json) `host_permissions`:

- `https://www.discogs.com/*`, `https://discogs.com/*`, `https://*.discogs.com/*`
- `http://127.0.0.1/*`, `http://localhost/*`
- `https://*.nip.io/*` (demo Gateway)

DOM helpers: [`../vinyliq-extension/listing_dom.js`](../vinyliq-extension/listing_dom.js) — keep in sync with Playwright locators; run `npm test` in `vinyliq-extension/`.

## Validate (Tier A)

```bash
uv sync --extra test
uv run pytest price_estimator/tests --no-cov -m "not monitoring" -v
```

Cross-package edits to `shared/` or `configs/`: run all three Tier A suites (see root [AGENTS.md](../AGENTS.md)).

Skill: **`vinyliq-price-estimator`** (`.cursor/skills/vinyliq-price-estimator/SKILL.md`).
