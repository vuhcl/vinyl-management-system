# Agent index — vinyl_management_system

Quick orientation for coding agents. **Backlog and priorities live in [`.notes/task_list.md`](.notes/task_list.md), not here.**

## Context to read first

| Doc | Purpose |
|-----|---------|
| [`.notes/project_overview.md`](.notes/project_overview.md) | Goals, architecture, user journeys |
| [`.notes/task_list.md`](.notes/task_list.md) | Living backlog (canonical) |
| [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) | Packages, config roots, layout |
| [`.cursorrules`](.cursorrules) | Portable workflow + repo-specific rules |

## Workspace packages (uv, Python 3.12+)

| Package | Path | Role |
|---------|------|------|
| vinyl-shared | `shared/` | Discogs client, AOTY loader |
| vinyl-core | `core/` | Config load, auth, ingest jobs |
| vinyl-recommender | `recommender/` | Hybrid recommender |
| vinyl-grader | `grader/` | Condition grader + optional serving |
| vinyl-price-estimator | `price_estimator/` | VinylIQ training + price API |
| vinyl-web | `web/` | FastAPI dashboard + ML proxy routes |

Run commands from **repo root** with `uv run` unless a README says `PYTHONPATH=.`.

## Config roots

- **Shared:** `configs/base.yaml` — `core.config.load_config()`
- **Recommender:** `recommender/configs/base.yaml` (inherits shared)
- **VinylIQ:** `price_estimator/configs/base.yaml` — override via `VINYLIQ_CONFIG` where supported

## Default validation (PR fast — Tier A)

Root `pyproject.toml` enables coverage in `addopts`; CI and agents use **`--no-cov`**.

```bash
uv sync --extra test
# Grader Tier A includes NLP monitoring tests; if GE import fails locally:
# uv sync --extra test --extra monitoring

uv run pytest grader/tests --no-cov -m "not integration" -v
uv run pytest price_estimator/tests --no-cov -m "not monitoring" -v
uv run pytest recommender/tests --no-cov -m "not monitoring" -v
```

**Cross-cutting edits** (`shared/`, `core/`, `configs/`, `pyproject.toml`, `uv.lock`): run all three Tier A commands.

Full CI tiers (nightly, integration, monitoring, Docker): invoke skill **`ci-and-test-matrix`** (`.cursor/skills/ci-and-test-matrix/SKILL.md`).

## Package skills (invoke by name)

| Skill | When |
|-------|------|
| `ci-and-test-matrix` | Which pytest tier to run; CI failure triage |
| `vinyliq-price-estimator` | `price_estimator/` training, features, scraping, API |
| `recommender-pipeline` | `recommender/` ingest, ALS, artifacts |
| `grader-and-serving` | `grader/` models, serving, demo golden JSON |
| `discogs-ingest-and-api` | `shared/`, `core/`, Discogs API + ingest |
| `web-and-vinyliq-extension` | `web/`, `vinyliq-extension/`, demo Playwright |
| `demo-k8s-and-docker` | `k8s/demo/`, deploy branches, Docker images |
| `notes-and-agent-context` | Update `.notes/` after substantive work |

Path rules under `.cursor/rules/` point at these skills when matching files are open.

## Do not load into context

`uv.lock`, bulk `data/raw/`, `recommender/data/`, `feature_store.sqlite`, artifacts, `.specstory/` — see [`.cursorignore`](.cursorignore).
