# Build from repository root:
#   docker build -f price_estimator/Dockerfile -t vinyl-price-estimator:latest .
#
# Two-stage build (matches grader/Dockerfile):
#   - deps:    stub pyproject.toml for every workspace member; `uv export`
#              writes a lean requirements.txt (no full source needed thanks
#              to --frozen).
#   - runtime: pip install + selective source COPY. Drops `core/`, `grader/`,
#              `recommender/`, `web/`, plus `price_estimator/{scripts,tests,
#              data,artifacts}/` — none of which the API serves from.
#
# The price API loads its trained model from the local filesystem at
# `vinyliq.paths.model_dir`. In the GKE deploy the path is overridden to
# `/data/artifacts/vinyliq` via ConfigMap; trained assets live on a small PVC.
# Feature/marketplace data use Postgres (Cloud SQL) when `postgres_dsn_env` is set.

FROM python:3.12-slim AS deps

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /workspace

COPY pyproject.toml uv.lock ./
COPY repo_meta ./repo_meta
COPY shared/pyproject.toml          ./shared/
COPY core/pyproject.toml            ./core/
COPY grader/pyproject.toml          ./grader/
COPY recommender/pyproject.toml     ./recommender/
COPY web/pyproject.toml             ./web/
COPY price_estimator/pyproject.toml ./price_estimator/

RUN uv export --package vinyl-price-estimator --frozen --no-hashes \
    -o /tmp/requirements.txt

FROM python:3.12-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PIP_NO_CACHE_DIR=1

COPY --from=deps /tmp/requirements.txt /tmp/requirements.txt

# Selective runtime source. `vinyl-price-estimator` only depends on
# `vinyl-shared` in the workspace (see price_estimator/pyproject.toml),
# so `core/` is intentionally omitted. The package code grep confirms
# the API + inference + storage paths only `from shared.*` and
# `from price_estimator.src.*`.
#
# `price_estimator/pyproject.toml` is copied because `requirements.txt`
# contains `-e ./price_estimator` and `-e ./shared`; pip needs each
# package's pyproject.toml at install time to resolve the editable
# installs.
COPY shared                            ./shared
COPY price_estimator/pyproject.toml    ./price_estimator/pyproject.toml
COPY price_estimator/__init__.py       ./price_estimator/__init__.py
COPY price_estimator/src               ./price_estimator/src
COPY price_estimator/configs           ./price_estimator/configs

# Repo-root ``configs/`` exists for collector/recommender; VinylIQ merges use
# ``price_estimator/configs/base.yaml``. GKE should inherit the latter via
# ``VINYLIQ_CONFIG`` so ``vinyliq.inference.*`` survives path overrides.
COPY configs ./configs

RUN pip install --upgrade pip && pip install -r /tmp/requirements.txt

EXPOSE 8801
ENTRYPOINT ["uvicorn", "price_estimator.src.api.main:app", "--host", "0.0.0.0", "--port", "8801"]
CMD []
