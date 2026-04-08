# Build from repository root:
#   docker build -f price_estimator/Dockerfile -t vinyl-price-estimator:latest .

FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv

COPY pyproject.toml uv.lock ./
COPY repo_meta ./repo_meta
COPY shared ./shared
COPY core ./core
COPY grader ./grader
COPY recommender ./recommender
COPY web ./web
COPY price_estimator ./price_estimator

COPY configs ./configs

RUN uv sync --package vinyl-price-estimator --frozen --no-dev --no-install-project

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=/app

# VinylIQ HTTP API (override for training: docker run ... python -m price_estimator.src.training.train_vinyliq)
ENTRYPOINT ["uvicorn", "price_estimator.src.api.main:app", "--host", "0.0.0.0", "--port", "8801"]
CMD []
