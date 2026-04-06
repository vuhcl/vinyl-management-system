# Build from repository root:
#   docker build -f recommender/Dockerfile -t vinyl-recommender:latest .

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

RUN uv sync --package vinyl-recommender --frozen --no-dev --no-install-project

ENV PATH="/app/.venv/bin:$PATH"

# No HTTP server in this package; use for batch jobs or docker run ... python -m recommender.pipeline
ENTRYPOINT ["python", "-m", "recommender.pipeline"]
CMD ["--help"]
