# Build from repository root:
#   docker build -f grader/Dockerfile -t vinyl-grader-api:latest .
#
# Two-stage build:
#   - deps:    stub pyproject.toml for every workspace member so uv can resolve
#              the workspace, then `uv export` writes a lean requirements.txt.
#   - runtime: pip install + selective source COPY (only what serves traffic).

FROM python:3.12-slim AS deps

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /workspace

# uv workspace metadata: every member's pyproject.toml only.
# Stubs are enough for `uv export --frozen` because the lockfile
# already pins all resolved versions; no source code needed here.
COPY pyproject.toml uv.lock ./
COPY repo_meta ./repo_meta
COPY shared/pyproject.toml          ./shared/
COPY core/pyproject.toml            ./core/
COPY grader/pyproject.toml          ./grader/
COPY recommender/pyproject.toml     ./recommender/
COPY web/pyproject.toml             ./web/
COPY price_estimator/pyproject.toml ./price_estimator/

RUN uv export --package vinyl-grader --extra serve --frozen --no-hashes \
    -o /tmp/requirements.txt

FROM python:3.12-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface \
    TRANSFORMERS_CACHE=/root/.cache/huggingface

COPY --from=deps /tmp/requirements.txt /tmp/requirements.txt

# Selective runtime source — only what `grader.serving.main` and its transitive
# imports need. `grader.src.export.coreml_export` lives under `grader/src/`
# but is never imported by the serving path, so its `from core import ...`
# does not require shipping `core/` here.
#
# `grader/pyproject.toml` is copied because `requirements.txt` contains
# `-e ./grader` and `-e ./shared`; pip needs each package's pyproject.toml
# (or setup.py) at install time to resolve the editable installs.
COPY shared                  ./shared
COPY grader/pyproject.toml   ./grader/pyproject.toml
COPY grader/__init__.py      ./grader/__init__.py
COPY grader/src              ./grader/src
COPY grader/serving          ./grader/serving
COPY grader/configs          ./grader/configs

RUN pip install --upgrade pip && pip install -r /tmp/requirements.txt

# Pre-cache the DistilBERT base weights so cold start does not have to
# download them inside the cluster on the first request.
RUN python -c "from transformers import DistilBertModel; DistilBertModel.from_pretrained('distilbert-base-uncased')"

EXPOSE 8080
ENV MLFLOW_TRACKING_URI="" MLFLOW_MODEL_URI=""
CMD ["uvicorn", "grader.serving.main:app", "--host", "0.0.0.0", "--port", "8080"]
