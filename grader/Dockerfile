# Build from repository root:
#   docker build -f grader/Dockerfile -t vinyl-grader-api:latest .
#
# Stage 1 needs the full uv workspace (lock + all member pyprojects) to export
# grader-only pins. Stage 2 copies only shared/ + grader/ into the runtime image.

FROM python:3.12-slim AS deps

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /workspace
COPY pyproject.toml uv.lock ./
COPY repo_meta ./repo_meta
COPY shared ./shared
COPY core ./core
COPY grader ./grader
COPY recommender ./recommender
COPY web ./web
COPY price_estimator ./price_estimator

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
COPY shared ./shared
COPY grader ./grader

RUN pip install --upgrade pip && pip install -r /tmp/requirements.txt

RUN python -c "from transformers import DistilBertModel; DistilBertModel.from_pretrained('distilbert-base-uncased')"

EXPOSE 8080
ENV MLFLOW_TRACKING_URI="" MLFLOW_MODEL_URI=""
CMD ["uvicorn", "grader.serving.main:app", "--host", "0.0.0.0", "--port", "8080"]
