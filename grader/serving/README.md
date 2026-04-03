# Vinyl grader FastAPI (MLflow registry)

Loads the registered **pyfunc** model (`vinyl_grader` artifact) with `mlflow.pyfunc.load_model` and serves JSON predictions.

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `MLFLOW_MODEL_URI` | yes | e.g. `models:/VinylGrader/latest`, `models:/VinylGrader/Production`, or a version number |
| `MLFLOW_TRACKING_URI` | recommended | MLflow tracking server URL (same as training) |
| `MLFLOW_REGISTRY_URI` | optional | Only if registry is on a different host than tracking |

For local dev without the registry, you can point at a run artifact:

`MLFLOW_MODEL_URI=runs:/<run_id>/vinyl_grader`

The container must resolve imports for `grader.src.models.grader_pyfunc` (installed via `pip install .`).

If artifacts live in **GCS**, give the container a service account or Workload Identity with **Storage Object Viewer** on the bucket.

## Local run (repo root)

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
export MLFLOW_MODEL_URI=models:/VinylGrader/latest
uv run uvicorn grader.serving.main:app --host 0.0.0.0 --port 8080
```

## Endpoints

- `GET /` — service name and package version
- `GET /health` — `200` only if the model loaded at startup (fail-fast if load fails)
- `POST /predict` — JSON body (exactly one of `text` or `items`)

Single:

```bash
curl -s http://127.0.0.1:8080/predict -H 'Content-Type: application/json' \
  -d '{"text":"VG+ sleeve, light hairlines, plays well"}'
```

Batch:

```bash
curl -s http://127.0.0.1:8080/predict -H 'Content-Type: application/json' \
  -d '{"items":[{"text":"NM in shrink","item_id":"a1"},{"text":"G+ sleeve","item_id":"a2"}]}'
```

## Docker build and push

From the **repository root**:

```bash
docker build -f grader/Dockerfile -t YOUR_REGISTRY/vinyl-grader-api:TAG .
docker push YOUR_REGISTRY/vinyl-grader-api:TAG
```

The Dockerfile is **multi-stage**: the build context still includes the whole uv workspace (so `uv export` can resolve `vinyl-grader` from `uv.lock`), but the **final image** only adds `shared/`, `grader/`, and the exported Python dependencies—not `web/`, `recommender/`, etc.

Run:

```bash
docker run --rm -p 8080:8080 \
  -e MLFLOW_TRACKING_URI=http://YOUR_MLFLOW_HOST:5000 \
  -e MLFLOW_MODEL_URI=models:/VinylGrader/Production \
  -v $GOOGLE_APPLICATION_CREDENTIALS:/secrets/sa.json:ro \
  -e GOOGLE_APPLICATION_CREDENTIALS=/secrets/sa.json \
  YOUR_REGISTRY/vinyl-grader-api:TAG
```

## MLflow server (GCP-oriented)

Use a durable backend store (Postgres recommended for production) and a reliable artifact root:

```text
mlflow server \
  --backend-store-uri postgresql://... \
  --default-artifact-root gs://BUCKET/mlflow-artifacts \
  --host 0.0.0.0 --port 5000 \
  --allowed-hosts 'YOUR_IP:*,localhost:*' \
  --cors-allowed-origins 'http://YOUR_IP:5000'
```

- **`--cors-allowed-origins`**: avoids browser `ajax-api` **403** when the UI origin is sent.
- **`--allowed-hosts`**: include `host:port` patterns as required by your MLflow version.
- **Large weights**: prefer **`gs://`** as default artifact root so laptops and API containers do not rely on long HTTP uploads or VM-local paths.

Training registers the pyfunc with `python -m grader.src.pipeline train` (see `mlflow.register_after_pipeline` in `grader/configs/grader.yaml`) or via `transformer_tune` with `--register`.
