# Vinyl grader FastAPI (MLflow registry)

Loads the registered **pyfunc** model (`vinyl_grader` artifact) with `mlflow.pyfunc.load_model`, runs **Preprocessor + RuleEngine** on each prediction (same behavior as `grader.src.pipeline` inference), and returns JSON.


## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `MLFLOW_MODEL_URI` | yes | e.g. `models:/VinylGrader/latest`, `models:/VinylGrader/Production`, or `runs:/<run_id>/vinyl_grader` |
| `MLFLOW_TRACKING_URI` | recommended | MLflow tracking server URL (same as training) |
| `GOOGLE_APPLICATION_CREDENTIALS` | required when artifacts use **gs://** | Path inside the container to a service account JSON with **Storage Object Viewer** (or broader) on the MLflow artifact bucket |
| `GOOGLE_CLOUD_PROJECT` | optional | GCP project id for the Storage client; if unset, the API sets it from **`project_id`** in the service-account JSON when possible |
| `MLFLOW_REGISTRY_URI` | optional | Only if registry is on a different host than tracking |
| `GRADER_CONFIG_PATH` | optional | Default: `<grader>/configs/grader.yaml` inside the package |
| `GRADER_GUIDELINES_PATH` | optional | Default: `<grader>/configs/grading_guidelines.yaml` |

For **production**, use MLflow with **`--default-artifact-root gs://…`**. The API and training clients talk to the tracking server over HTTP and **read/write run artifacts via GCS** using Application Default Credentials. Omit **`GOOGLE_APPLICATION_CREDENTIALS`** only for fully local stacks (e.g. file/SQLite artifact roots).

---

## Run the API locally (without Docker)

These steps assume you have **cloned this repository** and are using a shell whose **current directory is the repository root** (the folder that contains `pyproject.toml`, `grader/`, `shared/`, etc.).

### Prerequisites

- **Python 3.12+** (`python3.12 --version` or `python --version`)
- Access to your **MLflow** server (or a valid `runs:/…/vinyl_grader` URI and artifacts)

### 1. Clone the repository

```bash
git clone <YOUR_REPO_URL>
cd vinyl_management_system   # or your clone folder name
```

### 2a. Install `uv` and run the API (recommended)

**Install `uv`** (pick one):

- **macOS / Linux** (official installer):

  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

  Then open a **new** terminal (or follow the installer’s note to update `PATH`).

- **Windows** (PowerShell):

  ```powershell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```

- **Any OS** (if you already have Python):

  ```bash
  python3.12 -m pip install uv
  ```

More options: [Installing uv](https://docs.astral.sh/uv/getting-started/installation/).

**Install dependencies and start the server** (from the **repo root**):

```bash
cd /path/to/vinyl_management_system

export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
export MLFLOW_MODEL_URI=models:/VinylGrader/latest

uv sync --package vinyl-grader --extra serve
uv run uvicorn grader.serving.main:app --host 0.0.0.0 --port 8080
```

`uv sync` resolves the workspace and installs **`vinyl-grader`** plus the **`serve`** extra (FastAPI, Uvicorn).

### 2b. Run the API without `uv` (venv + pip)

From the **repo root**, create a virtual environment, install **`vinyl-shared`** and **`vinyl-grader[serve]`** as editable packages (order matters):

**macOS / Linux:**

```bash
cd /path/to/vinyl_management_system

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e ./shared -e "./grader[serve]"

export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
export MLFLOW_MODEL_URI=models:/VinylGrader/latest

python -m uvicorn grader.serving.main:app --host 0.0.0.0 --port 8080
```

**Windows (cmd):**

```bat
cd \path\to\vinyl_management_system
py -3.12 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -U pip
pip install -e .\shared -e ".\\grader[serve]"
set MLFLOW_TRACKING_URI=http://127.0.0.1:5000
set MLFLOW_MODEL_URI=models:/VinylGrader/latest
python -m uvicorn grader.serving.main:app --host 0.0.0.0 --port 8080
```

### Verify

The process loads the model and rule stack **at startup**. **`GET /health`** returns **`200`** with `{"status":"ok","model_loaded":true}` only if loading succeeded; otherwise fix `MLFLOW_*` and ensure the model artifacts are reachable.

---

## Build and run the Docker container

From the **repository root** (build context must include the full workspace for `uv export`):

```bash
docker build -f grader/Dockerfile -t vinyl-grader-api:latest .
```

**Run** (GCS artifact store — mount a key readable inside the container):

```bash
docker run --rm -p 8080:8080 \
  -e MLFLOW_TRACKING_URI=http://YOUR_MLFLOW_HOST:5000 \
  -e MLFLOW_MODEL_URI=models:/VinylGrader/latest \
  -v /ABSOLUTE/PATH/TO/mlflow-artifacts-viewer.json:/secrets/sa.json:ro \
  -e GOOGLE_APPLICATION_CREDENTIALS=/secrets/sa.json \
  vinyl-grader-api:latest
```

On **GCP** (GKE, Cloud Run, Compute Engine), prefer **Workload Identity** or the instance service account instead of a JSON key file; set ADC accordingly so `google-cloud-storage` can reach the bucket.

**Push to a registry** (example):

```bash
docker tag vinyl-grader-api:latest YOUR_REGISTRY/vinyl-grader-api:latest
docker push YOUR_REGISTRY/vinyl-grader-api:latest
```

The final image includes `shared/`, `grader/` (including `grader/configs` for the rule engine), and dependencies from `uv export --package vinyl-grader --extra serve`. The Dockerfile uses a two-stage build with a stub-pyproject deps stage so non-target workspace members are not copied into the build context — see [`grader/Dockerfile`](../Dockerfile).

---

## GKE / containerized deployment

The demo deploy on GKE Autopilot is documented in
[`k8s/demo/README.md`](../../k8s/demo/README.md). Highlights specific
to the grader:

- The image is published by
  [`.github/workflows/demo-deploy.yml`](../../.github/workflows/demo-deploy.yml)
  (Workload Identity Federation auth, no JSON SA keys in the repo) to
  Artifact Registry under
  `<region>-docker.pkg.dev/<project>/vinyl-images/grader:demo`.
- Routing: the GKE Gateway forwards `https://<host>/grader/*` to this
  Service after a `URLRewrite` filter strips the `/grader` prefix —
  the routes here (`/`, `/health`, `/predict`) match unchanged. See
  [`k8s/demo/httproute.yaml`](../../k8s/demo/httproute.yaml).
- `MLFLOW_TRACKING_URI` and `MLFLOW_MODEL_URI` are injected via the
  `vinyl-mlflow` Secret (created imperatively from `.env` at
  bootstrap; see Phase 4 in the runbook).
- Pod-level GCS auth uses **Workload Identity** (KSA `vinyl-runtime`
  bound to `RUNTIME_GSA`); no `GOOGLE_APPLICATION_CREDENTIALS` mount.
- The startup probe gives 5 minutes (`failureThreshold=30,
  periodSeconds=10`) to let `mlflow.pyfunc.load_model` complete on
  cold start.

---

## Predict API: request and response format

### `POST /predict`

**Content-Type:** `application/json`

Provide **exactly one** of:

| Mode | Body shape | Constraints |
|------|------------|-------------|
| Single string | `{ "text": "<seller notes>" }` | `text` non-empty, max 16 000 characters |
| Batch | `{ "items": [ ... ] }` | 1–256 items; each item has `text` (required) and optional `item_id` (string or int) |

**Validation error** (e.g. both `text` and `items`, or neither): HTTP **422** with a JSON detail body from FastAPI/Pydantic.

### Response shape (HTTP 200)

```json
{
  "predictions": [
    {
      "item_id": "<echoed id or index>",
      "predicted_sleeve_condition": "<grade string>",
      "predicted_media_condition": "<grade string>",
      "sleeve_confidence": 0.0,
      "media_confidence": 0.0,
      "contradiction_detected": false,
      "rule_override_applied": false,
      "rule_override_target": null
    }
  ]
}
```

- **`predicted_*_condition`**: After **RuleEngine** (may differ from raw model labels when rules override).
- **`sleeve_confidence` / `media_confidence`**: Model **top-1** confidence from the pyfunc (not renormalized after rule overrides).
- **`rule_override_target`**: If an override ran, may be `"sleeve"`, `"media"`, or `"both"`; otherwise `null`.
- **`contradiction_detected`**: If true, rule overrides were suppressed for that row.

Grade strings are from your label encoders (e.g. `Near Mint`, `Very Good Plus`, `Poor`, …).

### Example: single prediction

**Request:**

```bash
curl -s http://127.0.0.1:8080/predict \
  -H 'Content-Type: application/json' \
  -d '{"text":"VG+ sleeve, light hairlines, plays well"}'
```

**Example response** (illustrative; grades and confidences depend on model version and weights):

```json
{
  "predictions": [
    {
      "item_id": 0,
      "predicted_sleeve_condition": "Very Good Plus",
      "predicted_media_condition": "Very Good Plus",
      "sleeve_confidence": 0.412,
      "media_confidence": 0.385,
      "contradiction_detected": false,
      "rule_override_applied": false,
      "rule_override_target": null
    }
  ]
}
```

### Example: batch prediction

**Request:**

```bash
curl -s http://127.0.0.1:8080/predict \
  -H 'Content-Type: application/json' \
  -d '{"items":[{"text":"NM in shrink","item_id":"a1"},{"text":"G+ sleeve","item_id":"a2"}]}'
```

**Example response** (structure only; values vary):

```json
{
  "predictions": [
    {
      "item_id": "a1",
      "predicted_sleeve_condition": "Near Mint",
      "predicted_media_condition": "Mint",
      "sleeve_confidence": 0.55,
      "media_confidence": 0.62,
      "contradiction_detected": false,
      "rule_override_applied": false,
      "rule_override_target": null
    },
    {
      "item_id": "a2",
      "predicted_sleeve_condition": "Good Plus",
      "predicted_media_condition": "Very Good",
      "sleeve_confidence": 0.48,
      "media_confidence": 0.51,
      "contradiction_detected": false,
      "rule_override_applied": true,
      "rule_override_target": "sleeve"
    }
  ]
}
```

---

## Other endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Service name and package version |
| `GET` | `/health` | `200` with `{"status":"ok","model_loaded":true}` if the model loaded at startup; `503` if not |

---

## MLflow server

### Example: durable metadata + GCS artifacts

```text
mlflow server \
  --backend-store-uri postgresql://... \
  --default-artifact-root gs://BUCKET/mlflow-artifacts \
  --host 0.0.0.0 --port 5000 \
  --allowed-hosts 'YOUR_IP:*,localhost:*' \
  --cors-allowed-origins 'http://YOUR_IP:5000'
```

Clients that **log** or **load** models need **GCP credentials** (or ADC) for that bucket, as in **MLflow on GCP (GCS artifacts)** above.

### General notes

- **`--cors-allowed-origins`**: avoids browser `ajax-api` **403** when the UI origin is sent.
- **`--allowed-hosts`**: include `host:port` patterns as required by your MLflow version.

Training registers the pyfunc with `python -m grader.src.pipeline train` (see `mlflow.register_after_pipeline` in `grader/configs/grader.yaml`) or via `transformer_tune` with registration enabled.
