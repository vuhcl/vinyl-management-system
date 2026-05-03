# GKE Autopilot demo runbook (`vinyl-demo` namespace)

End-to-end runbook for the **demo wave 1** deploy: GCP bootstrap, CI auth
(WIF), Workload Identity, Cloud SQL + slim PVC for models, deploy + smoke. Sibling
to the application code (`deploy/demo-wave1-app`) and verification
(`deploy/demo-wave1-verify`) branches.

The plan (`demo_infra_wave1_c99464c4.plan.md`) is the design source of
truth; this README is the operator-facing companion.

## What the demo deploys

```mermaid
flowchart LR
    Browser[Chrome extension 0.2.0] -->|HTTPS via nip.io| Gateway
    subgraph GKE [GKE Autopilot us-central1]
        Gateway[gke-l7-global-external-managed Gateway] --> Grader[grader-api Pod]
        Gateway --> Price[price-api Pod]
        subgraph price_pod [price-api Pod]
            App[price container]
            Proxy[cloudsql-proxy sidecar]
        end
        App -->|127.0.0.1:5432| Proxy
        Proxy --> CloudSQL[(Cloud SQL Postgres)]
        App --> PVC[(PVC model artifacts)]
        App --> Redis[(Memorystore Redis Basic)]
    end
    Workstation -->|loader psycopg| CloudSQL
    Grader -->|mlflow.pyfunc.load_model| MLflow[MLflow tracking URI public]
    Price -->|cache miss| Discogs[Discogs API]
```

Two FastAPI services behind one Gateway. URLRewrite filters strip the
`/grader` and `/price` prefixes so the inner FastAPI routes
(`/health`, `/predict`, `/estimate`) match unchanged.

## Required environment (`.env` at repo root)

`set -a && source .env && set +a` before any of the imperative steps
below.

**Provided by you before Phase 0:**

```dotenv
GCP_PROJECT_ID=...
GCP_REGION=us-central1

MLFLOW_TRACKING_URI=https://mlflow.your-host
MLFLOW_MODEL_URI=models:/VinylGrader/latest

DISCOGS_USER_TOKEN=...
```

After **Phase 0e** (Cloud SQL bootstrap), append:

```dotenv
DATABASE_URL=postgresql://vinyl_app:<hex-password>@127.0.0.1:5432/vinyliq
```

**Optional overrides (defaults are fine):**

```dotenv
AR_REPO=vinyl-images
GKE_CLUSTER=vinyl-demo
REDIS_INSTANCE=vinyl-demo-redis
IMAGE_TAG=demo
```

**Auto-captured during Phase 0** (commands echo these — append back into
`.env`):

```dotenv
REDIS_HOST=10.x.y.z
STATIC_IP=35.x.y.z
DEMO_HOSTNAME=35-x-y-z.nip.io
WIF_PROVIDER=projects/<num>/locations/global/workloadIdentityPools/.../providers/...
WIF_SERVICE_ACCOUNT=vinyl-demo-ci@$GCP_PROJECT_ID.iam.gserviceaccount.com
RUNTIME_GSA=vinyl-demo-runtime@$GCP_PROJECT_ID.iam.gserviceaccount.com
```

## Phase 0 — GCP bootstrap (one-time)

```bash
set -a && source .env && set +a
gcloud config set project "$GCP_PROJECT_ID"

gcloud services enable \
  artifactregistry.googleapis.com \
  container.googleapis.com \
  redis.googleapis.com \
  storage.googleapis.com \
  iam.googleapis.com \
  compute.googleapis.com \
  certificatemanager.googleapis.com \
  networkservices.googleapis.com \
  networksecurity.googleapis.com \
  sqladmin.googleapis.com

gcloud artifacts repositories create "${AR_REPO:-vinyl-images}" \
  --repository-format=docker --location="$GCP_REGION"

gcloud container clusters create-auto "${GKE_CLUSTER:-vinyl-demo}" \
  --region="$GCP_REGION"

# Newer gcloud versions no longer accept --gateway-api / --workload-pool on
# create-auto; enable both via cluster update. Run two updates: some CLI
# versions reject combining --gateway-api and --workload-pool in one
# command (`Exactly one of (...) must be specified`).
gcloud container clusters update "${GKE_CLUSTER:-vinyl-demo}" \
  --region="$GCP_REGION" \
  --workload-pool="$GCP_PROJECT_ID.svc.id.goog"

gcloud container clusters update "${GKE_CLUSTER:-vinyl-demo}" \
  --region="$GCP_REGION" \
  --gateway-api=standard

gcloud redis instances create "${REDIS_INSTANCE:-vinyl-demo-redis}" \
  --size=1 --region="$GCP_REGION" --tier=basic --network=default

gcloud compute addresses create vinyl-demo-ip --global

# Capture values back into .env
{
  echo "REDIS_HOST=$(gcloud redis instances describe ${REDIS_INSTANCE:-vinyl-demo-redis} --region=$GCP_REGION --format='value(host)')"
  echo "STATIC_IP=$(gcloud compute addresses describe vinyl-demo-ip --global --format='value(address)')"
  echo "DEMO_HOSTNAME=$(gcloud compute addresses describe vinyl-demo-ip --global --format='value(address)' | tr . -).nip.io"
} >> .env
set -a && source .env && set +a
```

## Phase 0b — TLS via CertificateMap (Gateway API path)

GKE Gateway API does not consume the legacy `ManagedCertificate` k8s
resource — Google-managed certificates are wired through Certificate
Manager (CertificateMap), and the Gateway references the map via
annotation. Provision once:

```bash
# 1. Create the Certificate Manager certificate (DNS-01 / HTTP-01 auto)
gcloud certificate-manager certificates create vinyl-demo-cert \
  --domains="$DEMO_HOSTNAME"

# 2. Create the certificate map and bind the cert as the default entry
gcloud certificate-manager maps create vinyl-demo-certmap

gcloud certificate-manager maps entries create vinyl-demo-default \
  --map=vinyl-demo-certmap \
  --certificates=vinyl-demo-cert \
  --hostname="$DEMO_HOSTNAME"
```

The Gateway manifest already references `networking.gke.io/certmap:
vinyl-demo-certmap`. Provisioning is async (5-15 min) and only succeeds
after the static IP routes traffic to the Gateway, so plan to run this
before Phase 5 deploy and to expect a brief "PROVISIONING" window.

## Phase 0c — CI auth (GitHub Actions WIF)

```bash
PROJECT_NUMBER=$(gcloud projects describe "$GCP_PROJECT_ID" --format='value(projectNumber)')

# Pool + provider for the GitHub repo
gcloud iam workload-identity-pools create vinyl-demo-pool \
  --location=global --display-name="Vinyl demo CI"

gcloud iam workload-identity-pools providers create-oidc github \
  --location=global \
  --workload-identity-pool=vinyl-demo-pool \
  --display-name="GitHub Actions" \
  --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository,attribute.ref=assertion.ref" \
  --attribute-condition='assertion.repository=="vuhcl/vinyl_management_system"' \
  --issuer-uri="https://token.actions.githubusercontent.com"

# CI service account with AR write
gcloud iam service-accounts create vinyl-demo-ci

gcloud projects add-iam-policy-binding "$GCP_PROJECT_ID" \
  --member="serviceAccount:vinyl-demo-ci@${GCP_PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/artifactregistry.writer"

# Allow the GitHub repo to impersonate the CI SA
gcloud iam service-accounts add-iam-policy-binding \
  "vinyl-demo-ci@${GCP_PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/vinyl-demo-pool/attribute.repository/vuhcl/vinyl_management_system"

# Capture provider name for GitHub repo secrets
echo "WIF_PROVIDER=projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/vinyl-demo-pool/providers/github" >> .env
echo "WIF_SERVICE_ACCOUNT=vinyl-demo-ci@${GCP_PROJECT_ID}.iam.gserviceaccount.com" >> .env
```

Then go to **GitHub repo → Settings → Secrets and variables → Actions**
and add:

- `GCP_PROJECT_ID`
- `GCP_REGION`
- `AR_REPO` (optional; defaults to `vinyl-images`)
- `WIF_PROVIDER` (full provider resource name from `.env`)
- `WIF_SERVICE_ACCOUNT` (`vinyl-demo-ci@...`)

`.github/workflows/demo-deploy.yml` will now build and push images on
every push to the `deploy/demo-wave1*` branches.

## Phase 0d — Runtime Workload Identity (KSA -> GSA)

```bash
gcloud iam service-accounts create vinyl-demo-runtime

# GCS reads for any model/data the grader/price APIs need from a bucket
gcloud projects add-iam-policy-binding "$GCP_PROJECT_ID" \
  --member="serviceAccount:vinyl-demo-runtime@${GCP_PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/storage.objectViewer"

# Cloud SQL Auth Proxy + IAM DB auth path uses cloudsql.client on the runtime GSA
gcloud projects add-iam-policy-binding "$GCP_PROJECT_ID" \
  --member="serviceAccount:vinyl-demo-runtime@${GCP_PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/cloudsql.client"

# Bind KSA `vinyl-demo/vinyl-runtime` to this GSA
gcloud iam service-accounts add-iam-policy-binding \
  "vinyl-demo-runtime@${GCP_PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/iam.workloadIdentityUser" \
  --member="serviceAccount:${GCP_PROJECT_ID}.svc.id.goog[vinyl-demo/vinyl-runtime]"

echo "RUNTIME_GSA=vinyl-demo-runtime@${GCP_PROJECT_ID}.iam.gserviceaccount.com" >> .env
set -a && source .env && set +a
```

## Phase 0e — Cloud SQL for Postgres (feature store + marketplace_stats)

Workstation one-time tools (macOS):

```bash
brew install libpq cloud-sql-proxy
brew link --force libpq   # puts psql on PATH
```

Provision instance + DB user (hex password stays URL-safe in `DATABASE_URL`):

```bash
set -a && source .env && set +a

# Shared-core tier db-g1-small requires Enterprise edition (not Enterprise Plus).
# Without --edition=ENTERPRISE, gcloud may default to ENTERPRISE_PLUS, which
# needs tiers like db-perf-optimized-N-* instead.
gcloud sql instances create vinyl-demo-db \
  --database-version=POSTGRES_16 \
  --edition=ENTERPRISE \
  --tier=db-g1-small \
  --region="$GCP_REGION" \
  --storage-size=10GB \
  --storage-type=HDD \
  --storage-auto-increase \
  --backup-start-time=03:00

gcloud sql databases create vinyliq --instance=vinyl-demo-db

DB_PASSWORD=$(openssl rand -hex 32)
gcloud sql users create vinyl_app \
  --instance=vinyl-demo-db \
  --password="$DB_PASSWORD"

echo "DATABASE_URL=postgresql://vinyl_app:${DB_PASSWORD}@127.0.0.1:5432/vinyliq" >> .env
set -a && source .env && set +a
```

Pods and your workstation both use `127.0.0.1:5432`: in-cluster via the
`cloudsql-proxy` sidecar; locally via `cloud-sql-proxy
"${GCP_PROJECT_ID}:${GCP_REGION}:vinyl-demo-db" --port=5432`.

Cost tip: pause billing-ish idle state with
`gcloud sql instances patch vinyl-demo-db --activation-policy=NEVER` between demos.

## Phase 4 — Apply manifests, create secrets, populate data + PVC

### 1. Namespace + ServiceAccount (templated)

```bash
kubectl apply -f k8s/demo/namespace.yaml
envsubst < k8s/demo/serviceaccount.yaml | kubectl apply -f -
```

### 2. Secrets (imperative — values from `.env`)

```bash
kubectl -n vinyl-demo create secret generic vinyl-mlflow \
  --from-literal=MLFLOW_TRACKING_URI="$MLFLOW_TRACKING_URI" \
  --from-literal=MLFLOW_MODEL_URI="$MLFLOW_MODEL_URI"

# Price API MLflow pull (see `VINYLIQ_MLFLOW_MODEL_URI` in price_estimator README).
# Recreate from `.env` any time values change; `apply` updates the Secret in place.
kubectl -n vinyl-demo create secret generic vinyl-price-mlflow \
  --from-literal=MLFLOW_TRACKING_URI="$MLFLOW_TRACKING_URI" \
  --from-literal=VINYLIQ_MLFLOW_MODEL_URI="$VINYLIQ_MLFLOW_MODEL_URI" \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl -n vinyl-demo create secret generic vinyl-redis \
  --from-literal=REDIS_HOST="$REDIS_HOST"

kubectl -n vinyl-demo create secret generic vinyl-discogs \
  --from-literal=DISCOGS_USER_TOKEN="$DISCOGS_USER_TOKEN"

kubectl -n vinyl-demo create secret generic vinyl-cloudsql \
  --from-literal=DATABASE_URL="$DATABASE_URL"
```

`secrets.example.yaml` documents the expected key inventory but is
intentionally not applied — values stay out of version control.

### 3. ConfigMaps + PVC

```bash
kubectl apply -f k8s/demo/configmap.yaml
kubectl apply -f k8s/demo/price-config.yaml
kubectl apply -f k8s/demo/price-pvc.yaml
```

`vinyl-cloudsql` must exist **before** `price-deployment.yaml` is applied (Phase 5).

### 4. Schema + Postgres load + model PVC (before Phase 5 deploy)

Apply ConfigMaps + PVC first (§3). Then load **Cloud SQL** from your workstation
SQLite sources and copy **only** `artifacts/vinyliq` onto the PVC. Run this **before**
rolling `price-api` so `/health` sees rows + `model_loaded=true`.

```bash
set -a && source .env && set +a

# Local proxy (terminal 1 or background)
cloud-sql-proxy "${GCP_PROJECT_ID}:${GCP_REGION}:vinyl-demo-db" --port=5432 &
PROXY_PID=$!
until pg_isready -h 127.0.0.1 -p 5432 -U vinyl_app -d vinyliq; do sleep 1; done

# Sanity — demo release_id 456663 (Beatles White Album mono first press).
sqlite3 price_estimator/data/cache/marketplace_stats.sqlite \
  "SELECT release_id FROM marketplace_stats WHERE release_id='456663';"
ls price_estimator/artifacts/vinyliq/xgb_model.joblib

# Hydrate the demo golden release (demo_release_id in grader/demo/golden_predict_demo.json)
# via Discogs GET /releases + /marketplace/price_suggestions (requires DISCOGS_TOKEN).
PYTHONPATH=. uv run python price_estimator/scripts/populate_demo_golden_discogs.py

# Schema (idempotent)
psql "$DATABASE_URL" -f k8s/demo/schema.sql

# Bulk load (~10–30 min on db-g1-small for large SQLite files). Features load
# uses chunked COPY with commits (default 50k rows) so long runs survive proxy
# blips; tune with --copy-chunk-rows (smaller = more checkpoints, slower).
uv run python price_estimator/scripts/sqlite_to_cloudsql_loader.py \
  --feature-store price_estimator/data/feature_store.sqlite \
  --marketplace-db price_estimator/data/cache/marketplace_stats.sqlite \
  --database-url "$DATABASE_URL" \
  --batch-size 1000 \
  --copy-chunk-rows 50000

# If features died mid-load with duplicate-key errors on retry:
#   psql "$DATABASE_URL" -c "TRUNCATE releases_features;"
# then re-run the loader (use --skip-marketplace if marketplace_stats finished).

psql "$DATABASE_URL" -c "SELECT COUNT(*) FROM releases_features;"
psql "$DATABASE_URL" -c "SELECT COUNT(*) FROM marketplace_stats;"

kill "$PROXY_PID"

# Slim PVC: copy trained model assets only (~kubectl cp is fine at this size)
cat <<'PODEOF' | kubectl -n vinyl-demo apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: pvc-loader
  namespace: vinyl-demo
spec:
  restartPolicy: Never
  containers:
    - name: loader
      image: alpine:3
      command: ["sleep", "600"]
      volumeMounts:
        - { name: data, mountPath: /data/artifacts }
  volumes:
    - name: data
      persistentVolumeClaim: { claimName: vinyl-price-data }
PODEOF
kubectl -n vinyl-demo wait --for=condition=Ready pod/pvc-loader --timeout=2m
kubectl -n vinyl-demo exec pvc-loader -- mkdir -p /data/artifacts
kubectl -n vinyl-demo cp price_estimator/artifacts/vinyliq pvc-loader:/data/artifacts/
kubectl -n vinyl-demo delete pod pvc-loader --ignore-not-found=true
```

If your workstation already runs Postgres on port **5432**, run `cloud-sql-proxy`
on another port and adjust `DATABASE_URL` accordingly for local commands only.

## Phase 5 — Deploy and smoke

```bash
set -a && source .env && set +a
export IMAGE_TAG="${IMAGE_TAG:-demo}"
export AR_REPO="${AR_REPO:-vinyl-images}"

# Static manifests
kubectl apply -f k8s/demo/grader-service.yaml
kubectl apply -f k8s/demo/price-service.yaml

# Templated manifests
envsubst < k8s/demo/grader-deployment.yaml | kubectl apply -f -
envsubst < k8s/demo/price-deployment.yaml  | kubectl apply -f -
envsubst < k8s/demo/httproute.yaml         | kubectl apply -f -
kubectl apply -f k8s/demo/gateway.yaml

# Wait for rollout
kubectl -n vinyl-demo rollout status deploy/grader-api --timeout=10m
kubectl -n vinyl-demo rollout status deploy/price-api  --timeout=10m

# Verify Gateway picked up the static IP
kubectl -n vinyl-demo get gateway vinyl-demo-gw \
  -o jsonpath='{.status.addresses[0].value}'
# Expect: $STATIC_IP

# Watch CertificateMap entry until ACTIVE (5-15 min)
gcloud certificate-manager certificates describe vinyl-demo-cert \
  --format='value(managed.state)'
```

### Smoke checks

```bash
curl "https://${DEMO_HOSTNAME}/grader/health"
# {"status":"ok","model_loaded":true}

curl "https://${DEMO_HOSTNAME}/price/health"
# {"status":"ok","feature_store_count":...,"model_loaded":true}

curl -sX POST "https://${DEMO_HOSTNAME}/grader/predict" \
  -H 'Content-Type: application/json' \
  -d '{"text":"The vinyl is in perfect condition with original gloss and no marks. The jacket has minor edge wear, complete with poster NM & 4 photos Mint, as well as original back inners. An exceptional copy of a Beatles classic."}' | jq .

# Twin price check (minimal body; Redis/Postgres-backed stats only).
curl -sX POST "https://${DEMO_HOSTNAME}/price/estimate" \
  -H 'Content-Type: application/json' \
  -d '{"release_id":"456663","media_condition":"Good (G)","sleeve_condition":"Good (G)"}' | jq .

curl -sX POST "https://${DEMO_HOSTNAME}/price/estimate" \
  -H 'Content-Type: application/json' \
  -d '{"release_id":"456663","media_condition":"Near Mint (NM or M-)","sleeve_condition":"Near Mint (NM or M-)"}' | jq .
```

### Price `/estimate` — full JSON (Chrome extension parity)

Use `marketplace_client` when simulating scraped Discogs-visible values **before** trusting Redis-only rows. Omit individual keys inside `marketplace_client` if unknown; omit the whole `marketplace_client` object to use DB/Redis as-is. **`price_suggestions_json`** mirrors `GET https://api.discogs.com/marketplace/price_suggestions/{release_id}`: each key is Discogs’s **exact grade string**; entries use `value` (+ optional `currency`).

Fill in **`PLACEHOLDER_…` strings**, replace **`0`** with numeric values from Discogs/UI, keep **`null`** or delete optional keys until you substitute. Residual-trained models may return **`residual_anchor_usd`** (anchor used for reconstruction).

```bash
curl -sS -X POST "https://${DEMO_HOSTNAME}/price/estimate" \
  -H 'Content-Type: application/json' \
  -d '{
  "release_id": "456663",
  "media_condition": "Good (G)",
  "sleeve_condition": "Good (G)",
  "refresh_stats": true,
  "marketplace_client": {
    "release_lowest_price": 27.39,
    "num_for_sale": 60,
    "release_num_for_sale": 60,
    "community_want": 6171,
    "community_have": 7528,
    "blocked_from_sale": null,
    "price_suggestions_json": {
      "Mint (M)": {"value": 4682.02, "currency": "USD"},
      "Near Mint (NM or M-)": {"value": 4189.18, "currency": "USD"},
      "Very Good Plus (VG+)": {"value": 3203.49, "currency": "USD"},
      "Very Good (VG)": {"value": 2217.80, "currency": "USD"},
      "Good Plus (G+)": {"value": 1232.11, "currency": "USD"},
      "Good (G)": {"value": 739.27, "currency": "USD"},
      "Fair (F)": {"value": 429.81, "currency": "USD"},
      "Poor (P)": {"value": 246.42, "currency": "USD"}
    }
  }
}' | jq .
```

Keep **only rungs Discogs actually returned** (delete unused grades). **`price_suggestions_json`** may be a JSON **string** (escaped) instead of a nested object.

Set **`"refresh_stats": true`** to force a Discogs `GET /releases/{id}` upsert from the cluster (rewrites Postgres + Redis projection), without `marketplace_client`.

### Inspect Redis (in-cluster only)

Memorystore is on a **private RFC1918 IP** (`gcloud redis instances describe …`; `authorizedNetwork` is VPC `default`). **`redis-cli` from your workstation usually times out**. The **`price-api` image does not include `redis-cli`**, so use either a short-lived **`redis:7-alpine`** pod or **Python + `redis`** inside `price-api`.

Canonical value at **`GET vinyliq:marketplace:stats:{release_id}`**: one JSON object with **`_schema_ver: 2`** and the inference projection (same keys as Postgres/SQLite marketplace row used for `/estimate`):

| Key | Meaning |
| --- | --- |
| `release_lowest_price` | Listing floor (USD) from release/marketplace data |
| `num_for_sale` | Count used for depth features |
| `release_num_for_sale` | Often same source as listing depth |
| `community_want` / `community_have` | Community counts |
| `blocked_from_sale` | `0`/`1` or **`null`** when unknown |
| `price_suggestions_json` | **String** holding JSON: grade → `{ "value", "currency" }` (Discogs ladder) |
| `_schema_ver` | **`2`** = full projection; older entries without this may omit the ladder |

**Ephemeral `redis-cli`** (substitute Memorystore IP from `kubectl … exec … print REDIS_HOST`):

```bash
kubectl -n vinyl-demo run redis-cli-dbg --rm -it --restart=Never \
  --image=redis:7-alpine -- \
  redis-cli -h 10.xx.xx.xx -p 6379 GET "vinyliq:marketplace:stats:PLACEHOLDER_RELEASE_ID"
```

**Pretty-print + sanity-check inner ladder** (uses `redis` on the **`price`** container; set **`RELEASE_ID`** first):

```bash
export RELEASE_ID="456663"
kubectl -n vinyl-demo exec deploy/price-api -c price -- \
  env RELEASE_ID="$RELEASE_ID" python - <<'PY'
import json, os, redis

r = redis.Redis(
    host=os.environ["REDIS_HOST"],
    port=int(os.environ.get("REDIS_PORT", "6379")),
    decode_responses=True,
    socket_connect_timeout=5,
    socket_timeout=5,
)
release_id = os.environ["RELEASE_ID"].strip()
k = f"vinyliq:marketplace:stats:{release_id}"
raw = r.get(k)
if not raw:
    print("MISS", k)
else:
    d = json.loads(raw)
    meta = sorted(x for x in d if x != "price_suggestions_json")
    print("schema_ver", d.get("_schema_ver"), "other_keys", meta)
    ps = d.get("price_suggestions_json")
    lad = json.loads(ps) if isinstance(ps, str) else ps
    print("price_suggestions grades", sorted(lad.keys()) if isinstance(lad, dict) else type(lad))
PY
```

Print Memorystore coordinates from the pod env when you need **`redis-cli -h`**:

```bash
kubectl -n vinyl-demo exec deploy/price-api -c price -- \
  python -c 'import os; print(os.getenv("REDIS_HOST"), os.getenv("REDIS_PORT","6379"))'
```

## Failure modes and where to look

| Symptom | Likely cause | First check |
| --- | --- | --- |
| `grader` pod stuck in `CrashLoopBackOff` | MLflow URI unreachable from cluster | `kubectl logs -n vinyl-demo <pod>` for `ConnectionError` |
| `price` pod 503 with `model_loaded=false` | PVC populate skipped or wrong path | `kubectl -n vinyl-demo exec deploy/price-api -- ls /data/artifacts/vinyliq` |
| `price` `/estimate` flat wrong / missing ladder | Feature row missing or marketplace ladder empty | `psql "$DATABASE_URL" -c "SELECT release_id FROM releases_features WHERE release_id='456663'"`; same for `marketplace_stats` |
| `price` pod `CrashLoop`; proxy errors | Missing `roles/cloudsql.client`, wrong instance connection string, or secret `DATABASE_URL` | `kubectl logs -n vinyl-demo deploy/price-api -c cloudsql-proxy`; IAM bindings Phase 0d |
| `kubectl cp` **broken pipe** (artifacts only now) | Unstable API/network | Retry `kubectl cp`; artifacts dir should be small (~MB–low GB) |
| Cert never goes ACTIVE | Static IP not yet attached, or DNS not resolving | `dig "$DEMO_HOSTNAME"` should return `$STATIC_IP` |
| `gcloud sql instances create` **Invalid Tier … ENTERPRISE_PLUS** | Default edition is Enterprise Plus; `db-g1-small` is Enterprise-only | Add `--edition=ENTERPRISE`, or switch tier to `db-perf-optimized-N-*` for Plus |
| GHA push fails with 403 | WIF provider attribute_condition mismatch | check `attribute.repository` exactly matches `<owner>/<repo>` |

## Tear down

```bash
kubectl delete namespace vinyl-demo

gcloud certificate-manager maps entries delete vinyl-demo-default --map=vinyl-demo-certmap
gcloud certificate-manager maps delete vinyl-demo-certmap
gcloud certificate-manager certificates delete vinyl-demo-cert

gcloud compute addresses delete vinyl-demo-ip --global
gcloud redis instances delete "${REDIS_INSTANCE:-vinyl-demo-redis}" --region="$GCP_REGION"
gcloud sql instances delete vinyl-demo-db --quiet || true
gcloud container clusters delete "${GKE_CLUSTER:-vinyl-demo}" --region="$GCP_REGION"
gcloud artifacts repositories delete "${AR_REPO:-vinyl-images}" --location="$GCP_REGION"
```

WIF and runtime service accounts can stay; they cost nothing and
re-bootstrapping is one `gcloud add-iam-policy-binding` away.
