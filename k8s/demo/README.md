# GKE Autopilot demo runbook (`vinyl-demo` namespace)

End-to-end runbook for the **demo wave 1** deploy: GCP bootstrap, CI auth
(WIF), Workload Identity, secrets, PVC populate, deploy + smoke. Sibling
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
        Price --> PVC[(PVC /data SQLite + xgb_model.joblib)]
        Price --> Redis[(Memorystore Redis Basic)]
    end
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
GCS_BUCKET=...

MLFLOW_TRACKING_URI=https://mlflow.your-host
MLFLOW_MODEL_URI=models:/VinylGrader/latest

DISCOGS_USER_TOKEN=...
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
  gateway.googleapis.com \
  networkservices.googleapis.com

gcloud artifacts repositories create "${AR_REPO:-vinyl-images}" \
  --repository-format=docker --location="$GCP_REGION"

gcloud container clusters create-auto "${GKE_CLUSTER:-vinyl-demo}" \
  --region="$GCP_REGION" \
  --gateway-api=standard \
  --workload-pool="$GCP_PROJECT_ID.svc.id.goog"

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

# Bind KSA `vinyl-demo/vinyl-runtime` to this GSA
gcloud iam service-accounts add-iam-policy-binding \
  "vinyl-demo-runtime@${GCP_PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/iam.workloadIdentityUser" \
  --member="serviceAccount:${GCP_PROJECT_ID}.svc.id.goog[vinyl-demo/vinyl-runtime]"

echo "RUNTIME_GSA=vinyl-demo-runtime@${GCP_PROJECT_ID}.iam.gserviceaccount.com" >> .env
set -a && source .env && set +a
```

## Phase 4 — Apply manifests, create secrets, populate PVC

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

kubectl -n vinyl-demo create secret generic vinyl-redis \
  --from-literal=REDIS_HOST="$REDIS_HOST"

kubectl -n vinyl-demo create secret generic vinyl-discogs \
  --from-literal=DISCOGS_USER_TOKEN="$DISCOGS_USER_TOKEN"
```

`secrets.example.yaml` documents the expected key inventory but is
intentionally not applied — values stay out of version control.

### 3. ConfigMaps + PVC

```bash
kubectl apply -f k8s/demo/configmap.yaml
kubectl apply -f k8s/demo/price-config.yaml
kubectl apply -f k8s/demo/price-pvc.yaml
```

### 4. Populate the PVC (one-time, before the price-api Deployment rolls)

Spin up an idle alpine pod that mounts the PVC, then `kubectl cp` the
local SQLite + trained model files into it.

```bash
# Sanity checks before copying — the demo release_id is 456663 (Beatles
# White Album original mono first pressing).
sqlite3 price_estimator/data/cache/marketplace_stats.sqlite \
  "SELECT release_id FROM marketplace_stats WHERE release_id='456663';"
ls price_estimator/artifacts/vinyliq/xgb_model.joblib

# Optional: shrink SQLite before transfer
sqlite3 price_estimator/data/feature_store.sqlite "VACUUM;"
sqlite3 price_estimator/data/cache/marketplace_stats.sqlite "VACUUM;"

# Loader pod
cat <<'EOF' | kubectl -n vinyl-demo apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: pvc-loader
spec:
  restartPolicy: Never
  containers:
    - name: loader
      image: alpine:3
      command: ["sleep", "3600"]
      volumeMounts:
        - { name: data, mountPath: /data }
  volumes:
    - name: data
      persistentVolumeClaim: { claimName: vinyl-price-data }
EOF
kubectl -n vinyl-demo wait --for=condition=Ready pod/pvc-loader --timeout=2m

kubectl -n vinyl-demo exec pvc-loader -- mkdir -p /data/cache /data/artifacts/vinyliq

kubectl -n vinyl-demo cp price_estimator/data/feature_store.sqlite \
  pvc-loader:/data/
kubectl -n vinyl-demo cp price_estimator/data/cache/marketplace_stats.sqlite \
  pvc-loader:/data/cache/
kubectl -n vinyl-demo cp price_estimator/artifacts/vinyliq \
  pvc-loader:/data/artifacts/

kubectl -n vinyl-demo delete pod pvc-loader
```

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
kubectl -n vinyl-demo rollout status deploy/price-api  --timeout=5m

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
  -d '{"text":"VG+ sleeve, light hairlines, plays well"}' | jq .

# Twin price check (driven by the demo golden file in the verify branch)
curl -sX POST "https://${DEMO_HOSTNAME}/price/estimate" \
  -H 'Content-Type: application/json' \
  -d '{"release_id":"456663","media_condition":"Good (G)","sleeve_condition":"Good (G)"}' | jq .

curl -sX POST "https://${DEMO_HOSTNAME}/price/estimate" \
  -H 'Content-Type: application/json' \
  -d '{"release_id":"456663","media_condition":"Near Mint (NM or M-)","sleeve_condition":"Near Mint (NM or M-)"}' | jq .

# Verify Redis populated (debug pod with redis-cli)
kubectl -n vinyl-demo run redis-cli --rm -it --restart=Never \
  --image=redis:7-alpine -- \
  redis-cli -h "$REDIS_HOST" GET "vinyliq:marketplace:stats:456663"
```

## Failure modes and where to look

| Symptom | Likely cause | First check |
| --- | --- | --- |
| `grader` pod stuck in `CrashLoopBackOff` | MLflow URI unreachable from cluster | `kubectl logs -n vinyl-demo <pod>` for `ConnectionError` |
| `price` pod 503 with `model_loaded=false` | PVC populate skipped or wrong path | `kubectl -n vinyl-demo exec deploy/price-api -- ls /data/artifacts/vinyliq` |
| `price` `/estimate` returns the same number for any condition | Feature store missing the release_id | `sqlite3 .../feature_store.sqlite "SELECT * FROM ... WHERE release_id='456663'"` |
| Cert never goes ACTIVE | Static IP not yet attached, or DNS not resolving | `dig "$DEMO_HOSTNAME"` should return `$STATIC_IP` |
| GHA push fails with 403 | WIF provider attribute_condition mismatch | check `attribute.repository` exactly matches `<owner>/<repo>` |

## Tear down

```bash
kubectl delete namespace vinyl-demo

gcloud certificate-manager maps entries delete vinyl-demo-default --map=vinyl-demo-certmap
gcloud certificate-manager maps delete vinyl-demo-certmap
gcloud certificate-manager certificates delete vinyl-demo-cert

gcloud compute addresses delete vinyl-demo-ip --global
gcloud redis instances delete "${REDIS_INSTANCE:-vinyl-demo-redis}" --region="$GCP_REGION"
gcloud container clusters delete "${GKE_CLUSTER:-vinyl-demo}" --region="$GCP_REGION"
gcloud artifacts repositories delete "${AR_REPO:-vinyl-images}" --location="$GCP_REGION"
```

WIF and runtime service accounts can stay; they cost nothing and
re-bootstrapping is one `gcloud add-iam-policy-binding` away.
