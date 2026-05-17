# Pitch recording runbook (release 12830828)

Operator script for a **short, live-narrated** product pitch: one record you own,
condition grading on the seller form, one price estimate on the release page.

- Playwright assist: [`tests/pitch-assist.spec.ts`](tests/pitch-assist.spec.ts)
  opens release → your sell listing → **types the golden comment only**; you do
  Grade, estimate, and overlay yourself while screen recording.
- Golden file: [`grader/demo/golden_predict_demo_pitch.json`](../../grader/demo/golden_predict_demo_pitch.json)
- **You** capture screen + mic (QuickTime / OBS); Playwright drives the browser only.
- The 2-minute README demo stays on [`RECORDING.md`](RECORDING.md) + `golden_predict_demo.json`.

## 0. Prerequisites

**Local**

- Node 20+, `npm install` and `npx playwright install chromium` in this folder.
- Repo-root [`.env`](../../.env) from your last demo deploy (`GCP_*`, `DEMO_HOSTNAME`,
  `REDIS_HOST`, `DATABASE_URL`, `MLFLOW_*`, `DISCOGS_USER_TOKEN`, `IMAGE_TAG`, …).
- `CHROME_PROFILE_DIR` — persistent profile logged into the **project** Discogs account.
  If bundled Chromium keeps asking you to log in, seed it first (same as the 2-minute demo):
  `cd demo/vinyliq_demo_playwright && npm run warm-profile` (login-only; press Enter when done).
  See [`README.md`](README.md#cannot-pass-cloudflare-in-playwrights-chromium).
- `EXTENSION_PATH` — absolute path to [`vinyliq-extension/`](../../vinyliq-extension/).
- Golden file curated (§4); all `REPLACE_*` placeholders removed.

**After Gateway is up**

```bash
export PRICE_API_BASE="https://${DEMO_HOSTNAME}/price"
export GRADER_API_BASE="https://${DEMO_HOSTNAME}/grader"
```

## 1. What did tear-down remove?

Full tear-down is documented in [`k8s/demo/README.md` — Tear down](../../k8s/demo/README.md)
(namespace, cert map, static IP, Redis, Cloud SQL, cluster, Artifact Registry).

**Quick checks** (repo root):

```bash
set -a && source .env && set +a
gcloud container clusters describe "${GKE_CLUSTER:-vinyl-demo}" --region="$GCP_REGION" 2>/dev/null && echo "cluster: yes" || echo "cluster: no"
gcloud redis instances describe "${REDIS_INSTANCE:-vinyl-demo-redis}" --region="$GCP_REGION" 2>/dev/null && echo "redis: yes" || echo "redis: no"
gcloud sql instances describe vinyl-demo-db 2>/dev/null && echo "sql: yes" || echo "sql: no"
gcloud compute addresses describe vinyl-demo-ip --global 2>/dev/null && echo "static-ip: yes" || echo "static-ip: no"
kubectl get namespace vinyl-demo 2>/dev/null && echo "namespace: yes" || echo "namespace: no"
```

| Situation | Follow |
| --- | --- |
| **A.** Namespace gone; cluster, Redis, SQL, static IP, certs still exist | [§2 Resume workloads](#2-resume-workloads-only) |
| **B.** Namespace gone; SQL paused/empty or PVC lost | §2 + [§3 Data reload](#3-data-reload-if-cloud-sql-or-pvc-empty) |
| **C.** Cluster / SQL / Redis / IP gone | [§2b Full re-spin](#2b-full-re-spin-gcp) → [`k8s/demo/README.md`](../../k8s/demo/README.md) Phases 0→5 |

## 2. Resume workloads only

Use when GCP infra and **Postgres + PVC data** still exist (typical after
`kubectl delete namespace vinyl-demo` only).

**Cloud SQL paused?**

```bash
gcloud sql instances patch vinyl-demo-db --activation-policy=ALWAYS
# wait until instance state is RUNNABLE
```

**Cert state**

```bash
gcloud certificate-manager certificates describe vinyl-demo-cert \
  --format='value(managed.state)'
```

Re-run [`k8s/demo/README.md`](../../k8s/demo/README.md) Phase 0b only if cert/map were deleted.

**Deploy**

```bash
set -a && source .env && set +a
export IMAGE_TAG="${IMAGE_TAG:-demo}"
export AR_REPO="${AR_REPO:-vinyl-images}"

kubectl apply -f k8s/demo/namespace.yaml
envsubst < k8s/demo/serviceaccount.yaml | kubectl apply -f -

# First time: use `kubectl create secret` from k8s/demo/README.md.
# Retries: idempotent apply:
kubectl -n vinyl-demo create secret generic vinyl-mlflow \
  --from-literal=MLFLOW_TRACKING_URI="$MLFLOW_TRACKING_URI" \
  --from-literal=MLFLOW_MODEL_URI="$MLFLOW_MODEL_URI" \
  --dry-run=client -o yaml | kubectl apply -f -
kubectl -n vinyl-demo create secret generic vinyl-redis \
  --from-literal=REDIS_HOST="$REDIS_HOST" \
  --dry-run=client -o yaml | kubectl apply -f -
kubectl -n vinyl-demo create secret generic vinyl-discogs \
  --from-literal=DISCOGS_TOKEN="${DISCOGS_TOKEN:-$DISCOGS_USER_TOKEN}" \
  --from-literal=DISCOGS_USER_TOKEN="${DISCOGS_USER_TOKEN:-$DISCOGS_TOKEN}" \
  --dry-run=client -o yaml | kubectl apply -f -
kubectl -n vinyl-demo create secret generic vinyl-cloudsql \
  --from-literal=DATABASE_URL="$DATABASE_URL" \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl apply -f k8s/demo/configmap.yaml
kubectl apply -f k8s/demo/price-config.yaml
kubectl apply -f k8s/demo/price-pvc.yaml

kubectl apply -f k8s/demo/grader-service.yaml
kubectl apply -f k8s/demo/price-service.yaml
kubectl apply -f k8s/demo/price-healthcheck-policy.yaml
envsubst < k8s/demo/grader-deployment.yaml | kubectl apply -f -
envsubst < k8s/demo/price-deployment.yaml  | kubectl apply -f -
envsubst < k8s/demo/httproute.yaml         | kubectl apply -f -
kubectl apply -f k8s/demo/gateway.yaml

kubectl -n vinyl-demo rollout status deploy/grader-api --timeout=10m
kubectl -n vinyl-demo rollout status deploy/price-api  --timeout=10m
```

**Smoke**

```bash
curl -s "https://${DEMO_HOSTNAME}/grader/health" | jq .
curl -s "https://${DEMO_HOSTNAME}/price/health" | jq .
```

## 2b. Full re-spin (GCP)

If §1 checks show missing cluster, SQL, Redis, or static IP, work through
[`k8s/demo/README.md`](../../k8s/demo/README.md) in order:

| Phase | Content |
| --- | --- |
| 0 | APIs, cluster, Redis, static IP, `.env` capture |
| 0b | Certificate Manager map (if deleted) |
| 0d | Runtime Workload Identity (SAs may still exist) |
| 0e | Cloud SQL (if deleted; re-append `DATABASE_URL`) |
| 4 | Secrets, PVC, schema + loader + `pvc-loader` model copy |
| 5 | Deploy + smoke |

Pitch-specific smoke uses release `12830828` (§4), not demo `456663`.

## 3. Data reload (if Cloud SQL or PVC empty)

Required when `/price/estimate` for `12830828` returns null or `model_loaded=false`.

Follow [`k8s/demo/README.md` Phase 4.4](../../k8s/demo/README.md): `cloud-sql-proxy`,
`psql` + `schema.sql`, `sqlite_to_cloudsql_loader.py`, `pvc-loader` + `kubectl cp`
for `price_estimator/artifacts/vinyliq`.

**Pitch sanity**

```bash
psql "$DATABASE_URL" -c "SELECT release_id FROM marketplace_stats WHERE release_id='12830828';"
psql "$DATABASE_URL" -c "SELECT release_id FROM releases_features WHERE release_id='12830828';"
```

If rows are missing, ensure the release exists in your local SQLite sources before
re-running the loader, or ingest that release locally first (see `price_estimator/README.md`).

## 4. Curate golden + pre-flight

Edit [`grader/demo/golden_predict_demo_pitch.json`](../../grader/demo/golden_predict_demo_pitch.json):
`demo_release_id` `12830828`, your `sell_post_url`, one `examples[0]` with `text` and
`expected_*` grades. Curation workflow: [`grader/demo/README.md`](../../grader/demo/README.md)
(single example).

```bash
set -a && source .env && set +a
REL=12830828
TEXT=$(jq -r '.examples[0].text' grader/demo/golden_predict_demo_pitch.json)
M=$(jq -r '.examples[0].expected_media_condition' grader/demo/golden_predict_demo_pitch.json)
S=$(jq -r '.examples[0].expected_sleeve_condition' grader/demo/golden_predict_demo_pitch.json)

curl -sI "https://$DEMO_HOSTNAME/grader/health" | head -n1
curl -sI "https://$DEMO_HOSTNAME/price/health"  | head -n1

curl -sX POST "https://$DEMO_HOSTNAME/grader/predict" \
  -H 'Content-Type: application/json' \
  -d "{\"text\":\"$TEXT\"}" | jq '.predictions[0]'

curl -sX POST "https://$DEMO_HOSTNAME/price/estimate" \
  -H 'Content-Type: application/json' \
  -d "{\"release_id\":\"$REL\",\"media_condition\":\"$M\",\"sleeve_condition\":\"$S\",\"refresh_stats\":false}" | jq .
```

Warm both services once before recording (cold start can be slow).

## 5. Record the pitch

1. Confirm extension API bases in the popup match `PRICE_API_BASE` / `GRADER_API_BASE`.
2. **Start your screen recorder** (mic on).
3. Run assist:

```bash
cd demo/vinyliq_demo_playwright
set -a && source ../../.env && set +a
export GOLDEN_FILE=../../grader/demo/golden_predict_demo_pitch.json
export DEMO_RELEASE_ID=12830828
export PRICE_API_BASE="https://${DEMO_HOSTNAME}/price"
export GRADER_API_BASE="https://${DEMO_HOSTNAME}/grader"
# optional timing:
# export PITCH_HOLD_ON_RELEASE_MS=3000
# export PITCH_PAUSE_BEFORE_TYPE_MS=3000
# export DEMO_COMMENT_TYPING_DELAY_MS=38

npm run assist:pitch
```

When typing finishes, the browser **stays open** until you press **Enter** in the
terminal (finish Grade / estimate / overlay on your own timeline).

### Scene budget

| # | Scene | Who |
| --- | --- | --- |
| 1 | Open `/release/12830828` (brief hold) | Playwright |
| 2 | Open your `sell_post_url` | Playwright |
| 3 | Pause on empty comment field (narrate) | You (timing via `PITCH_PAUSE_BEFORE_TYPE_MS`) |
| 4 | Condition comment typed | Playwright — **assist stops here** |
| 5 | Grade on sell dock → release page → estimate overlay | You (extension) |
| 6 | Press **Enter** in terminal to close Chromium | You |

4. Stop your screen recorder; export e.g. `pitch-demo.mp4` locally.

**Audio:** Playwright video has no mic track. Your QuickTime/OBS file is the pitch asset.

## 6. After pitch (optional)

```bash
gcloud sql instances patch vinyl-demo-db --activation-policy=NEVER
# kubectl delete namespace vinyl-demo   # workloads only; keeps GCP infra
```

Full cleanup: [`k8s/demo/README.md` — Tear down](../../k8s/demo/README.md).

## 7. Troubleshooting

| Symptom | Check |
| --- | --- |
| 404 on `/grader/health` | Gateway up; cert `ACTIVE`; `DEMO_HOSTNAME` in `.env` |
| Estimate null for `12830828` | §3 data reload; DB rows for release |
| Hang on comment field | Discogs layout; selectors in `demo.spec.ts` / `pitch-assist.spec.ts` |
| Grade does nothing | Extension on sell page; `GRADER_API_BASE` in `chrome.storage.sync` |
| “type …” caption on screen | Re-run with current `pitch-assist` (`recordVideo` off). Update repo if you still see captions |
| No price overlay | Service worker present; see [`README.md`](README.md#troubleshooting) |
| Inspector never appears | Run headed; `assist:pitch` uses `headless: false` |
