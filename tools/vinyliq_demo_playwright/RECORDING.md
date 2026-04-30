# Recording runbook

End-to-end procedure for producing the 2-minute `demo.mp4` that the
root [`README.md`](../../README.md) embeds via a `<video>` tag.

This runbook is the ONLY source of truth for the recording steps —
plan documents are the design intent, this is the operator script.

## 0. Prerequisites

Local laptop:

- `ffmpeg` installed (`brew install ffmpeg` on macOS).
- Node 20+ and `npm install` already run inside this folder
  (see [`README.md`](README.md#setup)).
- `npx playwright install chromium` already run.
- Repo-root [`.env`](../../.env) populated with the auto-captured
  values from the infra deploy (see
  [`k8s/demo/README.md`](../../k8s/demo/README.md)).

GKE side (run [`k8s/demo/README.md`](../../k8s/demo/README.md) to
completion):

- Both Deployments rolled out; `/grader/health` and `/price/health`
  return 200 against `https://$DEMO_HOSTNAME/...`.
- `gcloud certificate-manager certificates describe vinyl-demo-cert
  --format='value(managed.state)'` returns `ACTIVE`.
- The PVC contains `feature_store.sqlite`, `marketplace_stats.sqlite`,
  and `artifacts/vinyliq/xgb_model.joblib`.

Discogs / Chrome:

- A persistent Chrome profile dir already logged in to the **project**
  Discogs account (separate from any personal account). The Playwright
  fixture reads `CHROME_PROFILE_DIR` and reuses it; the session must
  not be expired.
- A live `/sell/post/<id>` listing on the project account that you
  have write access to (the Playwright spec types into and grades).

Golden file:

- [`grader/demo/golden_predict_demo.json`](../../grader/demo/golden_predict_demo.json)
  curated per [`grader/demo/README.md`](../../grader/demo/README.md).
  All four `REPLACE_AT_CURATION_TIME` placeholders gone.

## 1. Pre-record checklist (run every recording session)

```bash
set -a && source ../../.env && set +a    # repo-root .env

# Liveness
curl -sI "https://$DEMO_HOSTNAME/grader/health" | head -n1
curl -sI "https://$DEMO_HOSTNAME/price/health"  | head -n1

# Warm caches: first hit per service eats the cold-start budget
GRADER_TEXT=$(jq -r '.examples[0].text' ../../grader/demo/golden_predict_demo.json)
curl -sX POST "https://$DEMO_HOSTNAME/grader/predict" \
  -H 'Content-Type: application/json' \
  -d "{\"text\":\"$GRADER_TEXT\"}" | jq '.predictions[0]'

REL=$(jq -r '.demo_release_id' ../../grader/demo/golden_predict_demo.json)
M_A=$(jq -r '.examples[0].expected_media_condition' ../../grader/demo/golden_predict_demo.json)
S_A=$(jq -r '.examples[0].expected_sleeve_condition' ../../grader/demo/golden_predict_demo.json)
curl -sX POST "https://$DEMO_HOSTNAME/price/estimate" \
  -H 'Content-Type: application/json' \
  -d "{\"release_id\":\"$REL\",\"media_condition\":\"$M_A\",\"sleeve_condition\":\"$S_A\"}" \
  | jq '.estimated_price'

# Verify the golden file would actually pass the assertion
M_B=$(jq -r '.examples[1].expected_media_condition' ../../grader/demo/golden_predict_demo.json)
S_B=$(jq -r '.examples[1].expected_sleeve_condition' ../../grader/demo/golden_predict_demo.json)
P_B=$(curl -sX POST "https://$DEMO_HOSTNAME/price/estimate" \
  -H 'Content-Type: application/json' \
  -d "{\"release_id\":\"$REL\",\"media_condition\":\"$M_B\",\"sleeve_condition\":\"$S_B\"}" \
  | jq -r '.estimated_price')
P_A=$(curl -sX POST "https://$DEMO_HOSTNAME/price/estimate" \
  -H 'Content-Type: application/json' \
  -d "{\"release_id\":\"$REL\",\"media_condition\":\"$M_A\",\"sleeve_condition\":\"$S_A\"}" \
  | jq -r '.estimated_price')
echo "P_A=$P_A  P_B=$P_B  delta=$(python3 -c "print(abs($P_A-$P_B))")"
```

If `delta < min_price_delta_usd`, **stop** and re-curate the golden
file before recording.

## 2. Record

```bash
cd tools/vinyliq_demo_playwright
set -a && source ../../.env && set +a
npm test
```

Expected output:

- A new dir under `recordings/.../` containing `video.webm`.
- The test passes (the `expect(|p1-p2| >= MIN_PRICE_DELTA_USD)` assert).

If the test fails, read the failure mode in
[`README.md`](README.md#troubleshooting) and re-run; recordings
that don't end with the price assertion passing should be discarded.

You can re-run as many times as you want — each run produces a fresh
video. Pick the cleanest take.

### 2-minute scene budget

| # | Scene | Target time |
| --- | --- | --- |
| 1 | Open `/sell/post/<id>`, scroll to Condition notes | 0:00 - 0:08 |
| 2 | Paste comment A → click "Grade condition" → dropdowns update | 0:08 - 0:30 |
| 3 | `waitForTimeout(2500)` — viewer reads grades | 0:30 - 0:33 |
| 4 | Clear field, paste comment B → click → different grades | 0:33 - 0:55 |
| 5 | `waitForTimeout(2500)` — hold on different grades | 0:55 - 0:58 |
| 6 | Navigate to `/release/$DEMO_RELEASE_ID` | 0:58 - 1:08 |
| 7 | Twin estimate: grade pair A → P1 in overlay | 1:08 - 1:25 |
| 8 | `waitForTimeout(2000)` — hold on P1 | 1:25 - 1:27 |
| 9 | Grade pair B → P2 (different from P1) | 1:27 - 1:44 |
| 10 | `waitForTimeout(4000)` — hold on P2 for comparison | 1:44 - 1:48 |
| 11 | Tail / fade | 1:48 - 2:00 |

If a take runs > 2:05, trim `waitForTimeout` durations in
[`tests/demo.spec.ts`](tests/demo.spec.ts) before reducing `slowMo`.
If a take runs short, extend the holds at scenes 3, 5, 8, 10.

## 3. Convert to mp4

```bash
WEBM=$(ls -t recordings/*/video.webm | head -1)
ffmpeg -i "$WEBM" -c:v libx264 -crf 23 -preset fast -c:a aac demo.mp4
ls -lh demo.mp4
```

Expect 15-40 MB at 2 minutes / 1280x800 / CRF 23. GitHub's per-file
upload limit for inline assets is 100 MB; CRF 23 is comfortably under.

## 4. Publish to GitHub user-attachments

GitHub's user-attachments URLs render natively via `<video>` in
markdown — no third-party host needed.

1. Open any issue, draft a new comment, or open a PR comment box on
   this repository (the file does not have to be saved/posted).
2. Drag `demo.mp4` into the comment box.
3. GitHub uploads and replaces the dropped file with a markdown
   snippet pointing at
   `https://github.com/user-attachments/assets/<id>/<filename>`.
4. Copy that URL.
5. Cancel the comment without posting (the upload persists).

## 5. Embed in `README.md`

The Phase 9 README pass updates the root [`README.md`](../../README.md)
with the embed. The minimal markdown snippet:

```markdown
<video src="https://github.com/user-attachments/assets/<id>/demo.mp4"
       controls
       width="720"></video>
```

Re-record? Re-upload? You'll get a new user-attachments URL — update
the `src` attribute. The old URL stays valid as long as the upload
exists; nothing breaks if you take time between recording and the
README update.

## 6. Phase 7b — Manual screen-recording fallback

For any specific moment where Playwright is awkward (popup
interaction, browser-chrome animations) record the segment manually
and splice into the Playwright `.webm` in post.

- Load the extension unpacked: `chrome://extensions → Load unpacked →
  vinyliq-extension/`. Confirm the version is 0.2.0.
- Record with QuickTime (macOS Cmd-Shift-5 → Record selected portion
  → start), OBS, or `ffmpeg -f avfoundation`.
- Repeat the same seller flow + release twin estimates against the
  same golden rows so the spliced segment is consistent with the
  Playwright take.

## Quick reference: failure cheat sheet

| Symptom during recording | First check |
| --- | --- |
| 404 at the gateway | URLRewrite mismatch — pod logs should show `/predict`, `/estimate`, not `/grader/predict` |
| Estimate spread less than `min_price_delta_usd` | Re-curate the golden file (different grade pairs) before re-recording |
| Cold start on the grader > 60 s | Bump `startupProbe.failureThreshold` in `k8s/demo/grader-deployment.yaml` |
| Redis unreachable from price pod | Graceful fallback should keep the demo working — check pod logs for "Redis connect failed" warning, not error |
| Playwright "Extension service worker not found" | `--load-extension` rejected — `EXTENSION_PATH/manifest.json` malformed or `manifest_version` not 3 |
| `chrome.storage.sync.set` returns undefined | Popup page didn't load — re-run with `npm run test:show` to inspect |
