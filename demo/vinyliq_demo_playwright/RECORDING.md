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
  If **`PLAYWRIGHT_CONNECT_CDP`** is set (**Google Chrome over remote debugging port** instead of bundled Chromium): export **`EXTENSION_PATH`** to **`vinyliq-extension`** (derives unpacked id — avoids hangs when CDP logs **`ext_sw=0`**) **or** **`VINYLIQ_EXTENSION_ID`** from **`chrome://extensions`**; **`CHROME_PROFILE_DIR`** is unused. Details: [`demo/vinyliq_demo_playwright/README.md`](./README.md#cannot-pass-cloudflare-in-playwrights-chromium).
- A live `/sell/post/<id>` listing on the project account that you
  have write access to (the Playwright spec types into and grades).

Golden file:

- [`grader/demo/golden_predict_demo.json`](../../grader/demo/golden_predict_demo.json)
  curated per [`grader/demo/README.md`](../../grader/demo/README.md).
  All four `REPLACE_AT_CURATION_TIME` placeholders gone.
  **`search_query`** and **`demo_master_id`** are required whenever **`DEMO_CATALOG_UX=1`** (default): **`search_query`** drives navbar search → Masters facet (`type=master`); **`demo_master_id`** opens the Hub, then **`/release/{demo_release_id}`**.

Scripted Demo (Playwright):

- Steps: [`fixtures/default_demo.script.json`](./fixtures/default_demo.script.json). Alternate path **`VINYLIQ_DEMO_SCRIPT`**.
  Recorder tab opens **`https://www.discogs.com/`** (**`DEMO_START_URL`**), then **`goto_seller_listing`** runs **catalog search → Masters → `/master/` → `/release/` → Sell** unless **`DEMO_CATALOG_UX=0`** (see **`npm run test:ci`**). **`inject`** still uses **`popup.html`**. **`#vinyliq-sell-dock`** gates grading/estimate.
- Fast passes: **`DEMO_SKIP_HOLDS=1`**. **`npm run test:ci`** also sets **`DEMO_CATALOG_UX=0`** (straight **`sell_post`**). Add **`DEMO_CATALOG_UX=1`** to CI if you need the catalogue walk.
- Legacy **`DEMO_FULL_WALKTHROUGH=1`** **forces catalog UX**, same **`discogs_navigation.ts`** helpers.

## 1. Pre-record checklist (run every recording session)

```bash
set -a && source ../../.env && set +a    # repo-root .env

# Liveness (GET — ``curl -I`` uses HEAD which often returns HTTP 405 on FastAPI mounts)
curl -sS -D - -o /dev/null "https://$DEMO_HOSTNAME/grader/health" | head -n1
curl -sS -D - -o /dev/null "https://$DEMO_HOSTNAME/price/health" | head -n1

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
cd demo/vinyliq_demo_playwright
set -a && source ../../.env && set +a
npm test
```

Expected output:

- A new **`.webm`** under **`recordings/`** (from Chromium **`launchPersistentContext`** **`recordVideo`** — see **`fixtures/extension.ts`** and **`fixtures/demo_video_ann.ts`**). Filename is opaque until the context closes cleanly. **The video is real-time:** **`slowMo`** and **`DEMO_COMMENT_TYPING_DELAY_MS`** are baked into each frame (`ffmpeg` mp4 stays **1×** unless you stretch the timeline in post).
- `showActions`: lower-right overlays label each atomic Playwright action (fills, navigations).
- Scripted chapter cards (`DEMO_VIDEO_CHAPTERS` default `1`; disable with `0`) blur the viewport for ~`DEMO_VIDEO_CHAPTER_MS` (default **12400** ms) so viewers can read setup narration; seller-beat strips use **`DEMO_SELLER_STRIP_READ_MS`** / **`DEMO_SELLER_STRIP_MS`** (**`fixtures/demo_video_ann.ts`** defaults ~**7200** / ~**26000** ms); segues use **`DEMO_SEGUE_STRIP_MS`** / **`DEMO_SEGUE_STRIP_READ_MS`**; after each estimate overlay + Copy A/B seller estimate strips: **`DEMO_AFTER_FIRST_ESTIMATE_BARE_MS`** (default **9800**) before **`after_first_estimate`** or session outro (raise e.g. **19000** for a lazier Copy B outro).
- The test passes (the `expect(|p1-p2| >= MIN_PRICE_DELTA_USD)` assert).

### Hybrid operator mode (`DEMO_HYBRID=1`)

The harness still scripts **inject** (popup tab), **`goto_seller_listing`** narration cards, **`hold`** delays, annotation strips, and **typing** golden condition notes. With **catalog UX** (**`DEMO_CATALOG_UX=1`**, typical for recording), drive **manual search → master → your golden release URL on the recorder tab**: when the pathname matches **`/release/{golden release}`**, the harness runs the same release-page fullscreen narration as the fully scripted path (**Hmm…**, scroll, confirmation), then you continue to **`discogs.com/sell/post/<id>`** (same **`DEMO_HYBRID_NAV_TIMEOUT_MS`** / step budgets). Deep-link (**`DEMO_CATALOG_UX=0`**) skips that release pause. Then click **Grade condition** until grades match the ladder and **Get estimate** when prompted. Budget envs: **`DEMO_HYBRID_NAV_TIMEOUT_MS`**, **`DEMO_HYBRID_STEP_TIMEOUT_MS`**, optional **`DEMO_HYBRID_PAUSE=1`** (**`page.pause()`** after each typed block).

If the test fails, read the failure mode in
[`README.md`](README.md#troubleshooting) and re-run; recordings
that don't end with the price assertion passing should be discarded.

You can re-run as many times as you want — each run produces a fresh
video. Pick the cleanest take.

### 2-minute scene budget

| # | Scene | Notes |
| --- | --- | --- |
| 1 | Landing on **`/sell/post`** (or full catalog UX before it) | — |
| 2 | Typed rough seller note → **Grade condition** | — |
| 3 | Short **`hold`** (~**0.5s**) after first grade (+ optional **`after_first_grade`** strip when **`DEMO_VIDEO_CHAPTERS=1`**) | Tweak **`hold.ms`** |
| 4 | Replace note → **Grade condition** (stronger copy) | — |
| 5 | Optional **`after_second_grade`** segue when **`DEMO_VIDEO_CHAPTERS=1`** (same pacing as **`after_first_grade`**), then **`hold`** (~**1.1s**) | — |
| 6 | **Get estimate** (rough-copy grades) → overlay (+ seller strip timings) | — |
| 7 | After Copy **A** seller estimate strip **read lead** (trimmed vs generic seller strip): **`DEMO_AFTER_FIRST_ESTIMATE_BARE_MS`** (**~9800 ms** default) → **`after_first_estimate`** + tiny scripted **`hold`** (~**400 ms**) | **`fixtures/demo_video_ann.ts`** (+ **`DEMO_SEGUE_AFTER_FIRST_ESTIMATE_MS`** for that segue’s on-screen time) |
| 8 | **Get estimate** (second grades) → overlay + estimate insight strip (mirror of scene **6**) | — |
| 9 | Second-estimate **`DEMO_AFTER_FIRST_ESTIMATE_BARE_MS`** + session **outro** when chapters on (mirror of **7**, outro replaces **`after_first_estimate`**) + **`hold`** (~**1.6s**) | — |
| 10 | Assert price wedge + tail space | — |

If a take feels slack or long, shorten **`hold.ms`** in
[**`fixtures/default_demo.script.json`**](fixtures/default_demo.script.json),
then tighten **`DEMO_VIDEO_CHAPTER_MS`**, **`DEMO_SELLER_STRIP_*`**, **`DEMO_SEGUE_STRIP_*`**, and **`DEMO_AFTER_FIRST_ESTIMATE_BARE_MS`** (**`fixtures/demo_video_ann.ts`**), then **`slowMo`** in [**`playwright.config.ts`**](./playwright.config.ts) (**`fixtures/extension.ts`** uses the same default when launching with video).
If a take runs short, raise those annotation env defaults first—the JSON **`hold`** steps (**3**, **5**, **9**) are intentionally small whenever chapter overlays supply the pacing.

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

For any specific moment where Playwright is awkward (browser-chrome animations) record the segment manually and splice into the Playwright `.webm` in post.

- Load the extension unpacked: `chrome://extensions → Load unpacked →
  vinyliq-extension/`. Confirm the seller dock appears on drafts (see extension `manifest.json` version).
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
