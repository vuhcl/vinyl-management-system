# vinyliq_demo_playwright

End-to-end demo automation for the [VinylIQ extension](../../vinyliq-extension/)
running against the GKE-deployed price + grader APIs. Drives the full
flow used in the recorded demo:

1. **Seller listing** (`/sell/post/<id>`): paste comment A → "Grade
   condition" button → media + sleeve dropdowns update; repeat with
   comment B and assert the dropdown values change.
2. **Release page** (`/release/<id>`): twin price estimates (grade pair
   A then grade pair B) rendered via the same overlay the popup uses;
   asserts `|p1 - p2| >= MIN_PRICE_DELTA_USD`.

The golden file (`grader/demo/golden_predict_demo.json`) is the source
of truth for both halves: comment text, expected grades, and the
release ID.

## Why this is its own project

It lives under `tools/` (per repo hygiene rules — `scripts/` is reserved
for actual pipelines) and uses npm rather than uv because Playwright
itself is a Node ecosystem tool. No Python is involved here.

## Prerequisites

- Node 20+
- A pre-authenticated Chrome profile for the project's Discogs account
  (logging in via Playwright is fragile; reuse a manually-prepared
  profile dir).
- An accessible seller listing URL on that account
  (`https://www.discogs.com/sell/post/<id>`).
- The GKE demo deploy reachable (or local uvicorn on the default ports).

## Setup

```bash
cd tools/vinyliq_demo_playwright
npm install
npm run install-browsers      # downloads Playwright's chromium build
```

## Configuration (env)

Set via repo-root `.env` and `set -a && source .env && set +a`, or via a
local `.env` in this folder. The recording runbook in `RECORDING.md`
lists the canonical defaults.

| Var | Default | Notes |
| --- | --- | --- |
| `CHROME_PROFILE_DIR` | _(required)_ | Persistent profile dir already logged in to Discogs (project account) |
| `EXTENSION_PATH` | _(required)_ | Absolute path to `vinyliq-extension/` |
| `PRICE_API_BASE` | `http://127.0.0.1:8801` | Price API base URL (no trailing slash) |
| `GRADER_API_BASE` | `http://127.0.0.1:8090` | Grader API base URL (no trailing slash) |
| `SELL_POST_URL` | from golden file | Seller listing URL on project account |
| `DEMO_RELEASE_ID` | from golden file | Discogs release ID; defaults to 456663 |
| `MIN_PRICE_DELTA_USD` | from golden file | Min `|p1-p2|` for the twin assertion |
| `GOLDEN_FILE` | `../../grader/demo/golden_predict_demo.json` | Golden file path |
| `VINYLIQ_API_KEY` | _(empty)_ | Sent as `X-API-Key` to both services |

## Run

```bash
set -a && source ../../.env && set +a   # or source the local .env
npm test
```

A successful run leaves a `.webm` recording at
`recordings/<spec>/<test>/video.webm`. The recording runbook
(`RECORDING.md`) documents the ffmpeg conversion and GitHub upload
steps that turn that into the embeddable `demo.mp4`.

## Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `Missing required env var: CHROME_PROFILE_DIR` | profile not exported | `export CHROME_PROFILE_DIR=...` (no quotes around `~`) |
| Test hangs at `await commentEl.waitFor` | Discogs page loaded with a new layout | re-discover selectors interactively, update both `seller-grade.js` and `tests/demo.spec.ts` |
| `Extension service worker not found` | `--load-extension` rejected | check `EXTENSION_PATH/manifest.json` is valid JSON, version 0.2.0+ |
| Estimate returns 401 | `VINYLIQ_API_KEY` mismatch | unset on server side, or set the same value here |
| `expect.poll` times out on the seller selectors | grader confidence on golden text is low and rules changed the grade | re-curate the golden file rows against the live grader |

## How the price-estimate path differs from a live demo

Playwright cannot open the MV3 toolbar-action popup
([upstream issue](https://github.com/microsoft/playwright/issues/5593)),
so the spec drives the price API directly through Playwright's
service-worker context — which uses the same fetch path and host
permissions the production popup uses — and then sends `SHOW_OVERLAY`
to the existing `content.js` handler so the rendered overlay is
indistinguishable from a popup-driven flow in the recording.

If you want a true popup-driven take for the recording, follow the
manual screen-recording fallback in `RECORDING.md` ("Phase 7b") and
splice it into the Playwright video in post.
