# `grader/demo/` — golden file curation

This folder owns the small "golden" fixture that drives the recorded
demo and the Playwright spec
([`tools/vinyliq_demo_playwright/`](../../tools/vinyliq_demo_playwright/)).

## What's here

- [`golden_predict_demo.json`](golden_predict_demo.json) — exactly two
  seller-comment examples (A, B) for one release ID
  (`demo_release_id=456663`, the Beatles White Album UK original mono
  "misprint" first pressing). Each example pins the comment text the
  user types into the listing form and the **exact** grade strings the
  grader is expected to predict.

The Playwright spec asserts:
1. `/predict` on `examples[i].text` returns
   `examples[i].expected_media_condition` and
   `examples[i].expected_sleeve_condition` exactly.
2. The two grade pairs differ on at least one axis (media or sleeve).
3. `/estimate` for the same `release_id` with the two grade pairs
   produces estimated prices whose absolute difference is
   `>= min_price_delta_usd`.

## Curation workflow (strict order)

> Run after the GKE deploy (or local services) is up and the grader
> + price models load cleanly.

1. **Confirm the release works for `/estimate`.**

   ```bash
   curl -sX POST "$PRICE_API_BASE/estimate" \
     -H 'Content-Type: application/json' \
     -d '{"release_id":"456663","media_condition":"Very Good (VG)","sleeve_condition":"Very Good (VG)"}' | jq .
   ```

   `estimated_price` must be non-null. If null/flat across grades,
   fall back to `7978337` (UK stereo first); update the file.

2. **Author the two `text` fields.** Distinctly different condition
   signals: A = lower-grade narrative (e.g. surface marks, edge wear,
   ring wear); B = higher-grade narrative (sealed, near-mint vinyl,
   minor sleeve scuff).

   Run each through the grader until the predictions match
   `expected_*` exactly **and** the two predictions differ on at
   least one axis:

   ```bash
   curl -sX POST "$GRADER_API_BASE/predict" \
     -H 'Content-Type: application/json' \
     -d "{\"text\":\"$(jq -r '.examples[0].text' grader/demo/golden_predict_demo.json)\"}" | jq .
   ```

   Iterate the comment text (not the expected grade) until the model
   produces what you want.

3. **Verify the price spread.** With the two grade pairs from step 2,
   call `/estimate` twice with `refresh_stats:false` and confirm the
   absolute delta is >= `min_price_delta_usd`. If it isn't, adjust
   `min_price_delta_usd` downward (modest) or pick more divergent
   grade pairs in step 2.

4. **Cross-check on the live UI.** Open the seller listing form on
   your project Discogs account and confirm both `expected_media_*`
   and `expected_sleeve_*` strings appear verbatim as `<option>` text
   in the dropdowns; same for the popup `<select>`s on the release
   page.

5. **Record actual P1 and P2 in the `notes` field** of each example so
   future readers see the demo numbers without re-running.

## Why a curated golden file

The grader is a real ML model (DistilBERT + rule postprocess) — its
predictions can drift across model versions or rule updates. Pinning
the demo to two known-stable comments avoids the recording flaking on
a confidence boundary. The Playwright spec enforces the contract: if
a future model retrain breaks one of the rows, the test fails loudly
and we re-curate before recording.

The single release (`456663`) is intentional: it makes the price-
sensitivity narrative concrete (same record, different condition,
different price) instead of abstract.

## Related files

- [`tools/vinyliq_demo_playwright/tests/demo.spec.ts`](../../tools/vinyliq_demo_playwright/tests/demo.spec.ts)
  — consumes this file via the `GOLDEN_FILE` env var.
- [`vinyliq-extension/seller-grade.js`](../../vinyliq-extension/seller-grade.js)
  — the production code path the recorded demo exercises.
