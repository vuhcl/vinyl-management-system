import { test } from "@playwright/test";

import { demoCatalogUxEnabled } from "../fixtures/discogs_navigation";

import {
  launchWithExtension,
  disposeDemoBrowser,
} from "../fixtures/extension";
import { loadDemoScript, runDemoScript, demoHybridOperatorMode, demoHybridSuggestedTestTimeoutMs } from "../fixtures/demo_runner";
import { readGoldenPredictDemo } from "../fixtures/golden";

/**
 * VinylIQ scripted demo (see ``fixtures/default_demo.script.json``).
 *
 * Env:
 *   PRICE_API_BASE, GRADER_API_BASE, VINYLIQ_API_KEY
 *   DEMO_RELEASE_ID, DEMO_MASTER_ID (optional override), MIN_PRICE_DELTA_USD, GOLDEN_FILE
 *   DEMO_START_URL — first **`goto`** on the recorder tab (**default:** **`https://www.discogs.com/`**
 *     when **`DEMO_CATALOG_UX=1`**, otherwise **`SELL_POST_URL`**)
 *   DEMO_CATALOG_UX — **`1`** (default): **`goto_seller_listing`** performs home → **`search_query`**
 *     → **Masters** results → **`/master/{DEMO_MASTER_ID}`** → **`/release/{DEMO_RELEASE_ID}`** → **Sell** → `/sell/post/…`. **`0`** (and **`npm run test:ci`**) deep-links **`SELL_POST_URL`**.
 *   DEMO_DEEP_LINK_SELL_POST=1 — same as opting out of catalog UX
 *   VINYLIQ_DEMO_SCRIPT — alternate JSON playbook path
 *   DEMO_SKIP_HOLDS — fast CI skips ``hold`` ms waits
 *   DEMO_FULL_WALKTHROUGH=1 — legacy alias; forces catalog UX **on**
 *   DEMO_HYBRID — **`1`** keeps **annotations** + **typing** condition comments scripted; **you** complete navigation (script waits for **`/sell/post/…`** unless already there), Grade, **Get estimate**, etc.
 *   DEMO_HYBRID_STEP_TIMEOUT_MS, DEMO_HYBRID_NAV_TIMEOUT_MS — per-beat / first-sell-page budgets (see **`demoHybridSuggestedTestTimeoutMs`**)
 *   DEMO_HYBRID_PAUSE — **`1`** invokes **`page.pause()`** after each automated comment block (Inspector)
 *
 * **`inject_extension_storage`** briefly opens **`popup.html`** in a **separate** tab to
 * seed ``chrome.storage.sync``, then (**after** that tab closes) runs the fullscreen
 * inject narration on the **Discogs recorder** tab (`fixtures/demo_runner.ts` order).
 * is asserted after **`goto_seller_listing`** (skips **`goto`** when already **`SELL_POST_URL`** **and**
 * catalog UX is off).
 */

function envOr(name: string, fallback: string): string {
  const v = process.env[name];
  return v && v.trim().length > 0 ? v : fallback;
}

test("vinyliq demo scripted", async ({}, testInfo) => {
  if (demoHybridOperatorMode()) {
    test.setTimeout(demoHybridSuggestedTestTimeoutMs());
  }

  const golden = readGoldenPredictDemo({ minExamples: 2 });
  const examples = golden.examples;
  if (!examples || examples.length < 2) {
    throw new Error("Golden needs ≥2 examples.");
  }

  const releaseId = envOr("DEMO_RELEASE_ID", String(golden.demo_release_id));
  const sellPostUrl = envOr("SELL_POST_URL", golden.sell_post_url);
  const minDelta = Number(
    envOr("MIN_PRICE_DELTA_USD", String(golden.min_price_delta_usd ?? 100)),
  );
  const priceApiBase = envOr("PRICE_API_BASE", "http://127.0.0.1:8801");
  const graderApiBase = envOr("GRADER_API_BASE", "http://127.0.0.1:8090");
  const apiKey = process.env.VINYLIQ_API_KEY ?? "";

  const { context, extensionId } = await launchWithExtension();
  testInfo.annotations.push({ type: "extension-id", description: extensionId });

  const seller = await context.newPage();
  const discogsLanding = demoCatalogUxEnabled()
    ? envOr("DEMO_START_URL", "https://www.discogs.com/")
    : envOr("DEMO_START_URL", sellPostUrl);
  await seller.goto(discogsLanding, {
    waitUntil: "domcontentloaded",
    timeout: 240_000,
  });
  const script = loadDemoScript();

  await runDemoScript(
    {
      golden,
      examples,
      seller,
      extensionId,
      releaseId,
      sellPostUrl,
      minDelta,
      priceApiBase,
      graderApiBase,
      apiKey,
    },
    script,
    (title, body) => test.step(title, body),
  );

  await disposeDemoBrowser(context);
});
