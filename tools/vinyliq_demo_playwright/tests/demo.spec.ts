import { expect, test } from "@playwright/test";
import * as fs from "fs";
import * as path from "path";
import { launchWithExtension } from "../fixtures/extension";

/**
 * The 2-minute demo: seller-page condition grading (twice, with two
 * comments mapped to two grade pairs) followed by twin price estimates
 * on the matching release page. The golden file
 * ``grader/demo/golden_predict_demo.json`` drives both halves so the
 * recording is deterministic and the price spread can be asserted.
 *
 * Required env (all set via repo-root .env in the recording runbook):
 *   PRICE_API_BASE        e.g. https://35-1-2-3.nip.io/price
 *   GRADER_API_BASE       e.g. https://35-1-2-3.nip.io/grader
 *   SELL_POST_URL         seller listing page URL on the project account
 *   DEMO_RELEASE_ID       456663 by default
 *   MIN_PRICE_DELTA_USD   100 by default
 *   GOLDEN_FILE           path to grader/demo/golden_predict_demo.json
 *                         (default resolves to the repo copy)
 */

interface GoldenExample {
  id: string;
  text: string;
  expected_media_condition: string;
  expected_sleeve_condition: string;
  notes?: string;
}

interface GoldenFile {
  demo_release_id: number | string;
  release_description: string;
  sell_post_url: string;
  min_price_delta_usd: number;
  examples: GoldenExample[];
}

interface SwFetchInput {
  url: string;
  apiKey: string;
  body: unknown;
}

interface SwFetchResult {
  ok: boolean;
  status?: number;
  data?: unknown;
  errorBody?: string;
}

function loadGolden(): GoldenFile {
  const fallback = path.resolve(
    __dirname,
    "..",
    "..",
    "..",
    "grader",
    "demo",
    "golden_predict_demo.json"
  );
  const file = process.env.GOLDEN_FILE ?? fallback;
  const raw = fs.readFileSync(file, "utf8");
  const parsed = JSON.parse(raw) as GoldenFile;
  if (!parsed.examples || parsed.examples.length < 2) {
    throw new Error(
      `Golden file ${file} must contain at least 2 examples (A and B).`
    );
  }
  return parsed;
}

function envOr(name: string, fallback: string): string {
  const v = process.env[name];
  return v && v.trim().length > 0 ? v : fallback;
}

test("vinyliq demo end to end", async () => {
  const golden = loadGolden();
  const [exampleA, exampleB] = golden.examples;
  const releaseId = envOr("DEMO_RELEASE_ID", String(golden.demo_release_id));
  const sellPostUrl = envOr("SELL_POST_URL", golden.sell_post_url);
  const minDelta = Number(
    envOr("MIN_PRICE_DELTA_USD", String(golden.min_price_delta_usd ?? 100))
  );
  const priceApiBase = envOr("PRICE_API_BASE", "http://127.0.0.1:8801");
  const graderApiBase = envOr("GRADER_API_BASE", "http://127.0.0.1:8090");
  const apiKey = process.env.VINYLIQ_API_KEY ?? "";

  const { context, extensionId } = await launchWithExtension();
  test.info().annotations.push({
    type: "extension-id",
    description: extensionId,
  });

  // 1. Seed extension settings via the popup page so the seller content
  //    script and the popup both pick up the right hosts.
  const popup = await context.newPage();
  await popup.goto(`chrome-extension://${extensionId}/popup.html`);
  await popup.evaluate(
    async ({ priceApiBase, graderApiBase, apiKey }) => {
      await new Promise<void>((resolve) =>
        chrome.storage.sync.set(
          { priceApiBase, graderApiBase, apiKey },
          () => resolve()
        )
      );
    },
    { priceApiBase, graderApiBase, apiKey }
  );
  await popup.close();

  // 2. Seller flow — comment A. The injected content script
  //    (seller-grade.js) drives the grader call; the test only
  //    types text, clicks the injected button, and reads the
  //    resulting <select> values back out.
  const seller = await context.newPage();
  await seller.goto(sellPostUrl);

  // Selectors mirror the lists in vinyliq-extension/seller-grade.js so
  // the spec finds the same elements the content script does.
  const commentSelector =
    'textarea[name="comments"], textarea[id*="comment" i], textarea[name="release_comments"], textarea[name="description"]';
  const mediaSelector =
    'select[name="condition"], select#condition, select[name="media_condition"], select[id*="media" i]';
  const sleeveSelector =
    'select[name="sleeve_condition"], select#sleeve_condition, select[id*="sleeve" i]';

  const commentEl = seller.locator(commentSelector).first();
  await commentEl.waitFor({ state: "visible", timeout: 30_000 });

  await commentEl.fill(exampleA.text);
  await seller.locator("#vinyliq-grade-btn").click();
  await expect
    .poll(async () => seller.locator(mediaSelector).first().inputValue(), {
      timeout: 30_000,
    })
    .toBe(exampleA.expected_media_condition);
  expect(await seller.locator(sleeveSelector).first().inputValue()).toBe(
    exampleA.expected_sleeve_condition
  );
  await seller.waitForTimeout(2500); // hold for the viewer

  // 3. Seller flow — comment B (different grade pair).
  await commentEl.fill(exampleB.text);
  await seller.locator("#vinyliq-grade-btn").click();
  await expect
    .poll(async () => seller.locator(mediaSelector).first().inputValue(), {
      timeout: 30_000,
    })
    .toBe(exampleB.expected_media_condition);
  expect(await seller.locator(sleeveSelector).first().inputValue()).toBe(
    exampleB.expected_sleeve_condition
  );
  expect(
    `${exampleA.expected_media_condition}|${exampleA.expected_sleeve_condition}`
  ).not.toBe(
    `${exampleB.expected_media_condition}|${exampleB.expected_sleeve_condition}`
  );
  await seller.waitForTimeout(2500);

  // 4. Release page — twin estimates. Playwright cannot open the MV3
  //    toolbar-action popup programmatically, so we drive the same
  //    production path through the extension's service worker context
  //    (which has chrome.* APIs and the manifest's host_permissions)
  //    and trigger the existing SHOW_OVERLAY handler in content.js so
  //    the recording looks identical to a live demo.
  const releaseUrl = `https://www.discogs.com/release/${releaseId}`;
  await seller.goto(releaseUrl);
  await seller.waitForLoadState("domcontentloaded");

  const sw = context
    .serviceWorkers()
    .find((s) => s.url().startsWith(`chrome-extension://${extensionId}`));
  if (!sw) {
    throw new Error("Extension service worker not found in context");
  }

  async function readEstimate(media: string, sleeve: string): Promise<number> {
    const url = `${priceApiBase.replace(/\/$/, "")}/estimate`;
    const body = {
      release_id: releaseId,
      media_condition: media,
      sleeve_condition: sleeve,
      refresh_stats: false,
    };

    const result = (await sw!.evaluate(async (input: SwFetchInput) => {
      const headers: Record<string, string> = {
        "Content-Type": "application/json",
      };
      if (input.apiKey) {
        headers["X-API-Key"] = input.apiKey;
      }
      const r = await fetch(input.url, {
        method: "POST",
        headers,
        body: JSON.stringify(input.body),
      });
      const text = await r.text();
      if (!r.ok) {
        return { ok: false, status: r.status, errorBody: text };
      }
      try {
        return { ok: true, data: JSON.parse(text) };
      } catch {
        return { ok: false, status: r.status, errorBody: text };
      }
    }, { url, apiKey, body })) as SwFetchResult;

    if (!result.ok) {
      throw new Error(`Estimate failed: ${JSON.stringify(result)}`);
    }
    const data = result.data as { estimated_price?: number } | undefined;
    if (!data || data.estimated_price == null) {
      throw new Error(
        `Estimate response missing estimated_price: ${JSON.stringify(result.data)}`
      );
    }

    // Render the same overlay the popup would show. content.js listens
    // for SHOW_OVERLAY on chrome.runtime.onMessage; chrome.tabs.sendMessage
    // from the SW reaches it.
    await sw!.evaluate(
      async (input: { releaseUrl: string; payload: unknown }) => {
        const tabs = await chrome.tabs.query({ url: input.releaseUrl });
        const target = tabs[0];
        if (target?.id != null) {
          await chrome.tabs.sendMessage(target.id, {
            type: "SHOW_OVERLAY",
            payload: input.payload,
          });
        }
      },
      { releaseUrl, payload: data }
    );

    return Number(data.estimated_price);
  }

  const p1 = await readEstimate(
    exampleA.expected_media_condition,
    exampleA.expected_sleeve_condition
  );
  await seller.waitForTimeout(2000);
  const p2 = await readEstimate(
    exampleB.expected_media_condition,
    exampleB.expected_sleeve_condition
  );
  await seller.waitForTimeout(4000);

  expect(Math.abs(p1 - p2)).toBeGreaterThanOrEqual(minDelta);

  await context.close();
});
