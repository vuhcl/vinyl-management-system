import { expect, test } from "@playwright/test";
import * as fs from "fs";
import * as path from "path";
import { launchWithExtension } from "../fixtures/extension";

/**
 * Pitch assist: trimmed ``demo.spec.ts`` for a live-narrated screen recording.
 * Release page first, one seller comment (typed only — you click Grade), one
 * price overlay. No Playwright test annotations.
 *
 * See ``RECORDING_PITCH.md`` and ``grader/demo/golden_predict_demo_pitch.json``.
 *
 * Env (repo-root ``.env`` + exports):
 *   PRICE_API_BASE, GRADER_API_BASE, SELL_POST_URL, DEMO_RELEASE_ID,
 *   GOLDEN_FILE, VINYLIQ_API_KEY
 *   PITCH_HOLD_ON_RELEASE_MS (default 3000)
 *   PITCH_PAUSE_BEFORE_TYPE_MS (default 3000)
 *   PITCH_HOLD_AFTER_ESTIMATE_MS (default 2000)
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
    "golden_predict_demo_pitch.json"
  );
  const file = process.env.GOLDEN_FILE ?? fallback;
  const raw = fs.readFileSync(file, "utf8");
  const parsed = JSON.parse(raw) as GoldenFile;
  if (!parsed.examples || parsed.examples.length < 1) {
    throw new Error(`Golden file ${file} must contain at least 1 example.`);
  }
  return parsed;
}

function envOr(name: string, fallback: string): string {
  const v = process.env[name];
  return v && v.trim().length > 0 ? v : fallback;
}

function envMs(name: string, fallback: number): number {
  const v = process.env[name];
  if (!v || v.trim().length === 0) {
    return fallback;
  }
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
}

test("vinyliq pitch assist", async () => {
  const golden = loadGolden();
  const example = golden.examples[0];
  const releaseId = envOr("DEMO_RELEASE_ID", String(golden.demo_release_id));
  const sellPostUrl = envOr("SELL_POST_URL", golden.sell_post_url);
  const priceApiBase = envOr("PRICE_API_BASE", "http://127.0.0.1:8801");
  const graderApiBase = envOr("GRADER_API_BASE", "http://127.0.0.1:8090");
  const apiKey = process.env.VINYLIQ_API_KEY ?? "";
  const holdOnRelease = envMs("PITCH_HOLD_ON_RELEASE_MS", 3000);
  const pauseBeforeType = envMs("PITCH_PAUSE_BEFORE_TYPE_MS", 3000);
  const holdAfterEstimate = envMs("PITCH_HOLD_AFTER_ESTIMATE_MS", 2000);

  const { context, extensionId } = await launchWithExtension();

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

  const seller = await context.newPage();
  const releaseUrl = `https://www.discogs.com/release/${releaseId}`;

  await seller.goto(releaseUrl);
  await seller.waitForLoadState("domcontentloaded");
  await seller.waitForTimeout(holdOnRelease);

  await seller.goto(sellPostUrl);

  const commentSelector =
    'textarea[name="comments"], textarea[id*="comment" i], textarea[name="release_comments"], textarea[name="description"]';

  const commentEl = seller.locator(commentSelector).first();
  await commentEl.waitFor({ state: "visible", timeout: 30_000 });
  await seller.waitForTimeout(pauseBeforeType);

  await commentEl.fill(example.text);

  // Operator clicks Grade condition while screen recording; resume Inspector when ready.
  await seller.pause();

  await seller.goto(releaseUrl);
  await seller.waitForLoadState("domcontentloaded");

  const sw = context
    .serviceWorkers()
    .find((s) => s.url().startsWith(`chrome-extension://${extensionId}`));
  if (!sw) {
    throw new Error("Extension service worker not found in context");
  }

  const url = `${priceApiBase.replace(/\/$/, "")}/estimate`;
  const body = {
    release_id: releaseId,
    media_condition: example.expected_media_condition,
    sleeve_condition: example.expected_sleeve_condition,
    refresh_stats: false,
  };

  const result = (await sw.evaluate(async (input: SwFetchInput) => {
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

  await sw.evaluate(
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

  await seller.waitForTimeout(holdAfterEstimate);

  expect(data.estimated_price).toBeGreaterThan(0);

  await context.close();
});
