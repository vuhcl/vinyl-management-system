import { test } from "@playwright/test";
import * as fs from "fs";
import * as path from "path";
import * as readline from "node:readline";

import {
  commentFieldLocator,
  focusAndTypeSellerComment,
  vinyliqSellDockSelector,
} from "../fixtures/demo_runner";
import { launchWithExtension } from "../fixtures/extension";

/**
 * Pitch assist: open release → sell listing → type one golden comment, then stop.
 * Grade, estimate, and overlay are manual (screen recorder + extension dock).
 *
 * See ``RECORDING_PITCH.md`` and ``grader/demo/golden_predict_demo_pitch.json``.
 *
 * Env (repo-root ``.env`` + exports):
 *   PRICE_API_BASE, GRADER_API_BASE, SELL_POST_URL, DEMO_RELEASE_ID,
 *   GOLDEN_FILE, VINYLIQ_API_KEY
 *   PITCH_HOLD_ON_RELEASE_MS (default 3000)
 *   PITCH_PAUSE_BEFORE_TYPE_MS (default 3000)
 *
 * Launch disables Playwright recordVideo/showActions (external capture only).
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

function loadGolden(): GoldenFile {
  const fallback = path.resolve(
    __dirname,
    "..",
    "..",
    "..",
    "grader",
    "demo",
    "golden_predict_demo_pitch.json",
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

function ttyQuestion(prompt: string): Promise<void> {
  return new Promise<void>((resolve) => {
    const rl = readline.createInterface({
      input: process.stdin,
      output: process.stderr,
    });
    rl.question(prompt, () => {
      rl.close();
      resolve();
    });
  });
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

  const { context, extensionId } = await launchWithExtension({
    recordVideo: false,
    slowMo: 0,
  });

  try {
    const popup = await context.newPage();
    await popup.goto(`chrome-extension://${extensionId}/popup.html`);
    await popup.evaluate(
      async ({ priceApiBase, graderApiBase, apiKey }) => {
        await new Promise<void>((resolve) =>
          chrome.storage.sync.set(
            { priceApiBase, graderApiBase, apiKey },
            () => resolve(),
          ),
        );
      },
      { priceApiBase, graderApiBase, apiKey },
    );
    await popup.close();

    const seller = await context.newPage();
    const releaseUrl = `https://www.discogs.com/release/${releaseId}`;

    await seller.goto(releaseUrl);
    await seller.waitForLoadState("domcontentloaded");
    await seller.waitForTimeout(holdOnRelease);

    await seller.goto(sellPostUrl, {
      waitUntil: "domcontentloaded",
      timeout: 120_000,
    });

    const sellDockTimeout = Number.parseInt(
      process.env.PLAYWRIGHT_SELL_DOCK_TIMEOUT_MS ?? "",
      10,
    );
    const dockTimeout =
      Number.isFinite(sellDockTimeout) && sellDockTimeout >= 60_000
        ? sellDockTimeout
        : 240_000;

    await seller
      .locator(vinyliqSellDockSelector())
      .waitFor({ state: "visible", timeout: dockTimeout });

    const commentEl = commentFieldLocator(seller);
    await commentEl.waitFor({ state: "visible", timeout: 120_000 });
    await commentEl.scrollIntoViewIfNeeded();
    await seller.waitForTimeout(pauseBeforeType);

    await focusAndTypeSellerComment(commentEl, example.text);

    console.error(
      "\n[pitch-assist] Comment filled. Finish Grade / estimate / overlay yourself, " +
        "then press Enter here to close Chromium.\n",
    );
    if (process.stdin.isTTY) {
      await ttyQuestion("[pitch-assist] Enter to close browser…\n");
    } else {
      console.error(
        "[pitch-assist] Non-TTY: quit Chromium from the Dock when finished.\n",
      );
      await context.waitForEvent("close");
    }
  } finally {
    await context.close().catch(() => undefined);
  }
});
