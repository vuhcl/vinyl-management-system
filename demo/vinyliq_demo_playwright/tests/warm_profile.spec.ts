import * as readline from "node:readline";

import { test } from "@playwright/test";

import { launchWarmupPersistentContext } from "../fixtures/extension";
import { readGoldenPredictDemo } from "../fixtures/golden";

function resolveSellPostUrlForWarm(): string | undefined {
  const envSell = process.env.SELL_POST_URL?.trim();
  if (envSell) {
    return envSell;
  }
  try {
    const golden = readGoldenPredictDemo({});
    const u = golden.sell_post_url?.trim();
    if (!u || u.includes("REPLACE")) {
      return undefined;
    }
    return u;
  } catch {
    return undefined;
  }
}

function truthyEnv(name: string): boolean {
  const v = (process.env[name] ?? "").trim().toLowerCase();
  return ["1", "true", "yes"].includes(v);
}

/**
 * Visiting `/sell/post/…` during warm usually forces a **second** Cloudflare
 * turnstile distinct from `/login`; default warmup stays on login-only and
 * leaves seller checks to **`npm test`** unless explicitly opted in.
 */
function warmProfileIncludeSellPost(): boolean {
  return truthyEnv("WARM_PROFILE_INCLUDE_SELL_POST");
}

function navSettleMs(): number {
  const raw = Number.parseInt(
    process.env.PLAYWRIGHT_WARM_NAV_SETTLE_MS ?? "",
    10,
  );
  const parsed = Number.isFinite(raw) && raw >= 0 ? raw : 3500;
  return parsed;
}

function uniqOrdered(urls: string[]): string[] {
  const out: string[] = [];
  const seen = new Set<string>();
  for (const u of urls) {
    if (!seen.has(u)) {
      seen.add(u);
      out.push(u);
    }
  }
  return out;
}

/** Discogs URLs to stabilize Cloudflare/session for the demo browser profile. */
function resolveWarmUrls(): string[] {
  const raw = process.env.WARM_PROFILE_URLS?.trim();
  if (raw) {
    return uniqOrdered(raw.split(/[,|]/).map((s) => s.trim()).filter(Boolean));
  }
  const urls: string[] = [];
  const override = process.env.WARM_PROFILE_START_URL?.trim();
  urls.push(
    override && override.length > 0
      ? override
      : "https://www.discogs.com/login",
  );
  const resolvedSell = resolveSellPostUrlForWarm();
  const includeSell = warmProfileIncludeSellPost();
  const sellHop = includeSell ? resolvedSell : undefined;
  if (!includeSell && resolvedSell) {
    console.error(
      "[warm-profile] Skipping `/sell/post` during warm (avoids extra Cloudflare). " +
        "Set WARM_PROFILE_INCLUDE_SELL_POST=1 to seed that route too.",
    );
  }
  if (sellHop) {
    urls.push(sellHop);
  }
  return uniqOrdered(urls);
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

async function idleMs(ms: number): Promise<void> {
  await new Promise<void>((r) => {
    setTimeout(r, ms);
  });
}

async function gracefulClose(context: {
  close: () => Promise<void>;
}): Promise<void> {
  await idleMs(2500);
  await context.close().catch(() => undefined);
}

/**
 * Seeds ``CHROME_PROFILE_DIR`` using bundled Chromium **without** the VinylIQ
 * extension loaded (same automation-signal trims as demo). Fewer scripted
 * network edges against Discogs / Cloudflare than ``launchWithExtension``.
 * Skipped unless ``WARM_PROFILE=1``.
 *
 * ```bash
 * export CHROME_PROFILE_DIR=/path/to/profile
 * npm run warm-profile                         # login only (fewest CF hoops)
 *
 * export WARM_PROFILE_INCLUDE_SELL_POST=1       # optional second hop (/sell/post)
 * npm run warm-profile
 * ```
 */
test("warm persistent profile for Discogs (manual Cloudflare/login)", async () => {
  test.skip(
    process.env.WARM_PROFILE !== "1",
    'Set WARM_PROFILE=1 — use `npm run warm-profile` instead of guessing.',
  );

  test.skip(
    !!process.env.PLAYWRIGHT_CONNECT_CDP?.trim(),
    "PLAYWRIGHT_CONNECT_CDP set — bypass warm-profile and clear Discogs manually in Chrome.",
  );

  test.setTimeout(45 * 60_000);

  const urls = resolveWarmUrls();
  if (urls.length === 0) {
    throw new Error(
      "No URLs resolved for warm-profile (check WARM_PROFILE_URLS)",
    );
  }

  const context = await launchWarmupPersistentContext();
  const initial =
    context.pages().find((p) => !p.url().startsWith("chrome-extension://")) ??
    context.pages()[0] ??
    (await context.newPage());

  try {
    for (let i = 0; i < urls.length; i++) {
      const url = urls[i];
      console.error(
        `\n[warm-profile] (${i + 1}/${urls.length}) Loading ${url}\n`,
      );

      await initial.goto(url, {
        waitUntil: "domcontentloaded",
        timeout: 240_000,
      });
      // Cloudflare clears often finish after DCL — give `_cf_bm`/`cf_clearance`
      // a moment before we prompt / advance to the next scripted navigation.
      await initial.waitForLoadState("load").catch(() => undefined);
      const settle = navSettleMs();
      if (settle > 0) {
        await idleMs(settle);
      }

      if (process.stdin.isTTY && urls.length > 1 && i < urls.length - 1) {
        await ttyQuestion(
          "[warm-profile] Cloudflare/login OK for THIS page? Press Enter for the next URL.\n",
        );
      }
    }

    console.error(
      "[warm-profile] Final step: confirm you see a usable Discogs page (logged in if required).",
    );
    if (process.stdin.isTTY) {
      await ttyQuestion(
        "[warm-profile] Press Enter to save cookies and exit Chromium.\n",
      );
      await gracefulClose(context);
    } else {
      console.error(
        "[warm-profile] Non-TTY session: quit Chromium from the Dock (⌘Q) when finished.",
      );
      await context.waitForEvent("close");
      await idleMs(2500);
    }
  } catch (cause) {
    await gracefulClose(context);
    throw cause;
  }
});
