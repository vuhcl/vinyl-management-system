import type { Locator, Page } from "@playwright/test";

import type { GoldenPredictDemoJson } from "./golden";

import {
  demoVideoChaptersDisabledFromEnv,
  releasePageAfterHmmScrollLeadMs,
  showDemoCatalogReleaseConfirmedChapter,
  showDemoCatalogReleaseUncertaintyChapter,
  showDemoCatalogSearchIntroChapter,
} from "./demo_video_ann";

/** Legacy flag: forcing **on** behaves like **`DEMO_CATALOG_UX=1`**. */
export function demoFullWalkthroughEnabled(): boolean {
  const v = (process.env.DEMO_FULL_WALKTHROUGH ?? "").trim().toLowerCase();
  return ["1", "true", "yes"].includes(v);
}

/**
 * Prefer **discogs.com** → navbar search → **Masters** facet → **`/master/{id}`**
 * → **`/release/{id}`** → **Sell** → **`/sell/post/{id}`**
 * (`golden.search_query`, `golden.demo_master_id`, `golden.demo_release_id`).
 *
 * Default **on** locally; **`npm run test:ci`** sets **`DEMO_CATALOG_UX=0`** for a fast **`SELL_POST_URL`** deep-link.
 *
 * Opt out anytime: **`DEMO_CATALOG_UX=0`** or **`DEMO_DEEP_LINK_SELL_POST=1`**.
 */
export function demoCatalogUxEnabled(): boolean {
  if (demoFullWalkthroughEnabled()) {
    return true;
  }
  const deep = (process.env.DEMO_DEEP_LINK_SELL_POST ?? "").trim().toLowerCase();
  if (["1", "true", "yes"].includes(deep)) {
    return false;
  }
  const raw = (process.env.DEMO_CATALOG_UX ?? "1").trim().toLowerCase();
  return !["0", "false", "no", "off"].includes(raw);
}

async function dismissDiscogsConsentIfPresent(page: Page): Promise<void> {
  await page
    .locator("#onetrust-accept-btn-handler")
    .click({ timeout: 5000 })
    .catch(() => undefined);
}

/**
 * Catalog intro chapter belongs on Discogs **marketing home** (`/` or locale hub `/en`, …).
 * Release/Master URLs must not reuse this waiter before the scripted search.
 */
async function waitForDiscogsMarketingHubPath(page: Page): Promise<void> {
  await page.waitForFunction(
    () => {
      const host = window.location.hostname.replace(/^www\./i, "");
      if (!/(^|\.)discogs\.com$/i.test(host)) {
        return false;
      }
      const raw =
        window.location.pathname.replace(/\/+$/u, "") || "/";
      if (raw === "/" || raw === "") return true;
      return /^\/[a-z]{2}$/i.test(raw);
    },
    undefined,
    { timeout: 120_000 },
  );
}

/**
 * Master / release picks sometimes open in a **new tab** (``target="_blank"``). The
 * scripted demo must remain on the **recorder tab** so URL waits and narration run
 * on the page Playwright owns — resolve ``href`` and ``goto`` when possible.
 */
async function followDiscogsAnchorSameTab(
  page: Page,
  anchor: Locator,
  clickTimeoutMs: number,
): Promise<void> {
  const raw =
    (await anchor.getAttribute("href"))?.trim() ??
    (await anchor.evaluate(
      (el) =>
        el instanceof HTMLAnchorElement
          ? el.getAttribute("href")?.trim() ?? ""
          : "",
    ));
  const lower = raw.toLowerCase();
  const usable =
    raw.length > 0 &&
    raw !== "#" &&
    !lower.startsWith("javascript:");
  if (usable) {
    const dest = new URL(raw, page.url()).href;
    await page.goto(dest, {
      waitUntil: "domcontentloaded",
      timeout: 240_000,
    });
    return;
  }
  await anchor.click({ timeout: clickTimeoutMs });
}

async function locateNavbarSearch(page: Page) {
  const candidates = [
    page.locator("#discogs-navbar-search-query"),
    page.locator('[data-testid="navbar-search"]').locator('input:not([type="hidden"])').first(),
    page.getByRole("searchbox"),
    page.locator('header input[type="search"]'),
    page.locator('[class*="search" i] input[type="text"]'),
    page.locator('input[placeholder*="Search" i]:visible'),
    page.locator('input[placeholder*="search" i]:visible'),
    page.locator('input[name="q"]').first(),
  ];

  let lastErr: Error | undefined;
  for (const loc of candidates) {
    try {
      const el = loc.first();
      await el.waitFor({ state: "visible", timeout: 12_000 });
      return el;
    } catch (e) {
      lastErr =
        e instanceof Error ? e : new Error(typeof e === "string" ? e : "search probe");
    }
  }
  throw new Error(
    `Discogs: navbar search control not found. ${lastErr?.message ?? "See discogs_navigation.ts selectors."}`,
  );
}

function catalogMasterId(golden: GoldenPredictDemoJson): string {
  const fromEnv = (process.env.DEMO_MASTER_ID ?? "").trim();
  if (fromEnv.length > 0) {
    return fromEnv;
  }
  const fromGolden =
    golden.demo_master_id !== undefined && golden.demo_master_id !== null
      ? String(golden.demo_master_id).trim()
      : "";
  return fromGolden;
}

/**
 * Narrow the active search URL to **Masters** (same **`q`** as the navbar search).
 */
async function gotoMasterSearchKeepingQuery(page: Page): Promise<void> {
  await page.waitForURL(/\/(?:www\.)?discogs\.com\/search\b/i, {
    timeout: 180_000,
  });
  let url: URL;
  try {
    url = new URL(page.url());
  } catch {
    throw new Error(`Discogs: could not parse search URL: ${page.url()}`);
  }
  url.searchParams.set("type", "master");
  await page.goto(url.toString(), {
    waitUntil: "domcontentloaded",
    timeout: 240_000,
  });
}



function masterResultLinkLocator(page: Page, mid: string) {
  const abs = `https://www.discogs.com/master/${mid}`;
  const absDisc = `https://discogs.com/master/${mid}`;
  return page
    .locator(
      `a[href^="/master/${mid}-"], ` +
        `a[href="/master/${mid}"], ` +
        `a[href^="/master/${mid}?"], ` +
        `a[href*="/master/${mid}-"], ` +
        `a[href^="${abs}-"], ` +
        `a[href="${abs}"], ` +
        `a[href^="${abs}?"], ` +
        `a[href^="${absDisc}-"], ` +
        `a[href="${absDisc}"], ` +
        `a[href^="${absDisc}?"]`,
    )
    .first();
}

function releasePickLinkLocator(page: Page, rid: string) {
  const abs = `https://www.discogs.com/release/${rid}`;
  const absDisc = `https://discogs.com/release/${rid}`;
  return page
    .locator(
      `a[href^="/release/${rid}-"], ` +
        `a[href="/release/${rid}"], ` +
        `a[href^="/release/${rid}?"], ` +
        `a[href*="/release/${rid}-"], ` +
        `a[href^="${abs}-"], ` +
        `a[href="${abs}"], ` +
        `a[href^="${abs}?"], ` +
        `a[href^="${absDisc}-"], ` +
        `a[href="${absDisc}"], ` +
        `a[href^="${absDisc}?"]`,
    )
    .first();
}

function sellPostLinkLocator(page: Page, rid: string) {
  const abs = `https://www.discogs.com/sell/post/${rid}`;
  const absDisc = `https://discogs.com/sell/post/${rid}`;
  return page
    .locator(
      `a[href^="/sell/post/${rid}"], ` +
        `a[href^="/sell/post/${rid}?"], ` +
        `a[href*="/sell/post/${rid}?"], ` +
        `a[href$="/sell/post/${rid}"], ` +
        `a[href*="/sell/post/${rid}"], ` +
        `a[href^="${abs}"], ` +
        `a[href^="${abs}?"], ` +
        `a[href^="${absDisc}"], ` +
        `a[href^="${absDisc}?"]`,
    )
    .first();
}

/**
 * Discogs may prefix paths with a locale (e.g. ``/en/release/123``). Wait predicates
 * must not assume ``pathname`` starts with ``master`` / ``release`` / ``sell``.
 */
function waitForMasterPath(page: Page, masterId: string, timeout: number) {
  return page.waitForFunction(
    (mid: string) => {
      const seg = window.location.pathname.split("/").filter(Boolean);
      const i = seg.indexOf("master");
      const slug =
        i >= 0 && seg[i + 1] !== undefined
          ? seg[i + 1]
          : seg[0] === "master"
            ? seg[1] ?? ""
            : "";
      if (!slug) {
        return false;
      }
      return slug === String(mid) || slug.startsWith(`${mid}-`);
    },
    masterId,
    { timeout },
  );
}

function waitForReleasePath(page: Page, releaseId: string, timeout: number) {
  return page.waitForFunction(
    (rid: string) => {
      const seg = window.location.pathname.split("/").filter(Boolean);
      const i = seg.indexOf("release");
      const slug =
        i >= 0 && seg[i + 1] !== undefined
          ? seg[i + 1]
          : seg[0] === "release"
            ? seg[1] ?? ""
            : "";
      if (!slug) {
        return false;
      }
      const id = String(rid);
      return slug === id || slug.startsWith(`${id}-`);
    },
    releaseId,
    { timeout },
  );
}

function waitForSellPostPath(page: Page, postId: string, timeout: number) {
  return page.waitForFunction(
    (pid: string) => {
      const seg = window.location.pathname.split("/").filter(Boolean);
      for (let i = 0; i < seg.length - 2; i++) {
        if (seg[i] === "sell" && seg[i + 1] === "post" && seg[i + 2]) {
          const slug = seg[i + 2];
          const sid = String(pid);
          return slug === sid || slug.startsWith(`${sid}-`);
        }
      }
      return false;
    },
    postId,
    { timeout },
  );
}

/**
 * Poll until the recorder tab pathname is a **release** view for ``releaseId``
 * (handles ``/en/release/…``, etc.). Shared by automated catalog UX and **hybrid** runs.
 */
export async function waitUntilDiscogsGoldenReleasePage(
  page: Page,
  releaseId: string | number,
  timeoutMs: number,
): Promise<void> {
  await waitForReleasePath(page, String(releaseId).trim(), timeoutMs);
}

/**
 * ``DEMO_HYBRID=1`` + catalog: after the search intro, the operator drives navigation;
 * we watch the same tab and replay the release-page beats (``Hmm…`` → scroll → confirmed)
 * matching ``gotoSellerViaDiscogsUx``. Skipped when ``DEMO_VIDEO_CHAPTERS=0``.
 */
export async function runHybridCatalogReleasePageNarration(
  page: Page,
  releaseId: string | number,
  navTimeoutMs: number,
): Promise<void> {
  if (demoVideoChaptersDisabledFromEnv()) {
    return;
  }
  await waitUntilDiscogsGoldenReleasePage(page, releaseId, navTimeoutMs);
  await page.waitForTimeout(160);
  await showDemoCatalogReleaseUncertaintyChapter(page);
  await page.evaluate(() => {
    window.scrollBy(
      0,
      Math.min(
        520,
        Math.max(220, document.body.scrollHeight * 0.28),
      ),
    );
  });
  await page.waitForTimeout(releasePageAfterHmmScrollLeadMs());
  await showDemoCatalogReleaseConfirmedChapter(page);
}

/**
 * Scripted operator path:
 *
 * **`https://www.discogs.com/`** → **`search_query`** Enter → **`type=master`** (same **`q`**)
 * → **`/master/{demo_master_id}`** → matching **`/release/{demo_release_id}`** → **Sell**
 * → **`/sell/post/{demo_release_id}`**.
 */
export async function gotoSellerViaDiscogsUx(
  page: Page,
  golden: GoldenPredictDemoJson,
): Promise<void> {
  const rid = String(golden.demo_release_id).trim();
  const query = String(golden.search_query ?? "").trim();
  const mid = catalogMasterId(golden);

  if (!query) {
    throw new Error(
      "golden_predict_demo.json is missing search_query (required when catalog UX is enabled).",
    );
  }
  if (!mid) {
    throw new Error(
      "Catalog UX requires demo_master_id on the golden (Discogs Master id). Set DEMO_MASTER_ID to override.",
    );
  }

  await page.goto("https://www.discogs.com/", {
    waitUntil: "domcontentloaded",
    timeout: 240_000,
  });
  await dismissDiscogsConsentIfPresent(page);
  await waitForDiscogsMarketingHubPath(page);

  await showDemoCatalogSearchIntroChapter(page);

  const searchInput = await locateNavbarSearch(page);
  await searchInput.click({ timeout: 15_000 });
  await searchInput.fill("");
  await searchInput.fill(query);

  await searchInput.press("Enter");

  await gotoMasterSearchKeepingQuery(page);
  await dismissDiscogsConsentIfPresent(page);

  const masterAnchor = masterResultLinkLocator(page, mid);
  await masterAnchor.scrollIntoViewIfNeeded();
  await followDiscogsAnchorSameTab(page, masterAnchor, 120_000);

  await waitForMasterPath(page, mid, 120_000);

  const releaseAnchor = releasePickLinkLocator(page, rid);
  await releaseAnchor.scrollIntoViewIfNeeded();
  await followDiscogsAnchorSameTab(page, releaseAnchor, 120_000);

  await waitForReleasePath(page, rid, 120_000);
  await page.waitForTimeout(160);
  await showDemoCatalogReleaseUncertaintyChapter(page);
  await page.evaluate(() => {
    window.scrollBy(
      0,
      Math.min(
        520,
        Math.max(220, document.body.scrollHeight * 0.28),
      ),
    );
  });
  await page.waitForTimeout(releasePageAfterHmmScrollLeadMs());
  await showDemoCatalogReleaseConfirmedChapter(page);

  const sellAnchor = sellPostLinkLocator(page, rid);
  await sellAnchor.waitFor({ state: "visible", timeout: 120_000 });
  await sellAnchor.scrollIntoViewIfNeeded();
  await followDiscogsAnchorSameTab(page, sellAnchor, 60_000);

  await waitForSellPostPath(page, rid, 180_000);
}
