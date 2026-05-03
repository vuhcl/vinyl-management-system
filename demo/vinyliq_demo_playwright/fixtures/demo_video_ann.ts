import { expect } from "@playwright/test";
import type { Frame, Page } from "@playwright/test";

import {
  readListingPriceFingerprintExtensionOrder,
  visibleMediaConditionSelect,
} from "./seller_listing_locators";

/**
 * **``demoVideoChaptersDisabledFromEnv()``** — narration **off** only when **`DEMO_VIDEO_CHAPTERS`** is **`0`/`false`/`no`/`off`**
 * (unset ⇒ **`1`**, **`npm run test:ci`** forces **`0`**). Ignore this when diagnosing **partial** narration (same setting for every step).
 *
 * **Injection frame:** prefer the **visible Media condition** **`select`** (**same node** as grade polls in **`demo_runner`**) →
 * **`elementHandle.ownerFrame()`**; then **`#vinyliq-sell-dock`**; else main. The dock can sit in a different embedding than the form.
 *
 * In **hybrid** runs, post-grade overlays require **`expect.poll`** on golden ladder dropdowns — stalled grading never reaches the narrator.
 */

/** SYNC: **`vinyliq-extension`** sell dock **`id`** (see **`demo_runner.vinyliqSellDockSelector`**). */
const VINYLIQ_SELL_DOCK_SELECTOR = "#vinyliq-sell-dock";

/**
 * Locate the **same** **`Frame`** Discogs+VinylIQ use for **`vinyliq-sell-dock`**, using **`elementHandle.ownerFrame()`**
 * ( **`Frame.evaluate(fn, payload)`** does **not** inject an extra **`Element`** arg — unlike **`locator.evaluate`** ).
 */
async function sellDockOwnerFrame(page: Page, timeoutMs: number): Promise<Frame | null> {
  const dockLoc = page.locator(VINYLIQ_SELL_DOCK_SELECTOR).first();
  try {
    await dockLoc.waitFor({ state: "attached", timeout: timeoutMs });
  } catch {
    return null;
  }
  const handle = await dockLoc.elementHandle({ timeout: 10_000 });
  if (!handle) {
    return null;
  }
  const owner = await handle.ownerFrame();
  await handle.dispose();
  return owner;
}

/** Frame that owns the live **Media condition** row (what **`expect.poll`** reads after **Grade**). */
async function sellFormOwnerFrameFromMediaSelect(
  page: Page,
  timeoutMs: number,
): Promise<Frame | null> {
  const ml = visibleMediaConditionSelect(page);
  try {
    await ml.waitFor({ state: "attached", timeout: timeoutMs });
  } catch {
    return null;
  }
  const handle = await ml.elementHandle({ timeout: 10_000 });
  if (!handle) {
    return null;
  }
  const owner = await handle.ownerFrame();
  await handle.dispose();
  return owner;
}

async function resolveSellerNarrationInjectionFrame(
  page: Page,
  opts: { mediaAttachTimeoutMs: number; dockFallbackTimeoutMs: number },
): Promise<Frame> {
  const fromForm = await sellFormOwnerFrameFromMediaSelect(
    page,
    opts.mediaAttachTimeoutMs,
  );
  if (fromForm) {
    return fromForm;
  }
  const fromDock = await sellDockOwnerFrame(page, opts.dockFallbackTimeoutMs);
  if (fromDock) {
    return fromDock;
  }
  return page.mainFrame();
}

/** Drop demo narrator DOM in every reachable frame. Mixing ``page.evaluate`` (top) vs dock ``evaluate`` (iframe) leaves orphaned segues/stacking-context winners — the next strip renders “under” the old copy. */
async function purgeDemoFloatingOverlaysInAllFrames(page: Page): Promise<void> {
  const purge = (): void => {
    document.getElementById("__vinyliq_demo_chapter_ann")?.remove();
    document.getElementById("__vinyliq_demo_seller_strip")?.remove();
    document
      .querySelectorAll('[id^="__vinyliq_demo_segue_"]')
      .forEach((el) => el.remove());
    document
      .querySelectorAll('[id^="__vinyliq_demo_aux_"]')
      .forEach((el) => el.remove());
  };
  for (const frame of page.frames()) {
    try {
      await frame.evaluate(purge);
    } catch {
      /* detached frame or cross-origin */
    }
  }
}

type ChapterInjectPayloadBrowser = {
  headline: string;
  subtitle: string;
  durationMs: number;
};

/** Serialized into the sell/catalog document via **`Frame.evaluate`**. */
function demoChapterInjectorInBrowser(p: ChapterInjectPayloadBrowser): void {
  const h = p.headline;
  const s = p.subtitle;
  const ms = p.durationMs;
  const clearPrior = (): void => {
    document.getElementById("__vinyliq_demo_chapter_ann")?.remove();
    document.getElementById("__vinyliq_demo_seller_strip")?.remove();
    document
      .querySelectorAll('[id^="__vinyliq_demo_segue_"]')
      .forEach((el) => el.remove());
    document
      .querySelectorAll('[id^="__vinyliq_demo_aux_"]')
      .forEach((el) => el.remove());
  };
  clearPrior();
  const nid = "__vinyliq_demo_chapter_ann";
  const root = document.createElement("div");
  root.id = nid;
  Object.assign(root.style, {
    position: "fixed",
    inset: "0",
    zIndex: "2147483647",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    flexDirection: "column",
    background: "rgba(8,10,14,0.58)",
    backdropFilter: "blur(6px)",
    WebkitBackdropFilter: "blur(6px)",
    fontFamily:
      'system-ui,-apple-system,"Segoe UI",Roboto,sans-serif',
    color: "#f6f8fc",
    pointerEvents: "none",
    padding: "24px",
    boxSizing: "border-box",
  });
  const card = document.createElement("div");
  Object.assign(card.style, {
    maxWidth: "min(920px, 92vw)",
    textAlign: "center",
  });
  const t = document.createElement("div");
  t.textContent = h;
  Object.assign(t.style, {
    fontSize: "28px",
    fontWeight: "700",
    lineHeight: "1.25",
    marginBottom: s ? "12px" : "0",
    textShadow: "0 2px 12px rgba(0,0,0,0.45)",
  });
  card.appendChild(t);
  if (s) {
    const d = document.createElement("div");
    d.textContent = s;
    Object.assign(d.style, {
      fontSize: "16px",
      lineHeight: "1.55",
      opacity: "0.95",
      fontWeight: "400",
      textShadow: "0 2px 8px rgba(0,0,0,0.5)",
    });
    card.appendChild(d);
  }
  root.appendChild(card);
  (document.documentElement ?? document.body).appendChild(root);
  window.setTimeout(() => root.remove(), ms);
}

type SellerStripInjectPayloadBrowser = {
  headline: string;
  body: string;
  stripId: string;
  ttl: number;
  dockGapPx: number;
};

/** Serialized into the sell-listing **`Frame`** via **`frame.evaluate`** (single **`payload`** arg only). */
function sellerInsightStripInjectorInBrowser(p: SellerStripInjectPayloadBrowser): void {
  const hl = p.headline;
  const bd = p.body;
  const stripId = p.stripId;
  const ttl = p.ttl;
  const dockGapPx = p.dockGapPx;
  const clearPrior = (): void => {
    document.getElementById("__vinyliq_demo_chapter_ann")?.remove();
    document.getElementById("__vinyliq_demo_seller_strip")?.remove();
    document
      .querySelectorAll('[id^="__vinyliq_demo_segue_"]')
      .forEach((el) => el.remove());
    document
      .querySelectorAll('[id^="__vinyliq_demo_aux_"]')
      .forEach((el) => el.remove());
  };
  clearPrior();
  const root = document.createElement("div");
  root.id = stripId;
  const dock = document.getElementById("vinyliq-sell-dock");
  let bottomPx = 88;
  if (dock) {
    const rect = dock.getBoundingClientRect();
    const dockBottomCss = Number.parseFloat(
      getComputedStyle(dock).bottom || "20",
    );
    const edge =
      Number.isFinite(dockBottomCss) && dockBottomCss >= 0
        ? dockBottomCss
        : 20;
    bottomPx = Math.ceil(rect.height + edge + dockGapPx);
  }
  Object.assign(root.style, {
    position: "fixed",
    left: "0",
    right: "0",
    bottom: `${bottomPx}px`,
    zIndex: "2147483647",
    pointerEvents: "none",
    boxSizing: "border-box",
    padding: "20px clamp(14px, 3vw, 28px)",
    paddingBottom: "max(20px, env(safe-area-inset-bottom))",
    background:
      "linear-gradient(to top, rgba(6,8,14,0.92) 0%, rgba(6,8,14,0.78) 70%, transparent 100%)",
    fontFamily:
      'system-ui,-apple-system,"Segoe UI",Roboto,sans-serif',
    color: "#f8fafc",
    textShadow: "0 1px 12px rgba(0,0,0,0.55)",
  });
  const h = document.createElement("div");
  h.textContent = hl;
  Object.assign(h.style, {
    fontWeight: "700",
    fontSize: "clamp(17px, 2.25vw, 22px)",
    lineHeight: "1.35",
    marginBottom: bd ? "8px" : "0",
    maxWidth: "960px",
    marginInline: "auto",
  });
  root.appendChild(h);
  if (bd) {
    const pr = document.createElement("div");
    pr.textContent = bd;
    Object.assign(pr.style, {
      fontSize: "clamp(14px, 1.95vw, 17px)",
      lineHeight: "1.52",
      opacity: "0.96",
      maxWidth: "960px",
      marginInline: "auto",
    });
    root.appendChild(pr);
  }
  const mountEl = document.body ?? document.documentElement;
  mountEl.appendChild(root);
  const observer = new MutationObserver(() => {
    if (!root.isConnected && mountEl.isConnected) {
      try {
        mountEl.appendChild(root);
      } catch {
        /* document tearing down */
      }
    }
  });
  observer.observe(mountEl, { childList: true, subtree: false });
  window.setTimeout(() => {
    observer.disconnect();
    root.remove();
  }, ttl);
}

/** Action highlights baked into Chromium recordVideo frames (persistent launch path). */
export const DEMO_RECORD_VIDEO = {
  dir: "recordings/",
  size: { width: 1280, height: 800 },
  showActions: {
    /** Long enough to read Playwright action titles (Fill, Wait, etc.). */
    duration: 1100,
    position: "bottom-right" as const,
    fontSize: 17,
  },
} as const;

/** Copy A seller estimate headline strip — mounted lifetime (distinct from generic ``DEMO_SELLER_STRIP_MS``). */
const COPY_A_ESTIMATE_STRIP_SCREEN_MS = 12_000;

/** Trim applied to Copy A seller estimate **readLead** vs ``DEMO_SELLER_STRIP_READ_MS``. */
const COPY_A_ESTIMATE_STRIP_TRIM_MS = 2000;

function copyAEstimateStripReadLeadMs(): number {
  return Math.max(400, sellerInsightStripLeadMs() - COPY_A_ESTIMATE_STRIP_TRIM_MS);
}

function coerceScriptExampleIndex(raw: unknown): number | undefined {
  if (typeof raw === "number" && Number.isFinite(raw)) {
    return Number.isInteger(raw) ? raw : Math.trunc(raw);
  }
  if (typeof raw === "string") {
    const v = Number.parseInt(raw.trim(), 10);
    return Number.isFinite(v) ? v : undefined;
  }
  return undefined;
}

/** Normalizes JSON/script ``example_index`` (handles stringly-typed scripts). */
export function demoScriptExampleIndex(raw: unknown): number {
  const ix = coerceScriptExampleIndex(raw);
  if (ix === undefined) {
    throw new Error(`Invalid demo example_index: ${JSON.stringify(raw)}`);
  }
  return ix;
}

/**
 * Full-frame chapter overlays (see ``showDemoChapterCard``).
 * Disabled automatically when ``demo_video_chapters_disabled_from_env()`` is true.
 */
export function demoChapterOverlayMs(): number {
  const raw = Number.parseInt(process.env.DEMO_VIDEO_CHAPTER_MS ?? "", 10);
  return Number.isFinite(raw) && raw >= 800 && raw <= 44_000 ? raw : 12_400;
}

/** Headline-only “Hmm…” beat on `/release/` — avoids using full ``DEMO_VIDEO_CHAPTER_MS`` for a short card. */
function releaseUncertaintyChapterOverlayMs(): number {
  const raw = Number.parseInt(process.env.DEMO_RELEASE_HMM_CHAPTER_MS ?? "", 10);
  return Number.isFinite(raw) && raw >= 700 && raw <= 20_000 ? raw : 2000;
}

/**
 * Bare release page dwell after dismiss of “Hmm…” and programmatic nudge-scroll,
 * before the confirmation chapter (**`fixtures/discogs_navigation.ts`**).
 */
export function releasePageAfterHmmScrollLeadMs(): number {
  const raw = Number.parseInt(process.env.DEMO_RELEASE_AFTER_HMM_SCROLL_MS ?? "", 10);
  return Number.isFinite(raw) && raw >= 600 && raw <= 45_000 ? raw : 9800;
}

/** Fullscreen intro before navbar catalog search (**`showDemoCatalogSearchIntroChapter`**). */
function catalogSearchIntroChapterOverlayMs(): number {
  const raw = Number.parseInt(process.env.DEMO_CATALOG_SEARCH_INTRO_CHAPTER_MS ?? "", 10);
  return Number.isFinite(raw) && raw >= 700 && raw <= 20_000 ? raw : 4000;
}

/** “The details match…” release confirmation before Sell. */
function catalogReleaseConfirmedChapterOverlayMs(): number {
  const raw = Number.parseInt(process.env.DEMO_RELEASE_CONFIRMED_CHAPTER_MS ?? "", 10);
  return Number.isFinite(raw) && raw >= 700 && raw <= 20_000 ? raw : 5000;
}

export function demoVideoChaptersDisabledFromEnv(): boolean {
  const raw = (process.env.DEMO_VIDEO_CHAPTERS ?? "1").trim().toLowerCase();
  return ["0", "false", "no", "off"].includes(raw);
}

/** Inject step only uses ``SETUP_KINDS``; catalog / deep-link goto chapters call exported helpers below. */
const SETUP_KINDS = new Set(["inject_extension_storage"]);

/** After seller insight strip mounts, dwell before typing resumes (recording legibility). */
export function sellerInsightStripLeadMs(): number {
  const raw = Number.parseInt(process.env.DEMO_SELLER_STRIP_READ_MS ?? "", 10);
  return Number.isFinite(raw) && raw >= 400 && raw <= 28_000 ? raw : 7200;
}

function sellerInsightStripScreenMs(): number {
  const raw = Number.parseInt(process.env.DEMO_SELLER_STRIP_MS ?? "", 10);
  return Number.isFinite(raw) && raw >= 2800 && raw <= 48_000 ? raw : 26_000;
}

export type SellerInsightStripOpts = {
  /** How long the strip stays visible (mounted). */
  screenMs?: number;
  /** After mount, dwell before caller continues (typing / next action). */
  readLeadMs?: number;
  /** DOM id (multiple strips must not collide). */
  stripElementId?: string;
};

/** Skip **post–Copy A estimate** segue and Copy B **outro** (**``DEMO_SKIP_SEGUES``**). Grade segues (**``after_first_grade``**, **``after_second_grade``**) still run when **`DEMO_VIDEO_CHAPTERS=1`**. */
export function demoSkipNarrativeSegues(): boolean {
  const v = (process.env.DEMO_SKIP_SEGUES ?? "").trim().toLowerCase();
  return ["1", "true", "yes"].includes(v);
}

/**
 * Bare dwell (chapters on): after Copy A estimate seller strip, before **`after_first_estimate`**; and
 * after Copy B estimate seller strip, before session **outro** — same **`DEMO_AFTER_FIRST_ESTIMATE_BARE_MS`** knob.
 * Default **`9800`** approximates **(Copy A **12 s** strip TTL − trimmed readLead) + 3 s** at stock tuning;
 * lengthen (e.g. **`19000`**) when you want more silence before the Copy **B** outro.
 */
export function afterFirstEstimateDwellBareMs(): number {
  const raw = Number.parseInt(process.env.DEMO_AFTER_FIRST_ESTIMATE_BARE_MS ?? "", 10);
  return Number.isFinite(raw) && raw >= 0 && raw <= 72_000 ? raw : 9800;
}

function narrativeTransitionScreenMs(): number {
  const raw = Number.parseInt(process.env.DEMO_SEGUE_STRIP_MS ?? "", 10);
  return Number.isFinite(raw) && raw >= 4500 && raw <= 52_000 ? raw : 24_000;
}

function narrativeTransitionReadLeadMs(): number {
  const raw = Number.parseInt(process.env.DEMO_SEGUE_STRIP_READ_MS ?? "", 10);
  return Number.isFinite(raw) && raw >= 700 && raw <= 28_000 ? raw : 7600;
}

/** Copy A estimate segue hold — ``DEMO_SEGUE_AFTER_FIRST_ESTIMATE_MS``. Closing outro — ``DEMO_SESSION_OUTRO_STRIP_MS``. */
function afterFirstEstimateSegueHoldMs(): number {
  const raw = Number.parseInt(process.env.DEMO_SEGUE_AFTER_FIRST_ESTIMATE_MS ?? "", 10);
  return Number.isFinite(raw) && raw >= 900 && raw <= 30_000 ? raw : 8000;
}

/** Final “two copies…” strip — **`screenMs`/`readLeadMs` paired** so the harness waits through the closing copy before the JSON tail **`hold`. */
function sessionOutroStripHoldMs(): number {
  const raw = Number.parseInt(process.env.DEMO_SESSION_OUTRO_STRIP_MS ?? "", 10);
  return Number.isFinite(raw) && raw >= 6000 && raw <= 52_000 ? raw : 14_000;
}

/** Trims timed segues **after_first_grade** and **after_second_grade** vs generic ``DEMO_SEGUE_STRIP_*`` (`0` = no trim). */
function afterFirstGradeSegueTrimMs(): number {
  const raw = Number.parseInt(process.env.DEMO_SEGUE_AFTER_FIRST_GRADE_TRIM_MS ?? "", 10);
  return Number.isFinite(raw) && raw >= 0 && raw <= 12_000 ? raw : 0;
}

/** Extra clearance (px) between sell-dock top and narration strip bottom. */
function sellerStripAboveDockGapPx(): number {
  const raw = Number.parseInt(process.env.DEMO_STRIP_ABOVE_DOCK_GAP_PX ?? "", 10);
  return Number.isFinite(raw) && raw >= 0 && raw <= 80 ? raw : 18;
}

function setupChapterHeadline(kind: string): string {
  switch (kind) {
    case "inject_extension_storage":
      return "We just pulled two copies of the same White Album out of the attic — and they might be the rare misprinted first UK pressing";
    default:
      return "VinylIQ demo";
  }
}

function setupChapterSubtitle(kind: string): string {
  switch (kind) {
    case "inject_extension_storage":
      return `First edition, numbered, Apple label — and missing the "E.M.I. Recording" text that should have been on the label. EMI caught it quickly and issued a corrected pressing almost immediately after. These could be worth serious money, but condition is everything on a record this old. We're not experts, so we'll describe what we see, let VinylIQ grade each copy, and trust the market data before we put a price on anything.`;
    default:
      return "";
  }
}

function sellerGradeInsight(exampleIndex: number): {
  headline: string;
  body: string;
} {
  if (exampleIndex === 0) {
    return {
      headline: "Copy A first — this one had a rough ride",
      body: "Groove noise, ring wear, seams that have seen better decades. Let's be honest about it — if this pressing is as rare as we think, buyers will scrutinise every flaw. Describe what's there and let VinylIQ grade it.",
    };
  }
  return {
    headline: "Now for Copy B — same pressing, but something's different here",
    body: "Quieter surfaces, cleaner sleeve corners — this one got luckier in storage. Same rare misprint underneath, but a much more presentable copy. Let's describe it honestly and see how VinylIQ reads it.",
  };
}

function sellerEstimateInsight(exampleIndex: number): {
  headline: string;
  body: string;
} {
  if (exampleIndex === 0) {
    return {
      headline: "What's a misprinted first pressing in this condition actually worth?",
      body: "Discogs's suggestion already accounts for condition — but not closely enough. Watch VinylIQ nudge it back down a little: the detail in our notes pulled the grade lower than Discogs assumed, and the estimate follows. Not a huge gap, but on a record this valuable it's the difference between a fair listing and one that sits.",
    };
  }
  return {
    headline: "Same rare pressing — better condition — what does the market say?",
    body: "Discogs's suggested price is the highest this record has ever sold for — a ceiling, not a realistic target. VinylIQ factors in that the sleeve is grading VG+, not NM, and pulls the estimate back accordingly. That all-time high was someone's perfect copy. This isn't quite that — and pricing it honestly is how you earn a buyer's trust.",
  };
}

const NARRATIVE_SEGUES = {
  after_first_grade: {
    headline: "Grades are in for Copy A — now the important question",
    body: "We know what condition it's in. But what does that actually mean for the price? A rare misprinted pressing in rough shape is a very different conversation to one in good nick. Let's find out where Copy A lands.",
  },
  after_second_grade: {
    headline: "VinylIQ graded them differently — and the gap matters",
    body: "Same pressing, same attic, two very different grades. That's not a small thing on a record like this — condition is what separates a serious sale from leaving money on the table.",
  },
  after_first_estimate: {
    headline: "That's Copy A's number — hold onto it",
    body: "Keep that figure in mind. Copy B came out of the same attic, but it's been telling a different story since we opened the sleeve. Let's see if the grades — and the price — back that up.",
  },
} as const;

export type NarrativeSegueKey = keyof typeof NARRATIVE_SEGUES;

function narrativeSegueCopy(key: NarrativeSegueKey): {
  headline: string;
  body: string;
} {
  return NARRATIVE_SEGUES[key];
}

/**
 * Clears **`#vinyliq-overlay`** in **every** same-origin frame (**extension** attaches beside the dock).
 */
export async function removeVinyliqEstimateOverlay(page: Page): Promise<void> {
  const remover = (): void => {
    document.getElementById("vinyliq-overlay")?.remove();
  };
  for (const frame of page.frames()) {
    try {
      await frame.evaluate(remover);
    } catch {
      //
    }
  }
}

/**
 * Blurred fullscreen card (captures inside recordVideo from the active tab).
 * Used only for onboarding-style beats (inject + navigation setup).
 *
 * **`injectionTarget`:** **`main`** skip slow sell-form detection (no media row/dock on
 * Discogs `/`, `/release/…`, popup) — **`auto`** probes media **`select`** **Frame** then dock
 * (**`/sell/post/…`**).
 */
export async function showDemoChapterCard(
  page: Page,
  headline: string,
  subtitle: string,
  overlayDurationMs?: number,
  injectionTarget: "auto" | "main" = "auto",
): Promise<void> {
  if (demoVideoChaptersDisabledFromEnv()) {
    return;
  }
  await purgeDemoFloatingOverlaysInAllFrames(page);
  const durationMs =
    overlayDurationMs != null &&
    Number.isFinite(overlayDurationMs) &&
    overlayDurationMs >= 0
      ? Math.min(Math.max(Math.floor(overlayDurationMs), 0), 120_000)
      : demoChapterOverlayMs();
  const chapterPayload: ChapterInjectPayloadBrowser = {
    headline,
    subtitle,
    durationMs,
  };
  const host =
    injectionTarget === "main"
      ? page.mainFrame()
      : await resolveSellerNarrationInjectionFrame(page, {
          mediaAttachTimeoutMs: 120_000,
          dockFallbackTimeoutMs: 8000,
        });
  await host.evaluate(demoChapterInjectorInBrowser, chapterPayload);
  await page.waitForTimeout(durationMs);
}

/**
 * Bottom insight strip — leaves the Discogs listing visible while a seller-facing line reads out.
 */
export async function showSellerInsightStrip(
  page: Page,
  insight: { headline: string; body: string },
  opts?: SellerInsightStripOpts,
): Promise<void> {
  const readLead = opts?.readLeadMs ?? sellerInsightStripLeadMs();
  if (demoVideoChaptersDisabledFromEnv()) {
    await page.waitForTimeout(readLead);
    return;
  }
  await purgeDemoFloatingOverlaysInAllFrames(page);
  const screenMs = opts?.screenMs ?? sellerInsightStripScreenMs();
  const nid = opts?.stripElementId ?? "__vinyliq_demo_seller_strip";
  const stripPayload: SellerStripInjectPayloadBrowser = {
    headline: insight.headline,
    body: insight.body,
    stripId: nid,
    ttl: screenMs,
    dockGapPx: sellerStripAboveDockGapPx(),
  };
  const host = await resolveSellerNarrationInjectionFrame(page, {
    mediaAttachTimeoutMs: 120_000,
    dockFallbackTimeoutMs: 8000,
  });
  await host.evaluate(sellerInsightStripInjectorInBrowser, stripPayload);
  await page.waitForTimeout(readLead);
}

/**
 * Post-**Grade** strips (**demo_runner**): **Copy A** only when the ladder **changed** toward golden;
 * **Copy B** runs after each successful Grade (chapters on) even if selects already matched — autofill can
 * hide a **pre ≠ golden** delta. **Copy B:** two strips **12 s** / **10 s**; **Copy A:** single strip (**env**-tuned).
 */
export async function showPostGradeLadderUpdatedStrip(
  page: Page,
  exampleIndex: number,
): Promise<void> {
  if (demoVideoChaptersDisabledFromEnv()) {
    return;
  }
  if (exampleIndex === 1) {
    const { body } = narrativeSegueCopy("after_second_grade");
    await showSellerInsightStrip(
      page,
      {
        headline: "VinylIQ graded them differently — and the gap matters",
        body,
      },
      {
        stripElementId: "__vinyliq_demo_segue_after_second_grade_gap",
        screenMs: 12_000,
        readLeadMs: 12_000,
      },
    );
    await showSellerInsightStrip(
      page,
      {
        headline:
          "Grades locked — now let's see what Copy B is actually worth",
        body: "The ladder matches what VinylIQ read from your notes — next we ask the market what this copy should list for.",
      },
      {
        stripElementId: "__vinyliq_demo_segue_after_second_grade_locked",
        screenMs: 10_000,
        readLeadMs: 10_000,
      },
    );
    return;
  }
  const { body } = narrativeSegueCopy("after_first_grade");
  const slug = "after_first_grade_ladder_snap";
  const trim = afterFirstGradeSegueTrimMs();
  const screen = narrativeTransitionScreenMs();
  const lead = narrativeTransitionReadLeadMs();
  await showSellerInsightStrip(
    page,
    {
      headline: "VinylIQ graded them differently — and the gap matters",
      body,
    },
    {
      screenMs: Math.max(4500, screen - trim),
      readLeadMs: Math.max(700, lead - trim),
      stripElementId: `__vinyliq_demo_segue_${slug}`,
    },
  );
}

/** Timeout for listing **price** field to change after **Copy B** estimate — **`DEMO_COPY_B_PRICE_FIELD_POLL_MS`**. */
function copyBListingPricePollTimeoutMs(): number {
  const raw = Number.parseInt(
    process.env.DEMO_COPY_B_PRICE_FIELD_POLL_MS ?? "",
    10,
  );
  return Number.isFinite(raw) && raw >= 3000 && raw <= 360_000 ? raw : 120_000;
}

/**
 * **Copy B** estimate beat: wait until **`readListingPriceFingerprintExtensionOrder`**
 * (same selector order as **`fillListingSuggestedPrice`**) **≠** pre-estimate snapshot,
 * then **`sellerEstimateInsight(1)`** for **12 s**, then session **outro** (unless **`DEMO_SKIP_SEGUES`**).
 * If price **DOM** drifts from the SW target, the poll may time out — the strip still mounts so the viewer is not left with a blank beat.
 */
export async function runCopyBEstimateNarrativeWhenListingPriceUpdates(
  page: Page,
  listingPriceFingerprintBeforeEstimate: string,
): Promise<void> {
  if (demoVideoChaptersDisabledFromEnv()) {
    return;
  }
  const deadline = copyBListingPricePollTimeoutMs();
  try {
    await expect
      .poll(
        async () => readListingPriceFingerprintExtensionOrder(page),
        { timeout: deadline },
      )
      .not.toBe(listingPriceFingerprintBeforeEstimate);
  } catch {
    /* narrate anyway — see module doc above */
  }

  await showSellerInsightStrip(page, sellerEstimateInsight(1), {
    stripElementId: "__vinyliq_demo_aux_estimate_1",
    screenMs: 12_000,
    readLeadMs: 12_000,
  });

  if (!demoSkipNarrativeSegues()) {
    await showDemoSessionOutroStrip(page);
  }
}

export async function showNarrativeTransitionStrip(
  page: Page,
  key: NarrativeSegueKey,
): Promise<void> {
  if (demoVideoChaptersDisabledFromEnv()) {
    return;
  }
  if (
    demoSkipNarrativeSegues() &&
    key !== "after_first_grade" &&
    key !== "after_second_grade"
  ) {
    return;
  }
  const insight = narrativeSegueCopy(key);
  const slug = key.replace(/[^a-zA-Z0-9_]/g, "_");
  if (key === "after_first_estimate") {
    const holdMs = afterFirstEstimateSegueHoldMs();
    await showSellerInsightStrip(page, insight, {
      screenMs: holdMs,
      readLeadMs: holdMs,
      stripElementId: `__vinyliq_demo_segue_${slug}`,
    });
    return;
  }
  if (key === "after_first_grade" || key === "after_second_grade") {
    const trim = afterFirstGradeSegueTrimMs();
    const screen = narrativeTransitionScreenMs();
    const lead = narrativeTransitionReadLeadMs();
    await showSellerInsightStrip(page, insight, {
      screenMs: Math.max(4500, screen - trim),
      readLeadMs: Math.max(700, lead - trim),
      stripElementId: `__vinyliq_demo_segue_${slug}`,
    });
    return;
  }
  await showSellerInsightStrip(page, insight, {
    screenMs: narrativeTransitionScreenMs(),
    readLeadMs: narrativeTransitionReadLeadMs(),
    stripElementId: `__vinyliq_demo_segue_${slug}`,
  });
}

/** After Copy 1's seller estimate overlay beat: **`DEMO_AFTER_FIRST_ESTIMATE_BARE_MS`**, then **That's Copy A's number…**. */
export async function afterFirstEstimateNarrativeDwell(page: Page): Promise<void> {
  if (demoVideoChaptersDisabledFromEnv()) {
    return;
  }
  await page.waitForTimeout(afterFirstEstimateDwellBareMs());
  if (!demoSkipNarrativeSegues()) {
    await showNarrativeTransitionStrip(page, "after_first_estimate");
  }
}

/**
 * After Copy B's estimate-insight strip: bare UI dwell (**``afterFirstEstimateDwellBareMs``**),
 * then session outro (same role as **`after_first_estimate`** after Copy A's estimate).
 */
export async function afterSecondEstimateNarrativeDwell(page: Page): Promise<void> {
  if (demoVideoChaptersDisabledFromEnv()) {
    return;
  }
  await page.waitForTimeout(afterFirstEstimateDwellBareMs());
  if (!demoSkipNarrativeSegues()) {
    await showDemoSessionOutroStrip(page);
  }
}

/**
 * Catalog GOTO intro — run on the recorder tab once Discogs marketing hub path is **`/`** or **`/{locale}`**
 * (**see ``gotoSellerViaDiscogsUx``**: it waits internally, then invokes this immediately before navbar search).
 */
export async function showDemoCatalogSearchIntroChapter(
  page: Page,
): Promise<void> {
  await showDemoChapterCard(
    page,
    "Let's find the right release on Discogs and get these listed",
    "Search, into Masters, down to the pressing — let's make sure we're looking at the right one before we do anything else.",
    catalogSearchIntroChapterOverlayMs(),
    "main",
  );
}

/** On ``/release/{id}``, before verifying details (“Hmm…”). */
export async function showDemoCatalogReleaseUncertaintyChapter(
  page: Page,
): Promise<void> {
  await showDemoChapterCard(
    page,
    "Hmm — is this actually the right release?",
    "",
    releaseUncertaintyChapterOverlayMs(),
    "main",
  );
}

/** After a short scroll on the release page — confirmation before Sell. */
export async function showDemoCatalogReleaseConfirmedChapter(
  page: Page,
): Promise<void> {
  await showDemoChapterCard(
    page,
    "The details match — this is the one",
    "Numbered gatefold, dark green Apple label, misprint confirmed. Both copies are this pressing. Let's open the Sell page and get started.",
    catalogReleaseConfirmedChapterOverlayMs(),
    "main",
  );
}

/** Deep-linked / CI path: landed on Sell draft (`goto_seller_listing` runner). */
export async function showDemoSellLandingDeepLinkChapter(page: Page): Promise<void> {
  await showDemoChapterCard(
    page,
    "We're on the Sell page — let's get into it",
    "Found the pressing, confirmed the details — now we're here. Two very different copies to describe, starting with the rough one.",
  );
}

/** After Copy B overlay estimate — end of scripted pricing (long **`readLead`** + **`screenMs`** pair so “till the end” reads cleanly). */
export async function showDemoSessionOutroStrip(page: Page): Promise<void> {
  const hold = sessionOutroStripHoldMs();
  await showSellerInsightStrip(
    page,
    {
      headline: "Two copies, two honest prices — no guessing",
      body: "We described what we saw, VinylIQ graded each copy and checked the comps, and now we've got a number we can actually stand behind for both. That's the whole session.",
    },
    {
      stripElementId: "__vinyliq_demo_aux_outro",
      screenMs: hold,
      readLeadMs: hold,
    },
  );
}

/**
 * Audience-facing overlays (seller persona). Does **not** echo JSON ``title`` /
 * harness labels—the script uses fixed copy tailored for viewers.
 */
export async function maybeShowDemoChapter(
  page: Page,
  kind: string,
  exampleIndex?: number,
): Promise<void> {
  if (!demoVideoChaptersDisabledFromEnv() && SETUP_KINDS.has(kind)) {
    await showDemoChapterCard(
      page,
      setupChapterHeadline(kind),
      setupChapterSubtitle(kind),
      undefined,
      "main",
    );
    return;
  }
  if (
    demoVideoChaptersDisabledFromEnv() &&
    SETUP_KINDS.has(kind)
  ) {
    return;
  }

  if (kind === "grade_golden_example") {
    const ix = coerceScriptExampleIndex(exampleIndex);
    if (ix === undefined) {
      return;
    }
    await showSellerInsightStrip(page, sellerGradeInsight(ix), {
      stripElementId: `__vinyliq_demo_aux_grade_${ix}`,
    });
    return;
  }

  if (kind === "price_estimate_via_sw") {
    const ix = coerceScriptExampleIndex(exampleIndex);
    if (ix === undefined) {
      return;
    }
    if (ix === 1 && !demoVideoChaptersDisabledFromEnv()) {
      return;
    }
    const id = `__vinyliq_demo_aux_estimate_${ix}`;
    if (ix === 0) {
      await showSellerInsightStrip(page, sellerEstimateInsight(0), {
        stripElementId: id,
        readLeadMs: copyAEstimateStripReadLeadMs(),
        screenMs: COPY_A_ESTIMATE_STRIP_SCREEN_MS,
      });
      return;
    }
    await showSellerInsightStrip(page, sellerEstimateInsight(ix), {
      stripElementId: id,
      readLeadMs: narrativeTransitionReadLeadMs(),
      screenMs: narrativeTransitionScreenMs(),
    });
    return;
  }
}
