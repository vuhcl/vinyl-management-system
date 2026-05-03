import type { Page } from "@playwright/test";

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

/** Skip bottom ``after_first_grade`` / ``after_first_estimate`` segue strips (chapter recording still honors bare dwell timings). */
export function demoSkipNarrativeSegues(): boolean {
  const v = (process.env.DEMO_SKIP_SEGUES ?? "").trim().toLowerCase();
  return ["1", "true", "yes"].includes(v);
}

/** Bare UI dwell after Copy 1's estimate clears (recording only; gated by chapters on). */
export function afterFirstEstimateDwellBareMs(): number {
  const raw = Number.parseInt(process.env.DEMO_AFTER_FIRST_ESTIMATE_BARE_MS ?? "", 10);
  return Number.isFinite(raw) && raw >= 0 && raw <= 72_000 ? raw : 19_000;
}

function narrativeTransitionScreenMs(): number {
  const raw = Number.parseInt(process.env.DEMO_SEGUE_STRIP_MS ?? "", 10);
  return Number.isFinite(raw) && raw >= 4500 && raw <= 52_000 ? raw : 24_000;
}

function narrativeTransitionReadLeadMs(): number {
  const raw = Number.parseInt(process.env.DEMO_SEGUE_STRIP_READ_MS ?? "", 10);
  return Number.isFinite(raw) && raw >= 700 && raw <= 28_000 ? raw : 7600;
}

/** “That's Copy A's number — hold onto it” segue-only (generic segues use ``DEMO_SEGUE_STRIP_*``). */
function afterFirstEstimateSegueHoldMs(): number {
  const raw = Number.parseInt(process.env.DEMO_SEGUE_AFTER_FIRST_ESTIMATE_MS ?? "", 10);
  return Number.isFinite(raw) && raw >= 900 && raw <= 30_000 ? raw : 5000;
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
      return `First edition, numbered, Apple label — and missing the "E.M.I. Recording" text that got fixed almost immediately. These could be worth serious money, but condition is everything on a record this old. We're not experts, so we'll describe what we see, let VinylIQ grade each copy, and trust the market data before we put a price on anything.`;
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
    body: "This is where it gets interesting. Discogs's suggestion is already high — but watch VinylIQ go significantly above it. Copy B's grades are genuinely strong, and precise grading on a record this rare earns a real premium. That's a gap worth knowing about before you publish.",
  };
}

const NARRATIVE_SEGUES = {
  after_first_grade: {
    headline: "Grades are in for Copy A — now the important question",
    body: "We know what condition it's in. But what does that actually mean for the price? A rare misprinted pressing in rough shape is a very different conversation to one in good nick. Let's find out where Copy A lands.",
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
 * Blurred fullscreen card (captures inside recordVideo from the active tab).
 * Used only for onboarding-style beats (inject + navigation setup).
 */
export async function showDemoChapterCard(
  page: Page,
  headline: string,
  subtitle: string,
  overlayDurationMs?: number,
): Promise<void> {
  if (demoVideoChaptersDisabledFromEnv()) {
    return;
  }
  const durationMs =
    overlayDurationMs != null &&
    Number.isFinite(overlayDurationMs) &&
    overlayDurationMs >= 0
      ? Math.min(Math.max(Math.floor(overlayDurationMs), 0), 120_000)
      : demoChapterOverlayMs();
  await page.evaluate(
    ({
      headline: h,
      subtitle: s,
      durationMs: ms,
    }: {
      headline: string;
      subtitle: string;
      durationMs: number;
    }) => {
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
    },
    { headline, subtitle, durationMs },
  );
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
  const screenMs = opts?.screenMs ?? sellerInsightStripScreenMs();
  const nid = opts?.stripElementId ?? "__vinyliq_demo_seller_strip";
  await page.evaluate(
    ({
      headline: hl,
      body: bd,
      stripId,
      ttl,
      dockGapPx,
    }: {
      headline: string;
      body: string;
      stripId: string;
      ttl: number;
      dockGapPx: number;
    }) => {
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
        zIndex: "2147483644",
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
        const p = document.createElement("div");
        p.textContent = bd;
        Object.assign(p.style, {
          fontSize: "clamp(14px, 1.95vw, 17px)",
          lineHeight: "1.52",
          opacity: "0.96",
          maxWidth: "960px",
          marginInline: "auto",
        });
        root.appendChild(p);
      }
      document.body.appendChild(root);
      window.setTimeout(() => document.getElementById(stripId)?.remove(), ttl);
    },
    { headline: insight.headline, body: insight.body, stripId: nid, ttl: screenMs, dockGapPx: sellerStripAboveDockGapPx() },
  );

  await page.waitForTimeout(readLead);
}

/**
 * Short bridge beats between scripted sections (recording / ``DEMO_VIDEO_CHAPTERS=1``).
 */
export async function showNarrativeTransitionStrip(
  page: Page,
  key: NarrativeSegueKey,
): Promise<void> {
  if (demoVideoChaptersDisabledFromEnv() || demoSkipNarrativeSegues()) {
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
  await showSellerInsightStrip(page, insight, {
    screenMs: narrativeTransitionScreenMs(),
    readLeadMs: narrativeTransitionReadLeadMs(),
    stripElementId: `__vinyliq_demo_segue_${slug}`,
  });
}

/**
 * After Copy 1's estimate: linger on the untouched UI, then segue strip into Copy B.
 */
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

/** ``assert_expected_grades_differ`` beat. */
export async function showDemoGradesDifferNarrationStrip(
  page: Page,
): Promise<void> {
  await showSellerInsightStrip(
    page,
    {
      headline: "VinylIQ graded them differently — and the gap matters",
      body: "Same pressing, same attic, two very different grades. That's not a small thing on a record like this — condition is what separates a serious sale from leaving money on the table.",
    },
    { stripElementId: "__vinyliq_demo_aux_grades_diff" },
  );
}

/** Before Copy B estimate strip. */
export async function showDemoPreSecondEstimateNarrationStrip(
  page: Page,
): Promise<void> {
  await showSellerInsightStrip(
    page,
    {
      headline: "Grades locked — now let's see what Copy B is actually worth",
      body: "Better condition on a rare pressing should move the number. Let's find out by how much.",
    },
    { stripElementId: "__vinyliq_demo_aux_pre_est_b" },
  );
}

/** After Copy B overlay estimate — end of scripted pricing. */
export async function showDemoSessionOutroStrip(page: Page): Promise<void> {
  await showSellerInsightStrip(
    page,
    {
      headline: "Two copies, two honest prices — no guessing",
      body: "We described what we saw, VinylIQ graded each copy and checked the comps, and now we've got a number we can actually stand behind for both. That's the whole session.",
    },
    { stripElementId: "__vinyliq_demo_aux_outro" },
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
    );
    return;
  }
  if (
    demoVideoChaptersDisabledFromEnv() &&
    SETUP_KINDS.has(kind)
  ) {
    return;
  }

  if (kind === "grade_golden_example" && typeof exampleIndex === "number") {
    await showSellerInsightStrip(page, sellerGradeInsight(exampleIndex));
    return;
  }

  if (kind === "price_estimate_via_sw" && typeof exampleIndex === "number") {
    const insight = sellerEstimateInsight(exampleIndex);
    const lead = sellerInsightStripLeadMs();
    const screen = sellerInsightStripScreenMs();
    const opts =
      exampleIndex === 0
        ? {
            readLeadMs: Math.max(400, Math.floor(lead / 2)),
            screenMs: Math.max(2800, Math.floor(screen / 2)),
          }
        : undefined;
    await showSellerInsightStrip(page, insight, opts);
    return;
  }
}
