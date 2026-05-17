import * as fs from "fs";
import * as path from "path";

import { expect } from "@playwright/test";

import type { Page } from "@playwright/test";

import { injectVinyliqStorageViaExtensionPopup } from "./extension";

import {
  demoCatalogUxEnabled,
  gotoSellerViaDiscogsUx,
  runHybridCatalogReleasePageNarration,
} from "./discogs_navigation";
import type { GoldenExample, GoldenPredictDemoJson } from "./golden";
import {
  readListingPriceFingerprintExtensionOrder,
  visibleMediaConditionSelect,
  visibleSleeveConditionSelect,
} from "./seller_listing_locators";

export {
  defaultSellerSelectors,
  listingPriceInputCompareKey,
  readListingPriceFingerprintExtensionOrder,
  visibleMediaConditionSelect,
  visibleSellerPriceInput,
  visibleSleeveConditionSelect,
} from "./seller_listing_locators";

import {
  afterFirstEstimateNarrativeDwell,
  afterSecondEstimateNarrativeDwell,
  demoScriptExampleIndex,
  demoVideoChaptersDisabledFromEnv,
  maybeShowDemoChapter,
  removeVinyliqEstimateOverlay,
  runCopyBEstimateNarrativeWhenListingPriceUpdates,
  showDemoCatalogReleaseConfirmedChapter,
  showDemoCatalogReleaseUncertaintyChapter,
  showDemoCatalogSearchIntroChapter,
  showDemoSellLandingDeepLinkChapter,
  showPostGradeLadderUpdatedStrip,
} from "./demo_video_ann";

export type DemoScriptStep =
  | { kind: "inject_extension_storage"; title?: string }
  | { kind: "goto_seller_listing"; title?: string }
  | { kind: "hold"; title?: string; ms: number }
  | { kind: "grade_golden_example"; title?: string; example_index: number }
  | {
      kind: "price_estimate_via_sw";
      title?: string;
      example_index: number;
    }
  | { kind: "assert_expected_grades_differ"; title?: string }
  | { kind: "assert_min_price_delta"; title?: string };

export interface DemoScriptDoc {
  schema_version: number;
  steps: DemoScriptStep[];
}

export function demoScriptPath(): string {
  const fromEnv = process.env.VINYLIQ_DEMO_SCRIPT?.trim();
  if (fromEnv && fromEnv.length > 0) {
    return path.resolve(fromEnv);
  }
  return path.resolve(__dirname, "default_demo.script.json");
}

export function loadDemoScript(): DemoScriptDoc {
  const fp = demoScriptPath();
  const raw = fs.readFileSync(fp, "utf8");
  const parsed = JSON.parse(raw) as DemoScriptDoc;
  if (!parsed?.steps?.length) {
    throw new Error(`Demo script ${fp} missing steps[].`);
  }
  return parsed;
}

export function skipHolds(): boolean {
  const v = (process.env.DEMO_SKIP_HOLDS ?? "").trim().toLowerCase();
  return ["1", "true", "yes"].includes(v);
}

/**
 * **`DEMO_HYBRID=1`** — scripted **annotations** + **condition-comment typing** stay
 * automated; **navigation**, **Grade condition**, dropdown tweaks, **Get estimate** are left
 * to the operator. Uses **`DEMO_HYBRID_STEP_TIMEOUT_MS`** (default ~**30 min** per graded/estimate beat).
 */
export function demoHybridOperatorMode(): boolean {
  const raw = (process.env.DEMO_HYBRID ?? "").trim().toLowerCase();
  return ["1", "true", "yes"].includes(raw);
}

/** Default hybrid wait budget — human beats can be slow during recording */
const HYBRID_DEFAULT_WAIT_MS = 1_800_000;

export function hybridStepTimeoutMs(): number {
  const raw = Number.parseInt(process.env.DEMO_HYBRID_STEP_TIMEOUT_MS ?? "", 10);
  return Number.isFinite(raw) && raw >= 30_000 ? raw : HYBRID_DEFAULT_WAIT_MS;
}

function hybridNavTimeoutMs(): number {
  const raw = Number.parseInt(process.env.DEMO_HYBRID_NAV_TIMEOUT_MS ?? "", 10);
  return Number.isFinite(raw) && raw >= 60_000 ? raw : hybridStepTimeoutMs();
}

/** After typing in hybrid mode, **`page.pause()`** opens the Playwright Inspector when **`DEMO_HYBRID_PAUSE=1`**. */
function hybridPagePauseAfterComment(): boolean {
  const raw = (process.env.DEMO_HYBRID_PAUSE ?? "").trim().toLowerCase();
  return ["1", "true", "yes"].includes(raw);
}

function automatedGradePollTimeoutMs(): number {
  return demoHybridOperatorMode() ? hybridStepTimeoutMs() : 120_000;
}

function overlayPollTimeoutMs(): number {
  return demoHybridOperatorMode() ? hybridStepTimeoutMs() : 120_000;
}

/** Loose bound for **`test.setTimeout`** when **`DEMO_HYBRID=1`** (many human beats). */
export function demoHybridSuggestedTestTimeoutMs(): number {
  const unclamped =
    hybridNavTimeoutMs() + hybridStepTimeoutMs() * 8 + 3_600_000;
  const cap = 14_400_000;
  const m = Math.min(unclamped, cap);
  return Math.max(m, 600_000);
}

function stepTitle(s: DemoScriptStep): string {
  return (s.title && s.title.trim()) || s.kind;
}

/** Keep in sync with ``vinyliq-extension/listing_dom.js`` ``COMMENT_SELECTORS``. */
export function commentFieldLocator(page: Page) {
  const chain =
    '#comments, textarea[name="comments"], input[name="comments"], input[id="comments"], ' +
    'textarea[id*="comment" i], textarea[name="release_comments"], textarea[name="description"]';
  return page.locator(chain).filter({ visible: true }).first();
}

/** Sellers’ comments render legibly in **`recordVideo`**; tune with **`DEMO_COMMENT_TYPING_DELAY_MS`**. */
function demoCommentTypingDelayMs(): number {
  const raw = Number.parseInt(process.env.DEMO_COMMENT_TYPING_DELAY_MS ?? "", 10);
  return Number.isFinite(raw) && raw >= 0 && raw <= 120 ? raw : 38;
}

async function focusAndTypeSellerComment(commentEl: ReturnType<typeof commentFieldLocator>, text: string): Promise<void> {
  await commentEl.click({ timeout: 30_000 });
  await commentEl.fill("");
  const delay = demoCommentTypingDelayMs();
  if (delay <= 0) {
    await commentEl.fill(text);
    return;
  }
  await commentEl.pressSequentially(text, { delay });
}

/** Same id as ``vinyliq-extension/content.js`` seller draft dock. */
export function vinyliqSellDockSelector(): string {
  return "#vinyliq-sell-dock";
}

function sellDockVisibilityTimeoutMs(): number {
  const raw = Number.parseInt(
    process.env.PLAYWRIGHT_SELL_DOCK_TIMEOUT_MS ?? "",
    10,
  );
  return Number.isFinite(raw) && raw >= 10_000 ? raw : 240_000;
}

/** Normalize ``/sell/post/id`` pathname match (handles trailing slashes, querystrings). */
function pagePathMatchesSellListing(
  currentUrl: string,
  sellPostUrl: string,
): boolean {
  let cur = "";
  let target = "";
  try {
    cur = new URL(currentUrl).pathname.replace(/\/$/, "") || "";
  } catch {
    return false;
  }
  try {
    target = new URL(sellPostUrl).pathname.replace(/\/$/, "") || "";
  } catch {
    return false;
  }
  if (!target) {
    return false;
  }
  return cur === target || cur.startsWith(`${target}/`);
}

function parseEstimateUsdFromVinyliqOverlay(text: string | null): number {
  if (!text) {
    throw new Error("VinylIQ overlay has no text.");
  }
  const normalized = text.replace(/\s+/g, " ").trim();
  const m = /Estimate:\s*\$([\d,]+\.?\d*)/i.exec(normalized);
  if (!m?.[1]) {
    throw new Error(
      `Could not parse Estimate USD from VinylIQ overlay: ${JSON.stringify(normalized.slice(0, 240))}`,
    );
  }
  const n = Number.parseFloat(m[1].replace(/,/g, ""));
  if (!Number.isFinite(n)) {
    throw new Error(`Non-finite overlay estimate: ${m[1]}`);
  }
  return n;
}

function escapeRegExp(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/**
 * Golden ladder strings are shorthand (**``Good``**, **``Near Mint``**). Discogs
 * often exposes **``Good (G)``** as **``<option>``** value or what
 * **`inputValue`** returns — strip one trailing **` (abbrev)`** tail for asserts.
 *
 * SYNC: aligns with **`selectedConditionLabel`** / ladder wording in
 * **`vinyliq-extension/listing_dom.js`**.
 */
export function ladderLabelForGoldenCompare(displayed: string): string {
  const t = displayed.trim();
  const idx = t.lastIndexOf(" (");
  if (idx >= 1 && t.endsWith(")")) {
    return t.slice(0, idx).trim();
  }
  return t;
}

export interface DemoRunContext {
  golden: GoldenPredictDemoJson;
  examples: GoldenExample[];
  seller: Page;
  extensionId: string;
  releaseId: string;
  sellPostUrl: string;
  minDelta: number;
  priceApiBase: string;
  graderApiBase: string;
  apiKey: string;
}

export async function runDemoScript(
  ctx: DemoRunContext,
  script: DemoScriptDoc,
  testStep: (
    title: string,
    body: () => Promise<void>,
  ) => Promise<void>,
): Promise<void> {
  const dockSel = vinyliqSellDockSelector();
  const dock = ctx.seller.locator(dockSel);

  async function pollOverlayEstimateUsd(timeoutMs: number): Promise<number> {
    const overlay = ctx.seller.locator("#vinyliq-overlay");
    await overlay.waitFor({ state: "visible", timeout: timeoutMs });
    await expect
      .poll(
        async () => {
          try {
            return parseEstimateUsdFromVinyliqOverlay(
              await overlay.textContent(),
            );
          } catch {
            return null;
          }
        },
        { timeout: timeoutMs },
      )
      .not.toBeNull();
    return parseEstimateUsdFromVinyliqOverlay(
      await overlay.textContent(),
    );
  }

  const estimateOnceViaDockOverlay = async (
    media: string,
    sleeve: string,
  ): Promise<number> => {
    const mediaLoc = visibleMediaConditionSelect(ctx.seller);
    const sleeveLoc = visibleSleeveConditionSelect(ctx.seller);
    await mediaLoc.waitFor({ state: "visible", timeout: 120_000 });
    await sleeveLoc.waitFor({ state: "visible", timeout: 120_000 });
    const lr = (s: string) =>
      new RegExp(`^${escapeRegExp(s.trim())}(\\s*\\(|$)`, "i");
    await mediaLoc.selectOption({ label: lr(media) });
    await sleeveLoc.selectOption({ label: lr(sleeve) });

    await dock.getByRole("button", { name: /^Get estimate$/i }).click({
      timeout: 120_000,
    });

    return pollOverlayEstimateUsd(overlayPollTimeoutMs());
  };

  let p1 = 0;
  let p2 = 0;
  let priceCall = 0;

  for (const step of script.steps) {
    const title = stepTitle(step);
    await testStep(title, async () => {
      switch (step.kind) {
        case "inject_extension_storage": {
          await injectVinyliqStorageViaExtensionPopup(
            ctx.seller.context(),
            ctx.extensionId,
            {
              priceApiBase: ctx.priceApiBase,
              graderApiBase: ctx.graderApiBase,
              apiKey: ctx.apiKey,
            },
          );
          await maybeShowDemoChapter(ctx.seller, step.kind);
          break;
        }
        case "goto_seller_listing": {
          if (demoHybridOperatorMode()) {
            if (demoCatalogUxEnabled()) {
              await showDemoCatalogSearchIntroChapter(ctx.seller);
              await runHybridCatalogReleasePageNarration(
                ctx.seller,
                ctx.releaseId,
                hybridNavTimeoutMs(),
              );
            } else {
              await showDemoSellLandingDeepLinkChapter(ctx.seller);
            }
            if (
              !pagePathMatchesSellListing(ctx.seller.url(), ctx.sellPostUrl)
            ) {
              await ctx.seller.waitForURL(
                /\/(?:www\.)?discogs\.com\/sell\/post\/\d+/i,
                { timeout: hybridNavTimeoutMs() },
              );
            }
          } else if (demoCatalogUxEnabled()) {
            await gotoSellerViaDiscogsUx(ctx.seller, ctx.golden);
          } else if (
            !pagePathMatchesSellListing(ctx.seller.url(), ctx.sellPostUrl)
          ) {
            await ctx.seller.goto(ctx.sellPostUrl, {
              waitUntil: "domcontentloaded",
              timeout: 240_000,
            });
          }
          await dock.waitFor({
            state: "visible",
            timeout: sellDockVisibilityTimeoutMs(),
          });
          if (!demoHybridOperatorMode() && !demoCatalogUxEnabled()) {
            await showDemoSellLandingDeepLinkChapter(ctx.seller);
          }
          break;
        }
        case "hold":
          if (!skipHolds() && step.ms > 0) {
            await ctx.seller.waitForTimeout(step.ms);
          }
          break;
        case "grade_golden_example": {
          const ix = demoScriptExampleIndex(step.example_index);
          const ex = ctx.examples[ix];
          if (!ex) {
            throw new Error(`No golden example index ${ix}`);
          }
          const commentEl = commentFieldLocator(ctx.seller);
          await commentEl.waitFor({ state: "visible", timeout: 120_000 });
          await commentEl.scrollIntoViewIfNeeded();
          await maybeShowDemoChapter(
            ctx.seller,
            step.kind,
            ix,
          );
          await focusAndTypeSellerComment(commentEl, ex.text);
          if (
            demoHybridOperatorMode() &&
            hybridPagePauseAfterComment()
          ) {
            await ctx.seller.pause();
          }
          const gradeBtn = dock.getByRole("button", {
            name: /^Grade condition$/i,
          });
          await gradeBtn.waitFor({ state: "visible", timeout: 120_000 });
          await expect(gradeBtn).toBeEnabled({ timeout: 120_000 });

          const mediaLoc = visibleMediaConditionSelect(ctx.seller);
          const sleeveLoc = visibleSleeveConditionSelect(ctx.seller);
          await mediaLoc.waitFor({ state: "visible", timeout: 120_000 });
          await sleeveLoc.waitFor({ state: "visible", timeout: 120_000 });
          const preMedia = ladderLabelForGoldenCompare(
            await mediaLoc.inputValue(),
          );
          const preSleeve = ladderLabelForGoldenCompare(
            await sleeveLoc.inputValue(),
          );

          if (!demoHybridOperatorMode()) {
            await gradeBtn.click({ timeout: 120_000 });
          }

          const gradeDeadline = automatedGradePollTimeoutMs();
          await expect
            .poll(
              async () =>
                ladderLabelForGoldenCompare(await mediaLoc.inputValue()),
              { timeout: gradeDeadline },
            )
            .toBe(ex.expected_media_condition);
          await expect
            .poll(
              async () =>
                ladderLabelForGoldenCompare(await sleeveLoc.inputValue()),
              { timeout: gradeDeadline },
            )
            .toBe(ex.expected_sleeve_condition);

          // Copy B narration must not depend on selects changing vs golden: Discogs/extension
          // can already show VG+/NM before **Grade**, so **pre === golden** and the old gate hid all strips.
          const ladderMovedOrCopyB =
            ix === 1 ||
            preMedia !== ex.expected_media_condition ||
            preSleeve !== ex.expected_sleeve_condition;
          if (
            ladderMovedOrCopyB &&
            !demoVideoChaptersDisabledFromEnv()
          ) {
            await showPostGradeLadderUpdatedStrip(ctx.seller, ix);
          }
          break;
        }
        case "price_estimate_via_sw": {
          await removeVinyliqEstimateOverlay(ctx.seller);
          const ix = demoScriptExampleIndex(step.example_index);
          const ex = ctx.examples[ix];
          if (!ex) {
            throw new Error(`No golden example index ${ix}`);
          }
          let listingPriceFingerprint = "";
          if (ix === 1) {
            listingPriceFingerprint =
              await readListingPriceFingerprintExtensionOrder(ctx.seller);
          }
          const p = demoHybridOperatorMode()
            ? await pollOverlayEstimateUsd(overlayPollTimeoutMs())
            : await estimateOnceViaDockOverlay(
                ex.expected_media_condition,
                ex.expected_sleeve_condition,
              );
          if (ix === 0) {
            await maybeShowDemoChapter(
              ctx.seller,
              step.kind,
              ix,
            );
          } else if (demoVideoChaptersDisabledFromEnv()) {
            await maybeShowDemoChapter(
              ctx.seller,
              step.kind,
              ix,
            );
          } else {
            await runCopyBEstimateNarrativeWhenListingPriceUpdates(
              ctx.seller,
              listingPriceFingerprint,
            );
          }
          priceCall++;
          if (priceCall === 1) {
            p1 = p;
            await afterFirstEstimateNarrativeDwell(ctx.seller);
          } else if (priceCall === 2) {
            p2 = p;
            if (demoVideoChaptersDisabledFromEnv()) {
              await afterSecondEstimateNarrativeDwell(ctx.seller);
            }
          }
          break;
        }
        case "assert_expected_grades_differ": {
          const [a0, b0] = ctx.examples;
          expect(
            `${a0.expected_media_condition}|${a0.expected_sleeve_condition}`,
          ).not.toBe(`${b0.expected_media_condition}|${b0.expected_sleeve_condition}`);
          break;
        }
        case "assert_min_price_delta":
          expect(Math.abs(p1 - p2)).toBeGreaterThanOrEqual(ctx.minDelta);
          break;
        default:
          throw new Error(
            `Unknown demo step ${(step as { kind?: string }).kind ?? "?"}`,
          );
      }
    });
  }
}
