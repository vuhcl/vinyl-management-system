import type { Locator, Page } from "@playwright/test";

/**
 * Keep sync with **`vinyliq-extension/listing_dom.js`** (``COMMENT_SELECTORS`` / condition selects).
 */
export function defaultSellerSelectors() {
  return {
    /** Deprecated for comment entry — prefer ``commentFieldLocator`` (``demo_runner.ts``). */
    commentSelector:
      '#comments, textarea[name="comments"], input[name="comments"], input[id="comments"], ' +
      'textarea[id*="comment" i], textarea[name="release_comments"], textarea[name="description"]',
    mediaSelector:
      'select#media_condition, select[name="media_condition"], select[name="condition"], select#condition, select[id*="media" i]',
    sleeveSelector:
      'select#sleeve_condition, select[name="sleeve_condition"], select[id*="sleeve" i]',
    /** SYNC: **`vinyliq-extension/listing_dom.js`** `PRICE_INPUT_SELECTORS`. */
    priceInputs:
      'input[name="price"], #price, input[id*="price" i], input[type="number"][name*="price" i]',
  };
}

/**
 * Visible Media / Sleeve only — Discogs hides duplicate selects; grading + polls must hit the live control.
 */
export function visibleMediaConditionSelect(page: Page): Locator {
  const s = defaultSellerSelectors();
  return page.locator(s.mediaSelector).filter({ visible: true }).first();
}

export function visibleSleeveConditionSelect(page: Page): Locator {
  const s = defaultSellerSelectors();
  return page.locator(s.sleeveSelector).filter({ visible: true }).first();
}

/**
 * Ordered like **`listing_dom.js`** **`PRICE_INPUT_SELECTORS`** — same precedence as **`querySelector`** in **`fillListingSuggestedPrice`** (no **`:visible`** filter).
 */
const PRICE_PROBE_SELECTORS = [
  'input[name="price"]',
  "#price",
  'input[id*="price" i]',
  'input[type="number"][name*="price" i]',
] as const;

/**
 * Read listing price **`input`** value using **`PRICE_INPUT_SELECTORS`** order (matches extension **`fillListingSuggestedPrice`**).
 */
export async function readListingPriceFingerprintExtensionOrder(
  page: Page,
): Promise<string> {
  for (const sel of PRICE_PROBE_SELECTORS) {
    const loc = page.locator(sel).first();
    const n = await loc.count().catch(() => 0);
    if (n === 0) {
      continue;
    }
    const raw = await loc.inputValue().catch(() => "");
    return listingPriceInputCompareKey(raw);
  }
  return "";
}

/**
 * @deprecated Prefer **`readListingPriceFingerprintExtensionOrder`** for SW sync — **`:visible`** can track the wrong **`input`**.
 */
export function visibleSellerPriceInput(page: Page): Locator {
  const s = defaultSellerSelectors();
  return page.locator(s.priceInputs).filter({ visible: true }).first();
}

/**
 * Stable string for asserting the visible price **changed** (commas/decimals tolerated).
 */
export function listingPriceInputCompareKey(raw: string): string {
  const t = raw.trim().replace(/,/g, "");
  if (t === "") {
    return "";
  }
  const n = Number.parseFloat(t.replace(/[^0-9.-]/g, ""));
  if (!Number.isFinite(n)) {
    return t;
  }
  const rounded = Math.round(n * 1e4) / 1e4;
  return String(rounded);
}
