import { chromium, type Browser, type BrowserContext, type Page } from "@playwright/test";
import * as crypto from "crypto";
import * as path from "path";
import * as fs from "fs";

import { DEMO_RECORD_VIDEO } from "./demo_video_ann";

/**
 * Launch a persistent Chromium context with the VinylIQ extension loaded.
 *
 * The extension and a Discogs-authenticated profile both have to persist
 * across runs so we use ``launchPersistentContext`` (Playwright's only
 * recommended path for MV3 extension testing — the default
 * ``chromium.launch`` cannot load extensions).
 *
 * Required env vars (persistent Chromium path — default):
 *   - CHROME_PROFILE_DIR  Absolute path to the persistent profile dir
 *                         (Discogs project account already logged in).
 *   - EXTENSION_PATH      Absolute path to vinyliq-extension/ in the repo.
 *
 * Optional:
 *   - PLAYWRIGHT_USE_SYSTEM_CHROME  **Ignored.** Stock Google Chrome and
 *     Edge no longer support unpacked extension side-loading via CLI flags;
 *     Playwright’s extension guide requires **bundled Chromium**
 *     (``channel: "chromium"``). The env var is left for backwards
 *     compatibility only; unset it after updating scripts.
 *   - PLAYWRIGHT_EXTENSION_SW_TIMEOUT_MS  Max ms to wait for the VinylIQ
 *     MV3 service worker (default 120000). Chrome may register it only after
 *     at least one tab exists; startup is often slower than 10s.
 *   - PLAYWRIGHT_REDUCE_AUTOMATION_SIGNALS  ``1``/``true``/``yes`` (default):
 *     drop Playwright’s ``--enable-automation`` and add
 *     ``--disable-blink-features=AutomationControlled`` so Discogs /
 *     Cloudflare re-challenge slightly less aggressively. Disable with ``0``.
 *   - PLAYWRIGHT_CONNECT_CDP  When set to an HTTP debugger URL such as ``http://127.0.0.1:9222``,
 *     skip bundled Chromium and **attach** to Google Chrome you started with
 *     ``--remote-debugging-port``. Load unpacked VinylIQ in that Chrome manually, clear
 *     Cloudflare there, then run **`npm test`**. End-of-test teardown calls ``browser.close()``,
 *     which disconnects Playwright without shutting down Chrome.
 * Optional with **``PLAYWRIGHT_CONNECT_CDP``**:
 *
 * - ``EXTENSION_PATH`` — when set alongside CDP (same repo path as bundled mode),
 *     Playwright **derives** the unpacked Chrome extension id from the canonical
 *     absolute path (SHA256 of path → Chrome's ``a–p`` id string), then opens ``popup.html`` in
 *     **every** CDP BrowserContext (**one initial wave**, plus sparse retries instead of spamming tabs each poll slice).
 *     Navigating ``chrome-extension://…/popup.html`` always appears as **a transient normal tab**, not Chrome's toolbar popup.
 *     Happens alongside the **seller** Discogs tab (preload **`DEMO_START_URL`** / **`SELL_POST_URL`**
 *     first in **`tests/demo.spec.ts`**); **`popup.html`** is still a separate robot tab —
 *     **never** Chrome’s toolbar bubble at ``npm test`` start.
 *     Poll the **default Chromium profile context plus** every enumerated context for MV3 targets.
 *
 *   - PLAYWRIGHT_CDP_MV3_PROBE_MS  milliseconds to wait for an enumerable MV3
 *     SW when attaching with a **pinned** id ( ``EXTENSION_PATH`` / ``VINYLIQ_EXTENSION_ID`` );
 *     default ``5000``. CDP commonly reports ``ext_sw=0`` while the extension runs.
 *   - PLAYWRIGHT_CDP_DOCK_FALLBACK  ``1``/``true`` (default): if the probe finishes
 *     without enumerable SW for pinned attach, skip the legacy long poll and proceed
 *     with `#vinyliq-sell-dock` + ``popup.html`` storage (demo runner). ``0``/``false``
 *     restores strict MV3 discovery for ``PLAYWRIGHT_EXTENSION_SW_TIMEOUT_MS``.
 *
 * Helpers:
 *   - ``launchWarmupPersistentContext``  ``CHROME_PROFILE_DIR`` only; bundled Chromium
 *     sans extension for ``warm-profile``.
 *
 * Attach path (**``PLAYWRIGHT_CONNECT_CDP``** non-empty): no ``CHROME_PROFILE_DIR`` /
 * ``EXTENSION_PATH`` — launch Google Chrome with ``--remote-debugging-port``, load
 * unpacked VinylIQ in **the same Chrome profile** Chrome was started with, manually pass
 * Cloudflare, optionally ``export EXTENSION_PATH=…/vinyliq-extension`` (path id
 * derivation) or ``export VINYLIQ_EXTENSION_ID=…`` from **chrome://extensions**, then attach.
 *
 * ``launchWithExtension`` returns ``BrowserContext`` plus ``extensionId`` (MV3 SW hostname
 * when visible, ``VINYLIQ_EXTENSION_ID`` / derived id when pinned and CDP falls back early).
 */
export interface LaunchResult {
  context: BrowserContext;
  extensionId: string;
}

/**
 * ``chromium.connectOverCDP`` keeps Chrome's implicit profile BrowserContext as
 * Playwright-internal ``_defaultContext``, which is **not** included in ``browser.contexts()``.
 * MV3 extension service workers attach to that context — scanning only ``.contexts()``
 * reports ``ext_sw=0`` for every slice (false negative).
 *
 * Exported for specs that probe ``worker`` handles over CDP the same way.
 */
export function chromiumContextsForMv3(browser: Browser): BrowserContext[] {
  const mapped = browser.contexts();
  const dc = (
    browser as unknown as {
      _defaultContext?: BrowserContext | null | undefined;
    }
  )._defaultContext;
  if (dc != null && !mapped.some((c) => c === dc)) {
    return [dc, ...mapped];
  }
  return mapped.slice();
}

function requireEnv(name: string): string {
  const v = process.env[name];
  if (!v) {
    throw new Error(`Missing required env var: ${name}`);
  }
  return v;
}

/** Shared with demo + warm-profile launches (sans extension). */
export function reduceAutomationSignalsFromEnv(): boolean {
  const raw = (process.env.PLAYWRIGHT_REDUCE_AUTOMATION_SIGNALS ?? "1")
    .trim()
    .toLowerCase();
  return raw !== "0" && raw !== "false" && raw !== "no";
}

/** Options for extension-backed persistent launches (demo recordings). */
export interface LaunchWithExtensionOptions {
  /** Demo defaults to recording video; warmup helpers can omit it for speed/I/O */
  recordVideo?: boolean;
  /** Default ``75`` (demo pacing); warmup uses ``0`` */
  slowMo?: number;
}

function connectCdpEndpoint(): string | undefined {
  const u = process.env.PLAYWRIGHT_CONNECT_CDP?.trim();
  return u && u.length > 0 ? u : undefined;
}

/** When CDP hides MV3 SW (`ext_sw=0`), attach can still succeed with pinned id + UI flow (dock / popup storage). */
function dockFirstCdpFallbackEnabled(): boolean {
  const raw = (process.env.PLAYWRIGHT_CDP_DOCK_FALLBACK ?? "1")
    .trim()
    .toLowerCase();
  return raw !== "0" && raw !== "false" && raw !== "no";
}

/** Short window to wait for an enumerable VinylIQ SW before fallback (pinned CDP attach). */
function cdpPinnedMv3ProbeMs(): number {
  const raw = Number.parseInt(
    process.env.PLAYWRIGHT_CDP_MV3_PROBE_MS ?? "",
    10,
  );
  return Number.isFinite(raw) && raw >= 500 && raw <= 120_000 ? raw : 5_000;
}

function pickPrimaryCdpBrowserContext(browser: Browser): BrowserContext {
  const ctxs = chromiumContextsForMv3(browser);
  if (ctxs.length === 0) {
    throw new Error(
      "INTERNAL: no Playwright contexts on CDP Browser (start Chrome open).",
    );
  }
  return ctxs[0]!;
}

function extensionPollBudgetMs(): number {
  const swCap = Number.parseInt(
    process.env.PLAYWRIGHT_EXTENSION_SW_TIMEOUT_MS ?? "",
    10,
  );
  return Number.isFinite(swCap) && swCap >= 5000 ? swCap : 120_000;
}

/** Optional pinned id when attaching to Chrome (``chrome-extension:///<id>/...`` hostname). */
function extensionIdFromEnv(): string | undefined {
  const id = process.env.VINYLIQ_EXTENSION_ID?.trim();
  if (!id) {
    return undefined;
  }
  if (/^[a-p]{32}$/.test(id)) {
    return id;
  }
  console.warn(
    "[launchWithExtension] VINYLIQ_EXTENSION_ID looks unusual (expected " +
      "32 lowercase hex letters Chrome assigns); continuing anyway.",
  );
  return id;
}

async function delay(ms: number): Promise<void> {
  await new Promise<void>((resolve) => {
    setTimeout(resolve, ms);
  });
}

function unpackedExtensionApId(extensionRootResolved: string): string {
  const alphabet = "abcdefghijklmnop";

  /** Matches Chromium unpacked path hashing (POSIX UTF-8; Windows UTF-16 LE). */
  function canonicalUnpackRoot(raw: string): string {
    let p = path.resolve(raw.trim());
    try {
      type RealpathFn = typeof fs.realpathSync;
      const nativeRp = (
        fs.realpathSync as RealpathFn & { native?: RealpathFn }
      ).native;
      p =
        typeof nativeRp === "function"
          ? nativeRp(p)
          : fs.realpathSync(p);
    } catch {
      try {
        p = fs.realpathSync(p);
      } catch {
        /* keep resolved literal */
      }
    }
    if (process.platform === "win32") {
      let w = p.replace(/\//g, "\\");
      if (/^[a-z]:\\/i.test(w)) {
        w = w.slice(0, 1).toUpperCase() + w.slice(1);
      }
      while (w.length > 3 && /\\$/.test(w)) {
        w = w.slice(0, -1);
      }
      return w;
    }
    while (p.length > 1 && p.endsWith(path.sep)) {
      p = p.slice(0, -1);
    }
    return p;
  }

  function unpackPathDigestInput(rootCanon: string): Buffer {
    if (process.platform === "win32") {
      const b = Buffer.allocUnsafe(rootCanon.length * 2);
      for (let i = 0; i < rootCanon.length; i++) {
        b.writeUInt16LE(rootCanon.charCodeAt(i), i * 2);
      }
      return b;
    }
    return Buffer.from(rootCanon, "utf8");
  }

  const rootCanon = canonicalUnpackRoot(extensionRootResolved);
  const input = unpackPathDigestInput(rootCanon);
  const hex32 = crypto
    .createHash("sha256")
    .update(input)
    .digest("hex")
    .slice(0, 32);
  let out = "";
  for (let i = 0; i < hex32.length; i++) {
    out += alphabet[parseInt(hex32[i]!, 16)];
  }
  return out;
}

/** When CDP attaches, ``EXTENSION_PATH`` can supply a deterministic unpacked id without MV3 enumeration. */
function resolvedExtensionPathForDerivation(): string | undefined {
  const raw = process.env.EXTENSION_PATH?.trim();
  if (!raw) {
    return undefined;
  }
  const abs = path.resolve(raw);
  if (
    !fs.existsSync(abs) ||
    !fs.existsSync(path.join(abs, "manifest.json"))
  ) {
    return undefined;
  }
  return abs;
}

/**
 * Extension hostname for ``chrome-extension://`` URLs — ``VINYLIQ_EXTENSION_ID`` wins,
 * else unpacked id derived from ``EXTENSION_PATH`` (optional with CDP).
 */
function resolvedCdpExtensionId(): string | undefined {
  const fromEnv = extensionIdFromEnv();
  if (fromEnv) {
    return fromEnv;
  }
  const rp = resolvedExtensionPathForDerivation();
  if (!rp) {
    return undefined;
  }
  try {
    return unpackedExtensionApId(rp);
  } catch {
    return undefined;
  }
}

async function wakeExtensionViaPopup(cx: BrowserContext, extensionId: string) {
  let pg: Page | undefined;
  try {
    pg = await cx.newPage();
    await pg.goto(`chrome-extension://${extensionId}/popup.html`, {
      timeout: 20_000,
      waitUntil: "domcontentloaded",
    });
    await pg.locator("body").waitFor({ timeout: 8000 }).catch(() => {});
    await delay(350);
  } catch {
    /* wrong id / policy */
  } finally {
    await pg?.close({ runBeforeUnload: false }).catch(() => {});
  }
}

/**
 * Visiting ``popup.html`` via ``goto`` opens a normal browser tab briefly (Chrome
 * does not use the toolbar bubble for scripted navigation). Use sparingly: only
 * the first CDP wake plus rare retries so operators are not flooded with tabs.
 */
async function wakeExtensionViaPopupSparse(
  browser: Browser,
  extensionId: string,
): Promise<void> {
  const contexts = chromiumContextsForMv3(browser);
  for (const cx of contexts) {
    await wakeExtensionViaPopup(cx, extensionId);
  }
}

/**
 * Seed ``chrome.storage.sync`` from ``popup.html`` extension origin — works when CDP
 * hides the MV3 service worker from Playwright enumeration.
 */
export async function injectVinyliqStorageViaExtensionPopup(
  context: BrowserContext,
  extensionId: string,
  vals: Record<string, string>,
): Promise<void> {
  let pg: Page | undefined;
  try {
    pg = await context.newPage();
    await pg.goto(`chrome-extension://${extensionId}/popup.html`, {
      timeout: 30_000,
      waitUntil: "domcontentloaded",
    });
    await pg.locator("body").waitFor({ timeout: 15_000 });
    await pg.evaluate(
      `
      vals => new Promise(function (resolve, reject) {
        chrome.storage.sync.set(vals, function () {
          var err = chrome.runtime.lastError;
          if (err && err.message) {
            reject(new Error(err.message));
          } else {
            resolve(undefined);
          }
        });
      })
    `,
      vals,
    );
    await delay(200);
  } finally {
    await pg?.close({ runBeforeUnload: false }).catch(() => {});
  }
}

async function waitForMv3ExtensionId(
  context: BrowserContext,
  origin: "persistent" | "cdp",
): Promise<string> {
  const fromEnv = extensionIdFromEnv();
  if (fromEnv) {
    return fromEnv;
  }

  function findWorker() {
    return context
      .serviceWorkers()
      .find((worker) => worker.url().startsWith("chrome-extension://"));
  }

  const budgetMs = extensionPollBudgetMs();
  const boot = context.pages()[0] ?? (await context.newPage());
  await boot.goto("about:blank");

  let sw = findWorker();
  const deadline = Date.now() + budgetMs;

  while (!sw && Date.now() < deadline) {
    const slice = Math.min(5_000, Math.max(0, deadline - Date.now()));
    if (slice <= 0) {
      break;
    }
    sw = findWorker();
    if (sw) {
      break;
    }
    try {
      await context.waitForEvent("serviceworker", {
        timeout: slice,
        predicate: (worker) =>
          worker.url().startsWith("chrome-extension://"),
      });
    } catch {
      /* keep polling until budget elapses */
    }
    sw = findWorker();
    if (!sw) {
      await delay(400);
    }
  }

  if (!sw) {
    const extHint =
      origin === "cdp"
        ? "CDP attach: enable VinylIQ at chrome://extensions in that Chrome profile, or set VINYLIQ_EXTENSION_ID."
        : "Persistent Chromium: verify EXTENSION_PATH and chrome://extensions for CHROME_PROFILE_DIR.";
    throw new Error(
      `Extension MV3 service worker did not appear within ms=${budgetMs}. ${extHint} ` +
        `Raise PLAYWRIGHT_EXTENSION_SW_TIMEOUT_MS if startup is slow.`,
    );
  }

  return new URL(sw.url()).hostname;
}

/**
 * ``context.close()`` on a Chromium profile launched by Playwright is correct.
 * When **``PLAYWRIGHT_CONNECT_CDP``** is set, ``context`` comes from Chrome over CDP —
 * call this instead so ``browser.close()`` disconnects automation without quitting Chrome.
 */
export async function disposeDemoBrowser(context: BrowserContext): Promise<void> {
  if (connectCdpEndpoint()) {
    const browser = context.browser();
    if (browser) {
      await browser.close().catch(() => undefined);
    }
    return;
  }
  await context.close();
}

function summarizeCdpContexts(browser: Browser): string {
  return chromiumContextsForMv3(browser)
    .map((c, i) => {
      let extSw = 0;
      try {
        extSw = c
          .serviceWorkers()
          .filter((w) => w.url().startsWith("chrome-extension://")).length;
      } catch {
        extSw = 0;
      }
      return `#${i}:pages=${c.pages().length},ext_sw=${extSw}`;
    })
    .join("; ");
}



function pickMv3ChromeExtensionById(
  browser: Browser,
  extensionId: string,
):
  | { context: BrowserContext; extensionId: string }
  | undefined {
  const prefix = `chrome-extension://${extensionId}`;
  for (const c of chromiumContextsForMv3(browser)) {
    const w = c
      .serviceWorkers()
      .find((worker) => worker.url().startsWith(prefix));
    if (w) {
      return { context: c, extensionId };
    }
  }
  return undefined;
}

function pickMv3ChromeExtensionFromBrowser(browser: Browser):
  | { context: BrowserContext; extensionId: string }
  | undefined {
  for (const c of chromiumContextsForMv3(browser)) {
    const w = c
      .serviceWorkers()
      .find((worker) => worker.url().startsWith("chrome-extension://"));
    if (w) {
      return {
        context: c,
        extensionId: new URL(w.url()).hostname,
      };
    }
  }
  return undefined;
}

async function acquireCdpLaunchResult(browser: Browser): Promise<LaunchResult> {
  const pinned = resolvedCdpExtensionId();
  if (pinned) {
    if (extensionIdFromEnv()) {
      console.warn(
        "[launchWithExtension] VINYLIQ_EXTENSION_ID set — pinning id; polling all CDP contexts for MV3.",
      );
    } else {
      console.warn(
        "[launchWithExtension] Derived extension id from EXTENSION_PATH (CDP attach). " +
          "If chrome://extensions shows a different id, export VINYLIQ_EXTENSION_ID.",
      );
    }

    await wakeExtensionViaPopupSparse(browser, pinned);

    const probeMs = cdpPinnedMv3ProbeMs();
    const probeDeadline = Date.now() + probeMs;
    let hit = pickMv3ChromeExtensionById(browser, pinned);

    while (!hit && Date.now() < probeDeadline) {
      const slice = Math.min(
        2_500,
        Math.max(400, probeDeadline - Date.now()),
      );
      await Promise.race(
        chromiumContextsForMv3(browser).map(async (cx) => {
          try {
            await cx.waitForEvent("serviceworker", {
              timeout: slice,
              predicate: (w) =>
                w.url().startsWith(`chrome-extension://${pinned}`),
            });
          } catch {
            /* keep polling */
          }
        }),
      );
      hit = pickMv3ChromeExtensionById(browser, pinned);
      if (hit) {
        console.warn("[launchWithExtension] MV3 service worker matched pinned extension id.");
        return hit;
      }
      await wakeExtensionViaPopupSparse(browser, pinned).catch(() => {});
      hit = pickMv3ChromeExtensionById(browser, pinned);
      if (hit) {
        console.warn("[launchWithExtension] MV3 service worker matched pinned extension id.");
        return hit;
      }
      await delay(200);
    }

    if (dockFirstCdpFallbackEnabled()) {
      console.warn(
        "[launchWithExtension] MV3 SW not visible over CDP after probe ms=" +
          String(probeMs) +
          " (ext_sw may stay false); dock-first demo. " +
          summarizeCdpContexts(browser) +
          " — suppress with PLAYWRIGHT_CDP_DOCK_FALLBACK=0 to require enumerable SW.",
      );
      return {
        context: pickPrimaryCdpBrowserContext(browser),
        extensionId: pinned,
      };
    }

    const budgetMs = extensionPollBudgetMs();
    const deadline = Date.now() + budgetMs;
    let lastLog = Date.now();
    const pollAnchorMs = Date.now();
    /** Extra popup-tab wakes (same cost as CDP opener): spaced, not every poll slice. */
    const secondaryWakeAfterMs = [22_000, 55_000];
    let secondaryWakeIdx = 0;

    while (!hit && Date.now() < deadline) {
      const slice = Math.min(4000, Math.max(50, deadline - Date.now()));

      await Promise.race(
        chromiumContextsForMv3(browser).map(async (cx) => {
          try {
            await cx.waitForEvent("serviceworker", {
              timeout: slice,
              predicate: (w) =>
                w.url().startsWith(`chrome-extension://${pinned}`),
            });
          } catch {
            /* keep polling */
          }
        }),
      );

      hit = pickMv3ChromeExtensionById(browser, pinned);
      if (hit) {
        console.warn("[launchWithExtension] MV3 service worker matched pinned extension id.");
        return hit;
      }

      while (
        secondaryWakeIdx < secondaryWakeAfterMs.length &&
        Date.now() - pollAnchorMs >= secondaryWakeAfterMs[secondaryWakeIdx]!
      ) {
        await wakeExtensionViaPopupSparse(browser, pinned).catch(() => {});
        secondaryWakeIdx++;
      }

      if (Date.now() - lastLog >= 12_000) {
        console.warn(
          "[launchWithExtension] Still waiting for MV3 (pinned). " +
            summarizeCdpContexts(browser) +
            " — ensure VinylIQ is enabled at chrome://extensions " +
            "or raise PLAYWRIGHT_EXTENSION_SW_TIMEOUT_MS. " +
            "(Script does not start on `/sell/post/` — no toolbar popup is expected.)",
        );
        lastLog = Date.now();
      }
      await delay(400);
      hit = pickMv3ChromeExtensionById(browser, pinned);
    }

    if (hit) {
      return hit;
    }

    throw new Error(
      `CDP attach: VinylIQ MV3 for id=${pinned} did not surface within ms=${budgetMs}. ` +
        `Contexts: ${summarizeCdpContexts(browser)} — ` +
        `confirm VinylIQ is enabled at chrome://extensions — ` +
        `if you set PLAYWRIGHT_CDP_DOCK_FALLBACK=0, unset it ` +
        `so default (dock-friendly attach without enumerable MV3) applies.`,
    );
  }

  console.warn(
    "[launchWithExtension] Scanning contexts for VinylIQ MV3. " +
      summarizeCdpContexts(browser),
  );

  const budgetMs = extensionPollBudgetMs();
  const deadline = Date.now() + budgetMs;
  let lastLog = Date.now();

  let seededContexts = false;
  while (Date.now() < deadline) {
    const immediate = pickMv3ChromeExtensionFromBrowser(browser);
    if (immediate) {
      return immediate;
    }

    // One-time nudge: MV3 SW often registers after a normal tab exists.
    if (!seededContexts) {
      for (const c of chromiumContextsForMv3(browser)) {
        try {
          const p = c.pages()[0] ?? (await c.newPage());
          await p.goto("about:blank", {
            timeout: 15_000,
            waitUntil: "domcontentloaded",
          });
        } catch {
          /* ignore flaky tabs */
        }
      }
      seededContexts = true;
    }

    const slice = Math.min(4_000, Math.max(0, deadline - Date.now()));
    if (slice <= 50) break;

    await Promise.race(
      chromiumContextsForMv3(browser).map(async (cx) => {
        try {
          await cx.waitForEvent("serviceworker", {
            timeout: slice,
            predicate: (w) =>
              w.url().startsWith("chrome-extension://"),
          });
        } catch {
          /* keep polling */
        }
      }),
    );

    const found = pickMv3ChromeExtensionFromBrowser(browser);
    if (found) {
      console.warn("[launchWithExtension] MV3 service worker detected.");
      return found;
    }

    if (Date.now() - lastLog >= 12_000) {
      console.warn(
        "[launchWithExtension] Still waiting for unpacked MV3. " +
          summarizeCdpContexts(browser) +
          " — open chrome://extensions, enable VinylIQ, or set VINYLIQ_EXTENSION_ID.",
      );
      lastLog = Date.now();
    }
    await delay(300);
  }

  throw new Error(
    `CDP attach: VinylIQ MV3 did not surface within ms=${budgetMs}. ` +
      `Details: ${summarizeCdpContexts(browser)} — ` +
      `set PLAYWRIGHT_EXTENSION_SW_TIMEOUT_MS higher, verify Load unpacked VinylIQ ` +
      `in this Chrome user-data-dir, or export VINYLIQ_EXTENSION_ID from chrome://extensions.`,
  );
}

async function launchWithExtensionViaCdp(
  options: LaunchWithExtensionOptions,
): Promise<LaunchResult> {
  const endpoint = connectCdpEndpoint();
  if (!endpoint) {
    throw new Error("INTERNAL: CDP attach without PLAYWRIGHT_CONNECT_CDP.");
  }

  console.warn(
    "[launchWithExtension] Attaching via CDP (" +
      endpoint +
      "). Start Google Chrome with --remote-debugging-port. " +
      "Load unpacked VinylIQ and pass Cloudflare before the demo navigates.",
  );

  if (options.recordVideo !== false) {
    console.warn(
      "[launchWithExtension] recordVideo is ignored while PLAYWRIGHT_CONNECT_CDP is set.",
    );
  }

  const browser = await chromium.connectOverCDP(endpoint);

  console.warn("[launchWithExtension] CDP connected: " + summarizeCdpContexts(browser));

  if (chromiumContextsForMv3(browser).length === 0) {
    await browser.close().catch(() => undefined);
    throw new Error(
      "connectOverCDP: no browser contexts — start Chrome with a normal window visible " +
        "before attaching.",
    );
  }

  return acquireCdpLaunchResult(browser);
}

/**
 * Same bundled Chromium fingerprint as extension demo, **without**
 * unpacking VinylIQ — fewer background requests hitting Discogs so Cloudflare
 * tends to intervene less often during profile seeding.
 *
 * Cookies still land under ``CHROME_PROFILE_DIR``; ``npm test`` loads the extension
 * in a later Chromium session against the same dir.
 */
export async function launchWarmupPersistentContext(): Promise<BrowserContext> {
  const profileDir = path.resolve(requireEnv("CHROME_PROFILE_DIR"));
  const reduceAutomationSignals = reduceAutomationSignalsFromEnv();

  const ignoreDefaultArgs =
    reduceAutomationSignals ? ["--enable-automation"] : undefined;

  const chromiumArgs: string[] = [];
  if (reduceAutomationSignals) {
    chromiumArgs.push("--disable-blink-features=AutomationControlled");
  }

  const context = await chromium.launchPersistentContext(profileDir, {
    channel: "chromium",
    headless: false,
    viewport: { width: 1280, height: 800 },
    ...(ignoreDefaultArgs ? { ignoreDefaultArgs } : {}),
    ...(chromiumArgs.length ? { args: chromiumArgs } : {}),
    slowMo: 0,
  });

  const boot = context.pages()[0] ?? (await context.newPage());
  await boot.goto("about:blank");
  return context;
}

export async function launchWithExtension(
  options: LaunchWithExtensionOptions = {},
): Promise<LaunchResult> {
  if (connectCdpEndpoint()) {
    return launchWithExtensionViaCdp(options);
  }

  const recordVideoEnabled = options.recordVideo !== false;
  const profileDir = path.resolve(requireEnv("CHROME_PROFILE_DIR"));
  const extensionPath = path.resolve(requireEnv("EXTENSION_PATH"));

  if (!fs.existsSync(extensionPath)) {
    throw new Error(`EXTENSION_PATH does not exist: ${extensionPath}`);
  }
  if (!fs.existsSync(path.join(extensionPath, "manifest.json"))) {
    throw new Error(
      `EXTENSION_PATH must contain manifest.json: ${extensionPath}`
    );
  }

  const wantsSysChrome = ["1", "true", "yes"].includes(
    (process.env.PLAYWRIGHT_USE_SYSTEM_CHROME ?? "").trim().toLowerCase()
  );
  const reduceAutomationSignals = reduceAutomationSignalsFromEnv();

  const ignoreDefaultArgs = ["--disable-extensions"];
  if (reduceAutomationSignals) {
    ignoreDefaultArgs.push("--enable-automation");
  }

  const chromiumArgs = [
    `--disable-extensions-except=${extensionPath}`,
    `--load-extension=${extensionPath}`,
  ];
  if (reduceAutomationSignals) {
    chromiumArgs.push("--disable-blink-features=AutomationControlled");
  }

  if (wantsSysChrome) {
    // https://playwright.dev/docs/chrome-extensions — side-load flags are not
    // supported on stock Chrome/Edge; using channel chrome here makes the process
    // exit immediately ("Browser window not found" / getWindowForTarget).
    console.warn(
      "[launchWithExtension] PLAYWRIGHT_USE_SYSTEM_CHROME is ignored for " +
        "extension E2E; using Playwright bundled Chromium (channel chromium). " +
        "Unset PLAYWRIGHT_USE_SYSTEM_CHROME to silence this.",
    );
  }

  const context = await chromium.launchPersistentContext(profileDir, {
    // Required for unpacked MV3 extensions; do not substitute Google Chrome.
    channel: "chromium",
    headless: false,
    viewport: { width: 1280, height: 800 },
    // Playwright injects --disable-extensions by default; that fights
    // --load-extension/--disable-extensions-except in some Chromium builds.
    ignoreDefaultArgs,
    args: chromiumArgs,
    ...(recordVideoEnabled
      ? {
          recordVideo: { ...DEMO_RECORD_VIDEO },
        }
      : {}),
    slowMo: options.slowMo ?? 75,
  });

  const extensionId = await waitForMv3ExtensionId(context, "persistent");
  return { context, extensionId };
}
