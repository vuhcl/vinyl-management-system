import { BrowserContext, chromium } from "@playwright/test";
import * as path from "path";
import * as fs from "fs";

/**
 * Launch a persistent Chromium context with the VinylIQ extension loaded.
 *
 * The extension and a Discogs-authenticated profile both have to persist
 * across runs so we use ``launchPersistentContext`` (Playwright's only
 * recommended path for MV3 extension testing — the default
 * ``chromium.launch`` cannot load extensions).
 *
 * Required env vars:
 *   - CHROME_PROFILE_DIR  Absolute path to the persistent profile dir
 *                         (Discogs project account already logged in).
 *   - EXTENSION_PATH      Absolute path to vinyliq-extension/ in the repo.
 *
 * Returns the launched ``BrowserContext`` plus the auto-detected
 * ``extensionId`` (parsed from the extension's service worker URL —
 * avoids the need to bake a ``key`` into manifest.json).
 */
export interface LaunchResult {
  context: BrowserContext;
  extensionId: string;
}

function requireEnv(name: string): string {
  const v = process.env[name];
  if (!v) {
    throw new Error(`Missing required env var: ${name}`);
  }
  return v;
}

export async function launchWithExtension(): Promise<LaunchResult> {
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

  const context = await chromium.launchPersistentContext(profileDir, {
    headless: false,
    viewport: { width: 1280, height: 800 },
    args: [
      `--disable-extensions-except=${extensionPath}`,
      `--load-extension=${extensionPath}`,
    ],
    recordVideo: {
      dir: "recordings/",
      size: { width: 1280, height: 800 },
    },
    slowMo: 200,
  });

  // Auto-detect the extension ID from the MV3 service worker URL.
  // The SW may register before we get a chance to ``waitForEvent`` (in
  // which case ``context.serviceWorkers()`` already lists it) or after
  // (so we wait). Cover both.
  const existing = context
    .serviceWorkers()
    .find((sw) => sw.url().startsWith("chrome-extension://"));
  const sw =
    existing ??
    (await context.waitForEvent("serviceworker", {
      timeout: 10_000,
      predicate: (worker) => worker.url().startsWith("chrome-extension://"),
    }));
  const extensionId = new URL(sw.url()).hostname;

  return { context, extensionId };
}
