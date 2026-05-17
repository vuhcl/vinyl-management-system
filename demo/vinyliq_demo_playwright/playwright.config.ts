import { defineConfig } from "@playwright/test";

// Headed Chromium + extension (see fixtures/extension.ts). 1280x800 matches README embed.
// Primary demo recordings use fixtures/extension.ts persistent launch recordVideo
// (see fixtures/demo_video_ann.ts). Keeping video: off here avoids a second capture
// from the implicit project browser.
//
// `demo` — 2-minute recording (demo.spec.ts). `pitch-assist` — live pitch (pitch-assist.spec.ts).
const sharedUse = {
  headless: false,
  viewport: { width: 1280, height: 800 },
  video: "off" as const,
  actionTimeout: 60_000,
  navigationTimeout: 120_000,
  launchOptions: {
    slowMo: 75,
  },
};

export default defineConfig({
  testDir: "./tests",
  fullyParallel: false,
  workers: 1,
  retries: 0,
  // Seller + release navigations often hit Cloudflare; allow manual solve +
  // slow Discogs DOM without tripping default 30s.
  timeout: 180_000,
  expect: { timeout: 60_000 },
  outputDir: "./recordings",
  projects: [
    {
      name: "demo",
      testMatch: /demo\.spec\.ts/,
      use: sharedUse,
    },
    {
      name: "pitch-assist",
      testMatch: /pitch-assist\.spec\.ts/,
      use: sharedUse,
    },
  ],
});
