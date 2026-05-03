import { defineConfig } from "@playwright/test";

// Single demo, sequential. Headed Chromium (extension testing requires a
// real browser, not the headless shell). 1280x800 is the YouTube
// 720p-friendly recording size used by the README embed.
//
// Primary demo recordings use `fixtures/extension.ts` persistent launch
// `recordVideo` (see `fixtures/demo_video_ann.ts` → action highlights + scripted
// chapter overlays when `npm test` runs bundled Chromium). Keeping `video: off`
// here avoids a second unrelated capture from the implicit project browser.
//
// `outputDir` retains Playwright artefacts (reports, screenshots). The runbook:
// RECORDING.md / README — convert `.webm` from `recordings/` to `demo.mp4`.
export default defineConfig({
  testDir: "./tests",
  fullyParallel: false,
  workers: 1,
  retries: 0,
  // Seller + release navigations often hit Cloudflare; allow manual solve +
  // slow Discogs DOM without tripping default 30s.
  timeout: 180_000,
  expect: { timeout: 60_000 },
  use: {
    headless: false,
    viewport: { width: 1280, height: 800 },
    video: "off",
    actionTimeout: 60_000,
    navigationTimeout: 120_000,
    launchOptions: {
      slowMo: 75,
    },
  },
  outputDir: "./recordings",
});
