import { defineConfig } from "@playwright/test";

// Single demo, sequential. Headed Chromium (extension testing requires a
// real browser, not the headless shell). 1280x800 is the YouTube
// 720p-friendly recording size used by the README embed.
//
// `outputDir` collects the .webm files Playwright generates per test;
// the recording runbook (k8s/demo/README.md does not own this — see
// tools/vinyliq_demo_playwright/README.md) converts the chosen .webm
// to .mp4 with ffmpeg before uploading to GitHub.
export default defineConfig({
  testDir: "./tests",
  fullyParallel: false,
  workers: 1,
  retries: 0,
  use: {
    headless: false,
    viewport: { width: 1280, height: 800 },
    video: { mode: "on", size: { width: 1280, height: 800 } },
    actionTimeout: 15_000,
    navigationTimeout: 30_000,
    launchOptions: {
      slowMo: 200,
    },
  },
  outputDir: "./recordings",
});
