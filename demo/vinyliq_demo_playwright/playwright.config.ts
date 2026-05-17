import { defineConfig } from "@playwright/test";

const sharedUse = {
  headless: false,
  viewport: { width: 1280, height: 800 },
  video: { mode: "on" as const, size: { width: 1280, height: 800 } },
  actionTimeout: 15_000,
  navigationTimeout: 30_000,
  launchOptions: {
    slowMo: 200,
  },
};

// Headed Chromium + extension (see fixtures/extension.ts). 1280x800 matches README embed.
// `demo` — 2-minute recording (demo.spec.ts). `pitch-assist` — live pitch (pitch-assist.spec.ts).
export default defineConfig({
  testDir: "./tests",
  fullyParallel: false,
  workers: 1,
  retries: 0,
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
