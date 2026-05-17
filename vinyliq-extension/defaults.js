/**
 * Bundled API defaults (MV3 SW importScripts — no ES modules).
 * Replace at pack time per demo/prod channel; dev keeps localhost twins.
 *
 * Overrides: chrome.storage.sync (Options page).
 */
globalThis.__VINYLIQ_API_DEFAULTS =
  typeof globalThis.__VINYLIQ_API_DEFAULTS === "object" &&
  globalThis.__VINYLIQ_API_DEFAULTS
    ? globalThis.__VINYLIQ_API_DEFAULTS
    : {
        priceApiBase: "http://127.0.0.1:8801",
        graderApiBase: "http://127.0.0.1:8090",
      };
