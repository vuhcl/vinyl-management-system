importScripts("defaults.js");

/**
 * VinylIQ extension service worker.
 *
 * Proxies API calls from page-context scripts to VinylIQ backends
 * (`host_permissions` in manifest bypass CORS).
 *
 * Resolve order (see planner): bundled **defaults.js** <
 * chrome.storage.sync (Options overrides) <
 * GRADE / ESTIMATE message explicit apiBase overrides (backward compat dev).
 */

function trimStr(v) {
  return v != null ? String(v).trim() : "";
}

function buildUrl(apiBase, path) {
  return `${String(apiBase || "").replace(/\/$/, "")}${path}`;
}

function buildHeaders(apiKey) {
  const headers = { "Content-Type": "application/json" };
  if (apiKey) {
    headers["X-API-Key"] = apiKey;
  }
  return headers;
}

async function resolveApiRuntimeConfig() {
  const defs = globalThis.__VINYLIQ_API_DEFAULTS ?? {
    priceApiBase: "http://127.0.0.1:8801",
    graderApiBase: "http://127.0.0.1:8090",
  };
  const synced = await chrome.storage.sync.get([
    "priceApiBase",
    "graderApiBase",
    "apiKey",
    "apiBase",
  ]);
  const priceApiBase =
    trimStr(synced.priceApiBase) ||
    trimStr(synced.apiBase) ||
    trimStr(defs.priceApiBase) ||
    "http://127.0.0.1:8801";
  const graderApiBase =
    trimStr(synced.graderApiBase) ||
    trimStr(defs.graderApiBase) ||
    "http://127.0.0.1:8090";
  const apiKey = trimStr(synced.apiKey) || "";
  return { priceApiBase, graderApiBase, apiKey };
}

async function callApi(url, headers, body) {
  const r = await fetch(url, {
    method: "POST",
    headers,
    body: JSON.stringify(body),
  });
  const text = await r.text();
  let data;
  try {
    data = JSON.parse(text);
  } catch {
    data = { error: text || r.statusText };
  }
  if (!r.ok) {
    return { ok: false, status: r.status, data };
  }
  return { ok: true, data };
}

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message?.type === "ESTIMATE") {
    const { apiBase: msgPriceBase, apiKey: msgKey, body } = message;
    resolveApiRuntimeConfig()
      .then((cfg) => {
        const apiBase =
          trimStr(msgPriceBase).length > 0 ? trimStr(msgPriceBase) : cfg.priceApiBase;
        const apiKey = trimStr(msgKey).length > 0 ? trimStr(msgKey) : cfg.apiKey;
        const url = buildUrl(apiBase, "/estimate");
        return callApi(url, buildHeaders(apiKey), body).then(sendResponse);
      })
      .catch((err) => sendResponse({ ok: false, error: String(err) }));
    return true;
  }

  if (message?.type === "GRADE") {
    const { apiBase: msgGrader, apiKey: msgKey, text } = message;
    resolveApiRuntimeConfig()
      .then((cfg) => {
        const apiBase =
          trimStr(msgGrader).length > 0 ? trimStr(msgGrader) : cfg.graderApiBase;
        const apiKey = trimStr(msgKey).length > 0 ? trimStr(msgKey) : cfg.apiKey;
        const url = buildUrl(apiBase, "/predict");
        return callApi(url, buildHeaders(apiKey), { text }).then(sendResponse);
      })
      .catch((err) => sendResponse({ ok: false, error: String(err) }));
    return true;
  }
});
