/**
 * VinylIQ extension service worker.
 *
 * Proxies API calls from page-context content scripts to the VinylIQ
 * FastAPI backends. The MV3 service worker bypasses page CORS and uses
 * host_permissions declared in manifest.json (discogs.com, localhost,
 * 127.0.0.1, *.nip.io) so requests succeed without server-side CORS.
 *
 * Supported message types:
 *   - "ESTIMATE": POST /estimate on the price API
 *   - "GRADE":    POST /predict  on the grader API
 *
 * Both messages share the same response envelope: ``{ ok: bool, data?,
 * status?, error? }`` to keep callers symmetric.
 */

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
    const { apiBase, apiKey, body } = message;
    const url = buildUrl(apiBase, "/estimate");
    callApi(url, buildHeaders(apiKey), body)
      .then(sendResponse)
      .catch((err) => sendResponse({ ok: false, error: String(err) }));
    return true;
  }

  if (message?.type === "GRADE") {
    const { apiBase, apiKey, text } = message;
    const url = buildUrl(apiBase, "/predict");
    callApi(url, buildHeaders(apiKey), { text })
      .then(sendResponse)
      .catch((err) => sendResponse({ ok: false, error: String(err) }));
    return true;
  }
});
