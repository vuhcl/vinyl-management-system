/**
 * Proxies POST /estimate to the VinylIQ FastAPI (avoids CORS from page context).
 */
chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message?.type !== "ESTIMATE") {
    return;
  }
  const { apiBase, apiKey, body } = message;
  const url = `${apiBase.replace(/\/$/, "")}/estimate`;
  const headers = { "Content-Type": "application/json" };
  if (apiKey) {
    headers["X-API-Key"] = apiKey;
  }
  fetch(url, {
    method: "POST",
    headers,
    body: JSON.stringify(body),
  })
    .then(async (r) => {
      const text = await r.text();
      let data;
      try {
        data = JSON.parse(text);
      } catch {
        data = { error: text || r.statusText };
      }
      if (!r.ok) {
        sendResponse({ ok: false, status: r.status, data });
      } else {
        sendResponse({ ok: true, data });
      }
    })
    .catch((err) => {
      sendResponse({ ok: false, error: String(err) });
    });
  return true;
});
