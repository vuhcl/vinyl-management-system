/**
 * Parse release ID from URL and render estimate overlay when asked.
 */
function parseReleaseId() {
  const m = window.location.pathname.match(/\/release\/(\d+)/);
  return m ? m[1] : null;
}

function removeOverlay() {
  const el = document.getElementById("vinyliq-overlay");
  if (el) {
    el.remove();
  }
}

function showOverlay(payload) {
  removeOverlay();
  const wrap = document.createElement("div");
  wrap.id = "vinyliq-overlay";
  wrap.style.cssText = [
    "position:fixed",
    "top:16px",
    "right:16px",
    "z-index:2147483646",
    "max-width:320px",
    "background:#1a1a1a",
    "color:#eee",
    "font-family:system-ui,sans-serif",
    "font-size:14px",
    "padding:12px 14px",
    "border-radius:8px",
    "box-shadow:0 4px 24px rgba(0,0,0,0.4)",
    "border:1px solid #333",
  ].join(";");
  const price = payload.estimated_price;
  const lo = payload.confidence_interval?.[0];
  const hi = payload.confidence_interval?.[1];
  const base = payload.baseline_median;
  wrap.innerHTML = `
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
      <strong>VinylIQ</strong>
      <button type="button" id="vinyliq-close" style="background:#333;border:none;color:#fff;cursor:pointer;padding:2px 8px;border-radius:4px;">×</button>
    </div>
    <div>Estimate: <strong>$${price != null ? price : "—"}</strong></div>
    <div style="opacity:0.85;font-size:12px;margin-top:4px;">Range: $${lo ?? "—"} – $${hi ?? "—"}</div>
    <div style="opacity:0.75;font-size:12px;margin-top:4px;">Discogs median: $${base ?? "—"}</div>
    <div style="opacity:0.6;font-size:11px;margin-top:6px;">${payload.model_version || ""} · ${payload.status || ""}</div>
  `;
  document.body.appendChild(wrap);
  wrap.querySelector("#vinyliq-close").addEventListener("click", removeOverlay);
}

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (msg?.type === "SHOW_OVERLAY") {
    showOverlay(msg.payload || {});
    sendResponse({ ok: true });
  }
  if (msg?.type === "GET_RELEASE_ID") {
    sendResponse({ releaseId: parseReleaseId() });
  }
  return true;
});
