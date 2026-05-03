/**
 * Release + seller listing overlays and messaging helpers.
 *
 * Depends on listing_dom.js (same isolated world — loaded first in manifest).
 *
 * Chrome MV3 toolbar popups never auto-open on navigation and dismiss when focus
 * leaves them. Seller drafts (/sell/post/*) therefore inject a persistent
 * floating dock with the same Grade + Estimate controls.
 */
function ld() {
  return globalThis.__vinyliqListingDom;
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
    "max-width:420px",
    "min-width:260px",
    "background:#1a1a1a",
    "color:#eee",
    "font-family:system-ui,sans-serif",
    "font-size:16px",
    "line-height:1.35",
    "padding:16px 18px",
    "border-radius:10px",
    "box-shadow:0 4px 28px rgba(0,0,0,0.45)",
    "border:1px solid #333",
  ].join(";");
  const price = payload.estimated_price;
  const lo = payload.confidence_interval?.[0];
  const hi = payload.confidence_interval?.[1];
  /* Deliberately omit baseline_median: API/feature-store value can lag; seller
   * estimates attach fresh ``marketplace_client`` from the page when scraped. */
  wrap.innerHTML = `
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
      <strong style="font-size:17px;">VinylIQ</strong>
      <button type="button" id="vinyliq-close" style="background:#333;border:none;color:#fff;cursor:pointer;padding:4px 10px;border-radius:4px;font-size:15px;">×</button>
    </div>
    <div>Estimate: <strong style="font-size:17px;">$${price != null ? price : "—"}</strong></div>
    <div style="opacity:0.85;font-size:14px;margin-top:6px;">Range: $${lo ?? "—"} – $${hi ?? "—"}</div>
    <div style="opacity:0.6;font-size:12px;margin-top:8px;">${payload.model_version || ""} · ${payload.status || ""}</div>
  `;
  document.body.appendChild(wrap);
  wrap.querySelector("#vinyliq-close").addEventListener("click", removeOverlay);
}

function readConditionCommentTrim(domApi) {
  const ta = domApi.findFirstCommentTextarea?.() ?? null;
  if (!ta || !("value" in ta)) {
    return "";
  }
  return String(ta.value || "").trim();
}

/**
 * @param {ReturnType<typeof ld>} domApi
 * @returns {Promise<{ok:true,mediaOk:boolean,sleeveOk:boolean,detail:string}|{ok:false,message?:string,code?:string}>}
 */
async function runGradeSellerListingAsync(domApi) {
  const ctx = domApi.parseReleaseIdAndSurface();
  if (ctx.surface !== "sell_post") {
    return {
      ok: false,
      code: "not_sell_post",
      message:
        "Open a seller listing draft (/sell/post/…) to grade from condition comments.",
    };
  }
  const text = readConditionCommentTrim(domApi);
  if (!text) {
    return {
      ok: false,
      code: "empty_comment",
      message: "Add a condition comment in the seller form first.",
    };
  }
  let resp;
  try {
    resp = await chrome.runtime.sendMessage({
      type: "GRADE",
      text,
    });
  } catch (err) {
    return { ok: false, message: `Grader error: ${String(err)}` };
  }
  if (!resp?.ok) {
    const detail =
      resp?.error ||
      (resp?.data && JSON.stringify(resp.data)) ||
      `HTTP ${resp?.status || "error"}`;
    return { ok: false, message: `Grader error: ${detail}` };
  }
  const pred = resp.data?.predictions?.[0];
  if (!pred) {
    return { ok: false, message: "Grader returned no prediction." };
  }
  const mediaEl = domApi.findFirstMediaSelect();
  const sleeveEl = domApi.findFirstSleeveSelect();
  const mediaOk = domApi.setConditionSelect(
    mediaEl,
    pred.predicted_media_condition
  );
  const sleeveOk = domApi.setConditionSelect(
    sleeveEl,
    pred.predicted_sleeve_condition
  );
  const conf = (
    ((pred.media_confidence ?? 0) + (pred.sleeve_confidence ?? 0)) /
    2
  ).toFixed(2);
  const hint =
    mediaOk && sleeveOk
      ? `Set Media=${pred.predicted_media_condition} / Sleeve=${pred.predicted_sleeve_condition} (avg conf ${conf}).`
      : `Predicted Media=${pred.predicted_media_condition} / Sleeve=${pred.predicted_sleeve_condition} but could not set one or both dropdowns.`;
  return { ok: true, mediaOk, sleeveOk, detail: hint };
}

/**
 * Collector used by toolbar popup messaging and dock.
 * @returns {Record<string, any>}
 */
function collectListingPayloadSync(domApi) {
  const ctx = domApi.parseReleaseIdAndSurface();
  if (ctx.surface === "release") {
    return {
      ok: false,
      code: "release_page_no_estimate",
      message:
        "Open a seller listing draft (/sell/post/…) to estimate from Discogs grades.",
    };
  }
  const vals = domApi.readMediaSleeveValues();
  if (
    !vals ||
    vals.media.length === 0 ||
    vals.sleeve.length === 0
  ) {
    return {
      ok: false,
      code: "missing_conditions",
      message:
        "Set Media + Sleeve on the seller form (Grade condition, or set grades manually).",
    };
  }
  if (!ctx.releaseId) {
    return {
      ok: false,
      code: "missing_release_id",
      message: "Could not derive release ID from URL.",
    };
  }
  const mpClient = domApi.scrapeMarketplaceClientSnapshot
    ? domApi.scrapeMarketplaceClientSnapshot()
    : null;
  /** @type {Record<string, any>} */
  const payload = {
    ok: true,
    release_id: ctx.releaseId,
    media_condition: vals.media,
    sleeve_condition: vals.sleeve,
  };
  if (mpClient && typeof mpClient === "object") {
    payload.marketplace_client = mpClient;
  }
  return payload;
}

/**
 * @returns {Promise<{ok:true,filledPriceInputs?:boolean}|{ok:false,message?:string,code?:string}>}
 */
async function runEstimateOnPageAsync(domApi) {
  const collected = collectListingPayloadSync(domApi);
  if (!collected.ok) {
    return collected;
  }
  /** @type {Record<string, any>} */
  const body = {
    release_id: collected.release_id,
    media_condition: collected.media_condition,
    sleeve_condition: collected.sleeve_condition,
    refresh_stats: false,
  };
  if (
    collected.marketplace_client &&
    typeof collected.marketplace_client === "object" &&
    Object.keys(collected.marketplace_client).length > 0
  ) {
    body.marketplace_client = collected.marketplace_client;
  }
  let resp;
  try {
    resp = await chrome.runtime.sendMessage({
      type: "ESTIMATE",
      body,
    });
  } catch (err) {
    return { ok: false, message: String(err) };
  }
  if (!resp?.ok) {
    return {
      ok: false,
      message:
        resp?.error ||
        (resp?.data && JSON.stringify(resp.data)) ||
        `HTTP ${resp?.status || "error"}`,
    };
  }
  showOverlay(resp.data);
  const filled = domApi.fillListingSuggestedPrice
    ? domApi.fillListingSuggestedPrice(resp.data)
    : false;
  return { ok: true, filledPriceInputs: Boolean(filled) };
}

/**
 * Persisted VinylIQ dock on seller listing drafts (/sell/post/*).
 * @param {NonNullable<ReturnType<typeof ld>>} domApi
 */
function installSellerListingDock(domApi) {
  if (document.getElementById("vinyliq-sell-dock")) {
    return;
  }

  let gradeBusy = false;

  const root = document.createElement("aside");
  root.id = "vinyliq-sell-dock";
  root.setAttribute("aria-label", "VinylIQ seller listing tools");

  root.style.cssText = [
    "position:fixed",
    "bottom:20px",
    "right:20px",
    "z-index:2147483645",
    "max-width:min(308px,calc(100vw - 40px))",
    "overflow-x:hidden",
    "font-family:system-ui,-apple-system,sans-serif",
    "font-size:14px",
    "line-height:1.4",
    "color:#eee",
    "background:#161616",
    "border:1px solid #3a3a3a",
    "border-radius:12px",
    "box-shadow:0 8px 32px rgba(0,0,0,0.45)",
    "padding:12px 14px",
    "box-sizing:border-box",
  ].join(";");

  const rid = domApi.parseReleaseIdAndSurface().releaseId;

  const hdr = document.createElement("div");
  hdr.style.cssText =
    "display:flex;align-items:flex-start;justify-content:space-between;gap:8px;margin-bottom:8px;min-width:0;";
  const titles = document.createElement("div");
  titles.style.cssText = "min-width:0;flex:1;";
  const tStrong = document.createElement("strong");
  tStrong.style.fontSize = "15px";
  tStrong.textContent = "VinylIQ";
  const tSub = document.createElement("div");
  tSub.style.cssText =
    "font-size:12px;color:#bdbdbd;margin-top:3px;line-height:1.35;word-wrap:break-word;";
  tSub.textContent = `Listing draft · release ${rid ?? "?"}`;
  titles.appendChild(tStrong);
  titles.appendChild(tSub);
  const minBtn = document.createElement("button");
  minBtn.type = "button";
  minBtn.title = "Minimize panel";
  minBtn.textContent = "−";
  minBtn.style.cssText =
    "flex-shrink:0;box-sizing:border-box;background:#2a2a2a;color:#eee;border:1px solid #444;width:28px;height:28px;line-height:1;border-radius:5px;cursor:pointer;padding:0;font-size:15px;";
  hdr.appendChild(titles);
  hdr.appendChild(minBtn);

  const body = document.createElement("div");
  body.dataset.vinyliqDockBody = "1";
  body.style.cssText =
    "min-width:0;width:100%;box-sizing:border-box;";

  const btnBase =
    "display:block;width:100%;max-width:100%;box-sizing:border-box;margin:0;margin-top:6px;padding:7px 10px;border-radius:6px;cursor:pointer;font-size:13px;font-weight:500;line-height:1.25;text-align:center;border-width:1px;border-style:solid;white-space:normal;overflow-wrap:anywhere;";

  const gradeBtn = document.createElement("button");
  gradeBtn.type = "button";
  gradeBtn.textContent = "Grade condition";
  gradeBtn.disabled = true;
  gradeBtn.style.cssText =
    btnBase +
    "background:#253045;color:#eaf0ff;border-color:#3d5675;";
  gradeBtn.style.cursor = "not-allowed";

  const estBtn = document.createElement("button");
  estBtn.type = "button";
  estBtn.textContent = "Get estimate";
  estBtn.style.cssText =
    btnBase +
    "margin-top:7px;background:#1a472a;color:#e8fce8;border-color:#2f6d45;";

  const opts = document.createElement("a");
  opts.href = "#";
  opts.textContent = "Options (API backends & key)";
  opts.style.cssText =
    "display:inline-block;margin-top:8px;font-size:12px;color:#8ab4ff;word-wrap:break-word;max-width:100%;vertical-align:top;";
  opts.addEventListener("click", (ev) => {
    ev.preventDefault();
    chrome.runtime.openOptionsPage();
  });

  const errEl = document.createElement("div");
  errEl.style.cssText =
    "color:#e88;margin-top:8px;font-size:12px;line-height:1.35;white-space:pre-wrap;word-wrap:break-word;max-width:100%;min-height:0;box-sizing:border-box;";

  body.appendChild(gradeBtn);
  body.appendChild(estBtn);
  body.appendChild(opts);
  body.appendChild(errEl);

  root.appendChild(hdr);
  root.appendChild(body);
  document.body.appendChild(root);

  function collapsed() {
    return root.dataset.vinyliqCollapsed === "1";
  }

  function setCollapsed(coll) {
    if (coll) {
      root.dataset.vinyliqCollapsed = "1";
      body.style.display = "none";
      minBtn.textContent = "+";
      minBtn.title = "Expand panel";
    } else {
      delete root.dataset.vinyliqCollapsed;
      body.style.display = "";
      minBtn.textContent = "−";
      minBtn.title = "Minimize panel";
    }
  }

  minBtn.addEventListener("click", () => setCollapsed(!collapsed()));

  function syncGradeGate() {
    if (gradeBusy) {
      gradeBtn.disabled = true;
      gradeBtn.style.opacity = "0.55";
      gradeBtn.style.cursor = "not-allowed";
      return;
    }
    const has =
      domApi.parseReleaseIdAndSurface().surface === "sell_post" &&
      readConditionCommentTrim(domApi).length > 0;
    gradeBtn.disabled = !has;
    gradeBtn.style.opacity = has ? "1" : "0.55";
    gradeBtn.style.cursor = has ? "pointer" : "not-allowed";
  }

  function attachCommentHooks() {
    const ta =
      domApi.findFirstCommentTextarea?.() ??
      ld()?.findFirstCommentTextarea?.() ??
      null;
    if (!(ta instanceof HTMLElement)) {
      return false;
    }
    if (ta.dataset.vinyliqDockBound === "1") {
      return true;
    }
    ta.dataset.vinyliqDockBound = "1";
    ta.addEventListener("input", syncGradeGate);
    ta.addEventListener("change", syncGradeGate);
    return true;
  }

  attachCommentHooks();
  const gateTimer = window.setInterval(() => {
    if (attachCommentHooks()) {
      syncGradeGate();
    }
  }, 450);

  /** Stop polling once dock leaves DOM (SPA rare) */
  const obs = new MutationObserver(() => {
    if (!document.body.contains(root)) {
      window.clearInterval(gateTimer);
      obs.disconnect();
    }
  });
  obs.observe(document.body, { childList: true });

  gradeBtn.addEventListener("click", async () => {
    errEl.textContent = "";
    gradeBusy = true;
    syncGradeGate();
    try {
      const done = await runGradeSellerListingAsync(domApi);
      if (!done.ok) {
        errEl.textContent =
          done.message ||
          done.error ||
          `Grade failed (${done.code || "blocked"}).`;
        return;
      }
      if (!done.mediaOk || !done.sleeveOk) {
        errEl.textContent =
          done.detail ||
          "Grader ran but dropdowns did not accept the predicted grades.";
        return;
      }
    } finally {
      gradeBusy = false;
      syncGradeGate();
    }
  });

  estBtn.addEventListener("click", async () => {
    errEl.textContent = "";
    const resp = await runEstimateOnPageAsync(domApi);
    if (!resp?.ok) {
      errEl.textContent =
        resp?.message ||
        `Cannot estimate (${resp?.code || "blocked"}).`;
      return;
    }
  });

  syncGradeGate();
}

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (!msg?.type) {
    return false;
  }

  const domApi = ld();
  if (!domApi && msg.type !== "SHOW_OVERLAY") {
    sendResponse({ ok: false, error: "listing_dom_missing" });
    return false;
  }

  if (msg.type === "SHOW_OVERLAY") {
    showOverlay(msg.payload || {});
    sendResponse({ ok: true });
    return false;
  }

  if (msg.type === "APPLY_ESTIMATE_UI") {
    const payload = msg.payload || {};
    showOverlay(payload);
    const filled =
      domApi.fillListingSuggestedPrice
        ? domApi.fillListingSuggestedPrice(payload)
        : false;
    sendResponse({ ok: true, filledPriceInputs: Boolean(filled) });
    return false;
  }

  if (msg.type === "GET_RELEASE_ID") {
    sendResponse(domApi.parseReleaseIdAndSurface());
    return false;
  }

  if (msg.type === "GET_GRADE_COMMENT_STATE") {
    const ctx = domApi.parseReleaseIdAndSurface();
    if (ctx.surface !== "sell_post") {
      sendResponse({ ok: false, hasText: false });
      return false;
    }
    const text = readConditionCommentTrim(domApi);
    sendResponse({ ok: true, hasText: text.length > 0 });
    return false;
  }

  if (msg.type === "GRADE_SELLER_LISTING") {
    (async () => {
      const result = await runGradeSellerListingAsync(domApi);
      sendResponse(result);
    })();
    return true;
  }

  if (msg.type === "COLLECT_LISTING_CONDITIONS") {
    sendResponse(collectListingPayloadSync(domApi));
    return false;
  }

  return false;
});

let dockInstallObserver = null;

function disconnectDockObserver() {
  if (dockInstallObserver) {
    dockInstallObserver.disconnect();
    dockInstallObserver = null;
  }
}

function tryInstallSellerDock() {
  if (document.getElementById("vinyliq-sell-dock")) {
    disconnectDockObserver();
    return;
  }
  const domApi = ld();
  if (!domApi) {
    return;
  }
  if (domApi.parseReleaseIdAndSurface().surface !== "sell_post") {
    return;
  }
  installSellerListingDock(domApi);
  disconnectDockObserver();
}

dockInstallObserver = new MutationObserver(tryInstallSellerDock);
dockInstallObserver.observe(document.documentElement, {
  childList: true,
  subtree: true,
});
tryInstallSellerDock();
