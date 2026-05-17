async function init() {
  const errEl = document.getElementById("err");
  const releaseLineEl = document.getElementById("release-line");
  const gradeBtn = document.getElementById("grade");
  /** While a grade request runs, keep Grade disabled regardless of textarea text. */
  let gradeBusy = false;

  document.getElementById("open-options").addEventListener("click", (ev) => {
    ev.preventDefault();
    chrome.runtime.openOptionsPage();
  });

  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  const rid =
    tab?.id != null
      ? await chrome.tabs.sendMessage(tab.id, { type: "GET_RELEASE_ID" }).catch(() => null)
      : null;

  const releaseId = rid?.releaseId;
  const surface = rid?.surface;

  releaseLineEl.textContent =
    "Open a VinylIQ-supported Discogs tab (seller listing or release page).";

  async function syncGradeGate() {
    if (surface !== "sell_post" || !tab?.id) {
      return;
    }
    if (gradeBusy) {
      gradeBtn.disabled = true;
      return;
    }
    const st = await chrome.tabs.sendMessage(tab.id, {
      type: "GET_GRADE_COMMENT_STATE",
    }).catch(() => ({ ok: false, hasText: false }));
    gradeBtn.disabled = !(st?.ok && st.hasText);
  }

  if (releaseId && surface === "sell_post") {
    releaseLineEl.textContent = `Selling page — listing draft for release ${releaseId}`;
    gradeBtn.classList.remove("hidden");
    gradeBtn.disabled = true;
    await syncGradeGate();
    setInterval(syncGradeGate, 400);
  } else if (releaseId && surface === "release") {
    releaseLineEl.textContent = `Release catalogue page — release ${releaseId} (Estimate needs /sell/post/…)`;
  }

  gradeBtn.onclick = async () => {
    errEl.textContent = "";
    if (!tab?.id) {
      errEl.textContent = "No active tab.";
      return;
    }
    gradeBusy = true;
    gradeBtn.disabled = true;
    let done;
    try {
      done = await chrome.tabs
        .sendMessage(tab.id, { type: "GRADE_SELLER_LISTING" })
        .catch((e) => ({ ok: false, message: String(e) }));
    } finally {
      gradeBusy = false;
      await syncGradeGate();
    }

    if (!done?.ok) {
      errEl.textContent =
        done?.message ||
        done?.error ||
        `Grade failed (${done?.code || "blocked"}).`;
      return;
    }
    if (!done.mediaOk || !done.sleeveOk) {
      errEl.textContent =
        done.detail ||
        "Grader ran but dropdowns did not accept the predicted grades.";
      return;
    }
  };

  document.getElementById("estimate").onclick = async () => {
    errEl.textContent = "";
    if (!tab?.id) {
      errEl.textContent = "No active tab.";
      return;
    }
    const collected = await chrome.tabs
      .sendMessage(tab.id, { type: "COLLECT_LISTING_CONDITIONS" })
      .catch((e) => ({ ok: false, message: String(e) }));

    if (!collected?.ok) {
      errEl.textContent =
        collected?.message ||
        `Cannot estimate (${collected?.code || "blocked"}): open a seller draft with Media + Sleeve set.`;
      return;
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
    const resp = await chrome.runtime.sendMessage({
      type: "ESTIMATE",
      body,
    });

    if (!resp?.ok) {
      errEl.textContent =
        resp?.error ||
        (resp?.data && JSON.stringify(resp.data)) ||
        `HTTP ${resp?.status || "error"}`;
      return;
    }

    await chrome.tabs.sendMessage(tab.id, {
      type: "APPLY_ESTIMATE_UI",
      payload: resp.data,
    });
    window.close();
  };
}

init();
