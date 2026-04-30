const CONDITIONS = [
  "Mint (M)",
  "Near Mint (NM or M-)",
  "Very Good Plus (VG+)",
  "Very Good (VG)",
  "Good Plus (G+)",
  "Good (G)",
  "Fair (F)",
  "Poor (P)",
];

function fillSelect(id) {
  const el = document.getElementById(id);
  CONDITIONS.forEach((c) => {
    const o = document.createElement("option");
    o.value = c;
    o.textContent = c;
    el.appendChild(o);
  });
  el.value = "Near Mint (NM or M-)";
}

async function init() {
  fillSelect("media");
  fillSelect("sleeve");

  // 0.2.0 split a single ``apiBase`` (price-only) into ``priceApiBase`` and
  // ``graderApiBase``. If only the legacy key exists, fold it into the new
  // price key so users do not have to re-enter their URL after upgrade.
  const stored = await chrome.storage.sync.get([
    "priceApiBase",
    "graderApiBase",
    "apiBase",
    "apiKey",
  ]);
  const priceApiBase =
    stored.priceApiBase || stored.apiBase || "http://127.0.0.1:8801";
  const graderApiBase = stored.graderApiBase || "http://127.0.0.1:8090";
  const apiKey = stored.apiKey || "";

  document.getElementById("priceApiBase").value = priceApiBase;
  document.getElementById("graderApiBase").value = graderApiBase;
  document.getElementById("apiKey").value = apiKey;

  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  const rid =
    tab?.id != null
      ? await chrome.tabs
          .sendMessage(tab.id, { type: "GET_RELEASE_ID" })
          .catch(() => null)
      : null;
  const releaseId = rid?.releaseId;
  document.getElementById("release-line").textContent = releaseId
    ? `Release: ${releaseId}`
    : "Not on a discogs.com/release/ page";

  document.getElementById("estimate").onclick = async () => {
    const err = document.getElementById("err");
    err.textContent = "";
    if (!releaseId) {
      err.textContent = "Open a release page first.";
      return;
    }
    const priceBase = document.getElementById("priceApiBase").value.trim();
    const graderBase = document.getElementById("graderApiBase").value.trim();
    const key = document.getElementById("apiKey").value.trim();
    await chrome.storage.sync.set({
      priceApiBase: priceBase,
      graderApiBase: graderBase,
      apiKey: key,
    });

    const body = {
      release_id: releaseId,
      media_condition: document.getElementById("media").value,
      sleeve_condition: document.getElementById("sleeve").value,
      refresh_stats: false,
    };

    const resp = await chrome.runtime.sendMessage({
      type: "ESTIMATE",
      apiBase: priceBase,
      apiKey: key,
      body,
    });

    if (!resp?.ok) {
      err.textContent =
        resp?.error ||
        (resp?.data && JSON.stringify(resp.data)) ||
        `HTTP ${resp?.status || "error"}`;
      return;
    }
    if (tab?.id != null) {
      await chrome.tabs.sendMessage(tab.id, {
        type: "SHOW_OVERLAY",
        payload: resp.data,
      });
    }
    window.close();
  };
}

init();
