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
  const { apiBase = "http://127.0.0.1:8801", apiKey = "" } =
    await chrome.storage.sync.get(["apiBase", "apiKey"]);
  document.getElementById("apiBase").value = apiBase;
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
    const base = document.getElementById("apiBase").value.trim();
    const key = document.getElementById("apiKey").value.trim();
    await chrome.storage.sync.set({ apiBase: base, apiKey: key });

    const body = {
      release_id: releaseId,
      media_condition: document.getElementById("media").value,
      sleeve_condition: document.getElementById("sleeve").value,
      refresh_stats: false,
    };

    const resp = await chrome.runtime.sendMessage({
      type: "ESTIMATE",
      apiBase: base,
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
