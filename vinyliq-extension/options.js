function trim(v) {
  return v != null ? String(v).trim() : "";
}

async function load() {
  const { priceApiBase, graderApiBase, apiKey, apiBase } =
    await chrome.storage.sync.get([
      "priceApiBase",
      "graderApiBase",
      "apiKey",
      "apiBase",
    ]);
  const price = trim(priceApiBase) || trim(apiBase);
  document.getElementById("priceApiBase").value = price || "";
  document.getElementById("graderApiBase").value = trim(graderApiBase) || "";
  document.getElementById("apiKey").value = trim(apiKey) || "";
}

async function save() {
  const status = document.getElementById("status");
  status.textContent = "";
  const priceApiBase = trim(document.getElementById("priceApiBase").value);
  const graderApiBase = trim(document.getElementById("graderApiBase").value);
  const apiKey = trim(document.getElementById("apiKey").value);
  await chrome.storage.sync.set({
    priceApiBase: priceApiBase || "",
    graderApiBase: graderApiBase || "",
    apiKey: apiKey || "",
  });
  status.textContent = "Saved.";
}

void load();

document.getElementById("save").addEventListener("click", () =>
  save().catch(console.error),
);
