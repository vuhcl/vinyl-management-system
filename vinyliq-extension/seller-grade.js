/**
 * Seller-listing condition grader (Discogs /sell/post/*).
 *
 * Injects a "Grade condition" button next to the listing form's comments
 * textarea. On click:
 *   1. Reads comment text.
 *   2. Forwards it to the grader API via the service worker (POST /predict).
 *   3. Sets the Media + Sleeve <select> values to the predicted Discogs
 *      condition strings and dispatches a bubbling 'change' event so
 *      React-rendered listing forms pick up the new values.
 *
 * Robustness notes:
 *   - Discogs' listing form is a React SPA; setting ``select.value``
 *     directly does not always propagate to React state. We use the
 *     standard native-setter + bubbled 'change' workaround.
 *   - Selector lists are intentionally redundant because Discogs has
 *     periodically renamed form fields. The first match wins.
 *   - The injection is idempotent (guarded by a unique button id) and
 *     a MutationObserver re-runs the wiring if the form is re-rendered.
 */
(function () {
  "use strict";

  const BUTTON_ID = "vinyliq-grade-btn";
  const STATUS_ID = "vinyliq-grade-status";

  const COMMENT_SELECTORS = [
    'textarea[name="comments"]',
    'textarea[id*="comment" i]',
    'textarea[name="release_comments"]',
    'textarea[name="description"]',
  ];
  const MEDIA_SELECTORS = [
    'select[name="condition"]',
    'select#condition',
    'select[name="media_condition"]',
    'select[id*="media" i]',
  ];
  const SLEEVE_SELECTORS = [
    'select[name="sleeve_condition"]',
    'select#sleeve_condition',
    'select[id*="sleeve" i]',
  ];

  function findFirst(selectors) {
    for (const sel of selectors) {
      const el = document.querySelector(sel);
      if (el) {
        return el;
      }
    }
    return null;
  }

  function setSelectValue(select, value) {
    if (!select) {
      return false;
    }
    const opts = Array.from(select.options || []);
    const match = opts.find(
      (o) => o.value === value || o.textContent.trim() === value
    );
    if (!match) {
      return false;
    }
    const nativeSetter = Object.getOwnPropertyDescriptor(
      HTMLSelectElement.prototype,
      "value"
    ).set;
    nativeSetter.call(select, match.value);
    select.dispatchEvent(new Event("change", { bubbles: true }));
    select.dispatchEvent(new Event("input", { bubbles: true }));
    return true;
  }

  function ensureStatusEl(after) {
    let el = document.getElementById(STATUS_ID);
    if (el) {
      return el;
    }
    el = document.createElement("div");
    el.id = STATUS_ID;
    el.style.cssText = [
      "font-size:12px",
      "margin-top:6px",
      "color:#555",
      "min-height:1em",
    ].join(";");
    after.parentNode.insertBefore(el, after.nextSibling);
    return el;
  }

  function setStatus(textareaEl, text, kind = "info") {
    const el = ensureStatusEl(textareaEl);
    el.textContent = text;
    el.style.color = kind === "error" ? "#b00" : "#555";
  }

  function buildButton(textareaEl) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.id = BUTTON_ID;
    btn.textContent = "Grade condition";
    btn.style.cssText = [
      "margin-top:8px",
      "padding:6px 12px",
      "border:1px solid #444",
      "background:#1a1a1a",
      "color:#fff",
      "border-radius:4px",
      "cursor:pointer",
      "font-size:13px",
    ].join(";");
    btn.addEventListener("click", async (ev) => {
      ev.preventDefault();
      const text = (textareaEl.value || "").trim();
      if (!text) {
        setStatus(textareaEl, "Add a condition comment first.", "error");
        return;
      }
      btn.disabled = true;
      setStatus(textareaEl, "Calling grader...", "info");
      const stored = await chrome.storage.sync.get([
        "graderApiBase",
        "apiKey",
      ]);
      // 0.2.0 introduced ``graderApiBase`` (price-only ``apiBase`` is
      // intentionally NOT used as a fallback here — it points at the
      // wrong service). Local uvicorn default keeps demos working
      // without first opening the popup.
      const graderBase =
        (stored.graderApiBase && String(stored.graderApiBase).trim()) ||
        "http://127.0.0.1:8090";
      const apiKey = stored.apiKey || "";
      let resp;
      try {
        resp = await chrome.runtime.sendMessage({
          type: "GRADE",
          apiBase: graderBase,
          apiKey,
          text,
        });
      } catch (err) {
        setStatus(textareaEl, `Grader error: ${String(err)}`, "error");
        btn.disabled = false;
        return;
      }
      if (!resp?.ok) {
        const detail =
          resp?.error ||
          (resp?.data && JSON.stringify(resp.data)) ||
          `HTTP ${resp?.status || "error"}`;
        setStatus(textareaEl, `Grader error: ${detail}`, "error");
        btn.disabled = false;
        return;
      }
      const pred = resp.data?.predictions?.[0];
      if (!pred) {
        setStatus(textareaEl, "Grader returned no prediction.", "error");
        btn.disabled = false;
        return;
      }
      const mediaEl = findFirst(MEDIA_SELECTORS);
      const sleeveEl = findFirst(SLEEVE_SELECTORS);
      const mediaOk = setSelectValue(mediaEl, pred.predicted_media_condition);
      const sleeveOk = setSelectValue(
        sleeveEl,
        pred.predicted_sleeve_condition
      );
      const conf = (
        ((pred.media_confidence ?? 0) + (pred.sleeve_confidence ?? 0)) /
        2
      ).toFixed(2);
      if (mediaOk && sleeveOk) {
        setStatus(
          textareaEl,
          `Set Media=${pred.predicted_media_condition} / Sleeve=${pred.predicted_sleeve_condition} (avg conf ${conf}).`,
          "info"
        );
      } else {
        setStatus(
          textareaEl,
          `Predicted Media=${pred.predicted_media_condition} / Sleeve=${pred.predicted_sleeve_condition} but could not locate one or both <select> elements.`,
          "error"
        );
      }
      btn.disabled = false;
    });
    return btn;
  }

  function inject() {
    if (document.getElementById(BUTTON_ID)) {
      return true;
    }
    const textareaEl = findFirst(COMMENT_SELECTORS);
    if (!textareaEl) {
      return false;
    }
    const btn = buildButton(textareaEl);
    textareaEl.insertAdjacentElement("afterend", btn);
    ensureStatusEl(btn);
    return true;
  }

  if (!inject()) {
    const obs = new MutationObserver(() => {
      if (inject()) {
        obs.disconnect();
      }
    });
    obs.observe(document.documentElement, {
      childList: true,
      subtree: true,
    });
    setTimeout(() => obs.disconnect(), 30_000);
  }
})();
