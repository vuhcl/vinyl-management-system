/**
 * Shared Discogs listing DOM probes (seller draft /sell/post/*).
 * SYNC with Playwright locator strings in demo/vinyliq_demo_playwright/tests/.
 */
(function attachVinylIqListingDom() {
  "use strict";

  const COMMENT_SELECTORS = [
    "#comments",
    'textarea[name="comments"]',
    'input[name="comments"]',
    'input[id="comments"]',
    'textarea[id*="comment" i]',
    'textarea[name="release_comments"]',
    'textarea[name="description"]',
  ];
  const MEDIA_SELECTORS = [
    "select#media_condition",
    'select[name="media_condition"]',
    'select[name="condition"]',
    "select#condition",
    'select[id*="media" i]',
  ];
  const SLEEVE_SELECTORS = [
    "select#sleeve_condition",
    'select[name="sleeve_condition"]',
    'select[id*="sleeve" i]',
  ];
  /** Heuristic — refresh when Discogs DOM changes */
  const PRICE_INPUT_SELECTORS = [
    'input[name="price"]',
    "#price",
    'input[id*="price" i]',
    'input[type="number"][name*="price" i]',
  ];

  /** Reject bogus scraped prices depth / marketplace UI glitches */
  const MAX_PRICE_USD = 1e6;

  function findFirst(selectors) {
    for (const sel of selectors) {
      const el = document.querySelector(sel);
      if (el) {
        return el;
      }
    }
    return null;
  }

  /** Skip hidden/template ``<select>`` duplicates (Discogs layouts often duplicate controls). */
  function isLikelyRenderableSelect(sel) {
    if (!(sel instanceof HTMLSelectElement) || sel.disabled) {
      return false;
    }
    try {
      if (typeof sel.checkVisibility === "function") {
        return sel.checkVisibility({
          opacity: true,
          visibilityProperty: true,
        });
      }
    } catch {
      /* fall through */
    }
    try {
      if (!sel.ownerDocument.documentElement.contains(sel)) {
        return false;
      }
      const st = window.getComputedStyle(sel);
      if (
        st.display === "none" ||
        st.visibility === "hidden" ||
        Number.parseFloat(st.opacity || "1") <= 0
      ) {
        return false;
      }
      const r = sel.getBoundingClientRect();
      return (
        typeof r.width === "number" &&
        typeof r.height === "number" &&
        r.width >= 10 &&
        r.height >= 8
      );
    } catch {
      return false;
    }
  }

  /**
   * First **visible** media/sleeve control with a populated ladder.
   * SYNC: Playwright demos use visible-first locators alongside this.
   */
  function findFirstUsableSelect(selectorList, /** @type {{ minOpts?: number }} */ optsIn) {
    const minOpts =
      optsIn?.minOpts != null && Number.isFinite(optsIn.minOpts) ? optsIn.minOpts : 2;
    for (const sel of selectorList) {
      /** @type {NodeListOf<Element>|undefined} */
      let list;
      try {
        list = document.querySelectorAll(sel);
      } catch {
        list = undefined;
      }
      if (!list) {
        continue;
      }
      for (const el of list) {
        if (!(el instanceof HTMLSelectElement)) {
          continue;
        }
        const nOpts = el.options ? el.options.length : 0;
        if (
          isLikelyRenderableSelect(el) &&
          nOpts >= Math.max(2, Math.floor(minOpts))
        ) {
          return el;
        }
      }
    }
    return null;
  }

  function parseReleaseIdAndSurface() {
    const path = window.location.pathname;
    let m = path.match(/^\/sell\/post\/(\d+)/);
    if (m) {
      return { releaseId: m[1], surface: "sell_post" };
    }
    m = path.match(/\/release\/(\d+)/);
    if (m) {
      return { releaseId: m[1], surface: "release" };
    }
    return { releaseId: null, surface: null };
  }

  /**
   * Prefer visible option label (canonical Discogs ladder wording, e.g.
   * "Good (G)") so ladders from ``price_suggestions`` map reliably; fallback
   * to ``select.value``.
   */
  function selectedConditionLabel(sel) {
    if (!sel || sel.tagName !== "SELECT") {
      return "";
    }
    try {
      const idx = sel.selectedIndex;
      if (idx >= 0 && sel.options && sel.options[idx]) {
        const t = String(sel.options[idx].textContent || "").trim();
        if (t) {
          return t;
        }
      }
    } catch {
      /* ignore */
    }
    return String(sel.value || "").trim();
  }

  function readMediaSleeveValues() {
    const mediaEl = findFirstUsableSelect(MEDIA_SELECTORS);
    const sleeveEl = findFirstUsableSelect(SLEEVE_SELECTORS);
    const media = mediaEl ? selectedConditionLabel(mediaEl) : "";
    const sleeve = sleeveEl ? selectedConditionLabel(sleeveEl) : "";
    if (!media || !sleeve) {
      return null;
    }
    return { media, sleeve };
  }

  const GRADE_LADDER_KEY_RE =
    /\b(Mint \(M\)|Near Mint \(NM or M-\)|Very Good Plus \(VG\+\)|Very Good \(VG\)|Good Plus \(G\+\)|Good \(G\)|Fair \(F\)|Poor \(P\))\b/i;

  function ladderEntryLooksValid(v) {
    if (!v || typeof v !== "object") {
      return false;
    }
    const raw = v.value != null ? v.value : v.Value;
    const n =
      typeof raw === "number" ? raw : typeof raw === "string" ? parseFloat(raw) : NaN;
    return Number.isFinite(n) && n > 0 && n < MAX_PRICE_USD;
  }

  function looksLikePriceSuggestionsLadder(obj) {
    if (!obj || typeof obj !== "object" || Array.isArray(obj)) {
      return false;
    }
    const keys = Object.keys(obj);
    if (keys.length < 6) {
      return false;
    }
    let gradeKeys = 0;
    let valueShape = 0;
    for (const k of keys) {
      if (GRADE_LADDER_KEY_RE.test(k)) {
        gradeKeys += 1;
      }
      if (ladderEntryLooksValid(obj[k])) {
        valueShape += 1;
      }
    }
    return gradeKeys >= 6 && valueShape >= 6;
  }

  function normalizePriceSuggestionsLadder(raw) {
    const out = {};
    if (!raw || typeof raw !== "object" || Array.isArray(raw)) {
      return out;
    }
    for (const [k, v] of Object.entries(raw)) {
      if (!GRADE_LADDER_KEY_RE.test(String(k))) {
        continue;
      }
      if (!ladderEntryLooksValid(v)) {
        continue;
      }
      const rawVal = v.value != null ? v.value : v.Value;
      const val =
        typeof rawVal === "number" ? rawVal : parseFloat(String(rawVal));
      const currency =
        typeof v.currency === "string" && v.currency.trim()
          ? String(v.currency).trim()
          : "USD";
      if (!Number.isFinite(val)) {
        continue;
      }
      out[String(k).trim()] = { value: Math.round(val * 1e8) / 1e8, currency };
    }
    return Object.keys(out).length >= 6 ? out : {};
  }

  function findNestedPriceSuggestions(root) {
    if (root == null) {
      return null;
    }
    const visited = typeof WeakSet === "undefined" ? new Set() : new WeakSet();
    const KEY_HINTS =
      /\b(price_suggestions|priceSuggestions|PriceSuggestions)\b/;

    function walk(node, depth) {
      if (node == null || depth > 18) {
        return null;
      }
      const t = typeof node;
      if (t !== "object") {
        return null;
      }
      if (visited.has(node)) {
        return null;
      }
      try {
        visited.add(node);
      } catch {
        /* cross-realm ignore */
      }
      if (Array.isArray(node)) {
        for (const el of node) {
          const r = walk(el, depth + 1);
          if (r) {
            return r;
          }
        }
        return null;
      }

      if (looksLikePriceSuggestionsLadder(node)) {
        return node;
      }
      let hintBoost = false;
      for (const k of Object.keys(node)) {
        if (KEY_HINTS.test(k)) {
          hintBoost = true;
          break;
        }
      }
      for (const k of Object.keys(node)) {
        if (hintBoost || KEY_HINTS.test(k)) {
          const r = walk(node[k], depth + 1);
          if (r) {
            return r;
          }
        }
      }
      for (const k of Object.keys(node)) {
        if (KEY_HINTS.test(k)) {
          continue;
        }
        const r = walk(node[k], depth + 1);
        if (r) {
          return r;
        }
      }
      return null;
    }

    const ladder = walk(root, 0);
    if (!ladder) {
      return null;
    }
    const norm = normalizePriceSuggestionsLadder(ladder);
    return Object.keys(norm).length ? norm : null;
  }

  function parseUsdFromText(snippet) {
    const m =
      /\$\s*([\d]{1,3}(?:,[0-9]{3})+(?:\.[0-9]+)?|[0-9]+(?:\.[0-9]{1,2})?)/.exec(
        snippet,
      );
    if (!m) {
      return null;
    }
    const n = parseFloat(m[1].replace(/,/g, ""));
    return Number.isFinite(n) && n >= 0.01 && n < MAX_PRICE_USD ? n : null;
  }

  /**
   * Discogs renders last-sold quartet as sibling ``$ USD`` text + ``<small>Label</small>`` per ``li``.
   * Example markup: ``ul.list_no_style.inline.no_width_limit > li``.
   * @returns {Record<string, number>}
   */
  function scrapeLastSoldStatsUlFromDom() {
    /** @type {Record<string, number>} */
    const found = {};

    /** @param {HTMLElement} ul */
    function harvestUl(ul) {
      ul.querySelectorAll(":scope > li").forEach((li) => {
        const small = li.querySelector("small");
        if (!small || small.closest("li") !== li) {
          return;
        }
        const lab = String(small.textContent || "")
          .trim()
          .toLowerCase();
        /** @type {string | null} */
        let apiKey = null;
        if (lab === "average") {
          apiKey = "sale_stats_average_usd";
        } else if (lab === "median") {
          apiKey = "sale_stats_median_usd";
        } else if (lab === "high") {
          apiKey = "sale_stats_high_usd";
        } else if (lab === "low") {
          apiKey = "sale_stats_low_usd";
        } else {
          return;
        }
        if (found[apiKey] != null) {
          return;
        }
        const usd = parseUsdFromText(li.textContent || "");
        if (usd != null && usd >= 0.01 && usd < MAX_PRICE_USD) {
          found[apiKey] = usd;
        }
      });
    }

    document.querySelectorAll("ul.list_no_style.inline").forEach((ul) => {
      if (!(ul instanceof HTMLElement)) {
        return;
      }
      harvestUl(ul);
    });

    return found;
  }

  /**
   * Pull scalars visible on Discogs marketplace / seller surfaces.
   * Classic ``/sell/post`` uses ``#page_content`` with “N want this / have this” copy;
   * keep scope narrow to avoid footer/header noise.
   */
  function scrapeMarketplaceScalarsFromDom() {
    const root =
      document.getElementById("page_content") ||
      document.querySelector(".release_preview") ||
      document.querySelector('[class*="marketplace" i]') ||
      document.querySelector('[class*="Marketplace" i]') ||
      document.querySelector("main") ||
      document.body;
    const t = root && root.innerText ? String(root.innerText) : "";
    if (!t) {
      return {};
    }

    /** @type {Record<string, number>} */
    const out = {};

    const lowM = t.match(
      /\b(lowest|from|starting\s+at)\b[^$\d]{0,40}\$/i,
    );
    if (lowM) {
      const after = t.slice(lowM.index ?? 0, (lowM.index ?? 0) + 72);
      const usd = parseUsdFromText(after);
      if (usd != null) {
        out.release_lowest_price = usd;
      }
    }

    const nfsM =
      /\b([\d,]+)\b\s+for\s+sale\b/i.exec(t) ||
      /\bfor\s+sale[^0-9]{0,24}([\d,]+)\b/i.exec(t);
    if (nfsM) {
      const n = parseInt(String(nfsM[1]).replace(/,/g, ""), 10);
      if (Number.isFinite(n) && n >= 0 && n < 5e6) {
        out.num_for_sale = n;
        out.release_num_for_sale = n;
      }
    }

    const parseCount = (s) =>
      parseInt(String(s || "").replace(/,/g, ""), 10);

    const wantSell = /\b([\d,]+)\s+want\s+this\b/i.exec(t);
    const haveSell = /\b([\d,]+)\s+have\s+this\b/i.exec(t);
    let wantIdx = wantSell ? parseCount(wantSell[1]) : NaN;
    let haveIdx = haveSell ? parseCount(haveSell[1]) : NaN;

    /* Release page / alternate copy */
    const wantLegacy = /\b([\d,]+)\s+in\s+wantlist\b/i.exec(t);
    const haveLegacy = /\b([\d,]+)\s+in\s+collection\b/i.exec(t);

    const wantAlt = /\b([\d,]+)\s+wants?\b/i.exec(t);
    const haveAlt = /\b([\d,]+)\s+haves?\b/i.exec(t);

    if (!Number.isFinite(wantIdx) || wantIdx < 0) {
      if (wantLegacy) {
        wantIdx = parseCount(wantLegacy[1]);
      } else if (wantAlt) {
        wantIdx = parseCount(wantAlt[1]);
      }
    }
    if (!Number.isFinite(haveIdx) || haveIdx < 0) {
      if (haveLegacy) {
        haveIdx = parseCount(haveLegacy[1]);
      } else if (haveAlt) {
        haveIdx = parseCount(haveAlt[1]);
      }
    }

    if (Number.isFinite(wantIdx) && wantIdx >= 0 && wantIdx < 2e9) {
      out.community_want = wantIdx;
    }
    if (Number.isFinite(haveIdx) && haveIdx >= 0 && haveIdx < 2e9) {
      out.community_have = haveIdx;
    }

    Object.assign(out, scrapeLastSoldStatsUlFromDom());
    return out;
  }

  function parseInlineJsonSnapshots() {
    /** @type {any[]} */
    const roots = [];
    const nx = document.getElementById("__NEXT_DATA__");
    if (nx && nx.textContent) {
      try {
        roots.push(JSON.parse(nx.textContent));
      } catch {
        /* ignore */
      }
    }
    /**
     * Big inline JSON payloads (often include marketplace blocks).
     * Skip tiny blobs and telemetry.
     */
    for (const s of document.querySelectorAll('script:not([src])')) {
      const raw = String(s.textContent || "").trim();
      if (
        raw.length < 240 ||
        !raw.startsWith("{") ||
        raw.indexOf("Mint") === -1
      ) {
        continue;
      }
      if (raw.indexOf("price") === -1 && raw.indexOf("Price") === -1) {
        continue;
      }
      try {
        roots.push(JSON.parse(raw));
      } catch {
        /* ignore */
      }
    }

    /** @type {Record<string, number>} */
    const mergedScalars = {};

    /** @param {number | string | null | undefined} raw */
    function asFiniteUsd(raw, min = 0.01, max = MAX_PRICE_USD - 1) {
      let n =
        typeof raw === "number" ? raw : parseFloat(String(raw || "").trim());
      if (!Number.isFinite(n)) {
        const s = String(raw || "").trim().replace(/,/g, "");
        n = parseFloat(s);
      }
      if (
        typeof n !== "number" ||
        !Number.isFinite(n) ||
        n < min ||
        n >= max
      ) {
        return null;
      }
      return n;
    }

    /** @param {number | string | null | undefined} raw */
    function asNonnegInt(raw, ceiling) {
      const n =
        typeof raw === "number"
          ? Math.floor(raw)
          : parseInt(String(raw ?? "").trim().replace(/,/g, ""), 10);
      if (
        typeof n !== "number" ||
        !Number.isFinite(n) ||
        n < 0 ||
        n > ceiling
      ) {
        return null;
      }
      return Math.floor(n);
    }

    function tryHarvestScalars(blob) {
      if (!blob || typeof blob !== "object" || Array.isArray(blob)) {
        return;
      }
      const lp = asFiniteUsd(
        blob.release_lowest_price ??
          blob.releaseLowestPrice ??
          blob.lowest_price ??
          blob.lowestPrice,
      );
      if (lp != null && mergedScalars.release_lowest_price == null) {
        mergedScalars.release_lowest_price = lp;
      }
      const nfs = asNonnegInt(blob.num_for_sale ?? blob.numForSale, 5e6);
      if (nfs != null && mergedScalars.num_for_sale == null) {
        mergedScalars.num_for_sale = nfs;
      }
      const nfs2 = asNonnegInt(blob.for_sale_count, 5e6);
      if (nfs2 != null && mergedScalars.num_for_sale == null) {
        mergedScalars.num_for_sale = nfs2;
      }
      const rns = asNonnegInt(
        blob.release_num_for_sale ?? blob.releaseNumForSale,
        5e6,
      );
      if (rns != null && mergedScalars.release_num_for_sale == null) {
        mergedScalars.release_num_for_sale = rns;
      }
      const cw = asNonnegInt(blob.community_want ?? blob.communityWant, 2e9);
      if (cw != null && mergedScalars.community_want == null) {
        mergedScalars.community_want = cw;
      }
      const ch = asNonnegInt(blob.community_have ?? blob.communityHave, 2e9);
      if (ch != null && mergedScalars.community_have == null) {
        mergedScalars.community_have = ch;
      }
    }

    function boundedWalkForScalars(root, maxSteps) {
      const seen =
        typeof WeakSet === "undefined" ? new Set() : new WeakSet();
      const stack = [root];
      let steps = 0;
      while (stack.length && steps < maxSteps) {
        steps += 1;
        const node = stack.pop();
        if (node == null || typeof node !== "object") {
          continue;
        }
        if (seen.has(node)) {
          continue;
        }
        try {
          seen.add(node);
        } catch {
          continue;
        }
        tryHarvestScalars(node);
        if (Array.isArray(node)) {
          const cap = Math.min(node.length, 400);
          for (let i = 0; i < cap; i += 1) {
            stack.push(node[i]);
          }
          continue;
        }
        const keys = Object.keys(node);
        for (let i = 0; i < Math.min(keys.length, 120); i += 1) {
          stack.push(node[keys[i]]);
        }
      }
    }

    /** @type {Record<string, { value: number, currency: string }> | null} */
    let ladder = null;
    for (const r of roots) {
      ladder = ladder || findNestedPriceSuggestions(r);
      boundedWalkForScalars(r, 65000);
    }

    return { ladder, mergedScalars };
  }

  /**
   * Build optional ``marketplace_client`` overlay for VinylIQ `/estimate`.
   * Fills ``sale_stats_*_usd`` from ``ul.list_no_style.inline`` when Discogs renders
   * each stat as USD text plus ``small`` Average / Median / High / Low.
   * Returns null when nothing trustworthy was found on the document.
   */
  function scrapeMarketplaceClientSnapshot() {
    let { ladder, mergedScalars } = parseInlineJsonSnapshots();
    const domScalars = scrapeMarketplaceScalarsFromDom();
    mergedScalars = { ...domScalars, ...mergedScalars };
    ladder = ladder || null;

    if (
      mergedScalars.release_lowest_price == null ||
      !(mergedScalars.release_lowest_price > 0)
    ) {
      delete mergedScalars.release_lowest_price;
    }

    const out = {};

    const assignInt = (dst, v, max) => {
      if (v != null && Number.isFinite(v) && v >= 0 && v < max) {
        out[dst] = Math.floor(v);
      }
    };

    assignInt("community_want", mergedScalars.community_want, 2e9);
    assignInt("community_have", mergedScalars.community_have, 2e9);
    assignInt("num_for_sale", mergedScalars.num_for_sale, 5e6);
    assignInt(
      "release_num_for_sale",
      mergedScalars.release_num_for_sale ?? mergedScalars.num_for_sale,
      5e6,
    );
    const lp =
      mergedScalars.release_lowest_price != null &&
      typeof mergedScalars.release_lowest_price === "number"
        ? mergedScalars.release_lowest_price
        : null;
    if (lp != null && lp >= 0.01 && lp < MAX_PRICE_USD) {
      out.release_lowest_price = Math.round(lp * 1e4) / 1e4;
    }

    if (ladder && Object.keys(ladder).length >= 6) {
      out.price_suggestions_json = ladder;
    }

    const assignUsdSaleStat = (dst, v) => {
      if (
        v != null &&
        typeof v === "number" &&
        Number.isFinite(v) &&
        v >= 0.01 &&
        v < MAX_PRICE_USD
      ) {
        out[dst] = Math.round(v * 1e4) / 1e4;
      }
    };

    assignUsdSaleStat(
      "sale_stats_average_usd",
      mergedScalars.sale_stats_average_usd,
    );
    assignUsdSaleStat(
      "sale_stats_median_usd",
      mergedScalars.sale_stats_median_usd,
    );
    assignUsdSaleStat("sale_stats_high_usd", mergedScalars.sale_stats_high_usd);
    assignUsdSaleStat("sale_stats_low_usd", mergedScalars.sale_stats_low_usd);

    if (Object.keys(out).length === 0) {
      return null;
    }
    return out;
  }

  /** React-friendly select update (dock “Grade condition” flow). */
  function setConditionSelect(select, value) {
    if (!select) {
      return false;
    }
    const wanted = String(value ?? "").trim();
    if (!wanted) {
      return false;
    }
    function normalizedOptionLabel(opt) {
      return String(opt.textContent || "")
        .replace(/\s+/g, " ")
        .trim();
    }
    const opts = Array.from(select.options || []);

    /** @type {HTMLOptionElement | undefined} */
    let match = opts.find((o) => {
      const val = String(o.value || "").trim();
      const lab = normalizedOptionLabel(o);
      return val === wanted || lab === wanted;
    });

    const lw = wanted.toLowerCase();
    if (!match) {
      match = opts.find((o) => {
        const val = String(o.value || "").trim().toLowerCase();
        const lab = normalizedOptionLabel(o).toLowerCase();
        return val === lw || lab === lw;
      });
    }
    if (!match) {
      match = opts.find((o) => {
        const lab = normalizedOptionLabel(o).toLowerCase();
        return lab.startsWith(lw + " (");
      });
    }

    if (!match) {
      return false;
    }
    try {
      const nativeSetter = Object.getOwnPropertyDescriptor(
        HTMLSelectElement.prototype,
        "value",
      )?.set;
      if (!nativeSetter) {
        return false;
      }
      nativeSetter.call(select, match.value);
      select.dispatchEvent(new Event("change", { bubbles: true }));
      select.dispatchEvent(new Event("input", { bubbles: true }));
      return true;
    } catch {
      return false;
    }
  }

  function formatUsdInput(n) {
    if (typeof n !== "number" || Number.isNaN(n)) {
      return "";
    }
    return (Math.round(n * 100) / 100).toFixed(2);
  }

  /**
   * Fills heuristic listing-price inputs after an estimate (~USD).
   * Returns true when at least one field was mutated.
   */
  function fillListingSuggestedPrice(estPayload) {
    const priceRaw = estPayload?.estimated_price;
    if (priceRaw == null) {
      return false;
    }
    const formatted = formatUsdInput(Number(priceRaw));
    if (!formatted) {
      return false;
    }
    let mutated = false;
    for (const sel of PRICE_INPUT_SELECTORS) {
      const inp = document.querySelector(sel);
      if (!inp || inp.tagName !== "INPUT") {
        continue;
      }
      try {
        const setter = Object.getOwnPropertyDescriptor(
          HTMLInputElement.prototype,
          "value"
        )?.set;
        if (setter) {
          setter.call(inp, formatted);
        } else {
          inp.value = formatted;
        }
        inp.dispatchEvent(new Event("input", { bubbles: true }));
        inp.dispatchEvent(new Event("change", { bubbles: true }));
        mutated = true;
      } catch {
        /* DOM edge */
      }
    }
    try {
      const primary =
        PRICE_INPUT_SELECTORS.map((sel) => document.querySelector(sel)).find(Boolean) ??
        null;
      if (primary instanceof HTMLElement && primary.blur) {
        primary.blur();
      }
    } catch {
      /* ignore */
    }
    return mutated;
  }

  globalThis.__vinyliqListingDom = {
    COMMENT_SELECTORS,
    MEDIA_SELECTORS,
    SLEEVE_SELECTORS,
    parseReleaseIdAndSurface,
    readMediaSleeveValues,
    scrapeMarketplaceClientSnapshot,
    findFirstCommentTextarea: () => findFirst(COMMENT_SELECTORS),
    findFirstMediaSelect: () => findFirstUsableSelect(MEDIA_SELECTORS),
    findFirstSleeveSelect: () => findFirstUsableSelect(SLEEVE_SELECTORS),
    setConditionSelect,
    fillListingSuggestedPrice,
    findFirstConditionSelects: () => ({
      media: findFirstUsableSelect(MEDIA_SELECTORS),
      sleeve: findFirstUsableSelect(SLEEVE_SELECTORS),
    }),
  };
})();
