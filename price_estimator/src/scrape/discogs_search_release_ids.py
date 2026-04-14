"""
Discogs **website** search helpers: build search URLs and extract release IDs.

Result cards link with ``href="/release/{id}-slug"``; we take the numeric ``id``
only (ignore ``/master/`` by never matching those hrefs). Year-in-decade and
multi-sort collection use ``iter_years_for_decade`` and
``build_vinyl_decade_year_sort_search_url``.
"""

from __future__ import annotations

import json
import re
from datetime import date
from urllib.parse import urlencode, urljoin, urlparse

from bs4 import BeautifulSoup

# Website search sort modes (see ``build_vinyl_decade_year_sort_search_url``).
SEARCH_HTML_SORT_MODES: tuple[str, ...] = (
    "relevance",
    "have",
    "want",
    "trending",
)

# Path segment: /release/<digits> (live site may use absolute URLs; we parse path).
RELEASE_PATH_RE = re.compile(r"^/release/(\d+)", re.IGNORECASE)
# When <a href> cards are absent (SPA shell, JSON-in-HTML), scan markup for paths.
RELEASE_ID_IN_MARKUP_RE = re.compile(
    r"/release/(\d+)(?=-|['\"\s<>#?]|$)", re.IGNORECASE
)


def build_vinyl_decade_search_url(
    *,
    decade: int,
    page: int = 1,
    base: str = "https://www.discogs.com/search",
) -> str:
    """Discogs search URL: Vinyl format + decade filter + pagination (no year/sort)."""
    q = {
        "type": "release",
        "page": max(1, int(page)),
        "format_exact": "Vinyl",
        "decade": int(decade),
        "limit": 250,
    }
    return f"{base}?{urlencode(q)}"


def iter_years_for_decade(
    decade: int, *, through_year: int | None = None
) -> tuple[int, ...]:
    """
    Inclusive calendar years ``decade .. min(decade + 9, cap)`` where ``cap`` is
    ``through_year`` or today's year. The current decade is truncated so future
    years are not requested (e.g. decade 2020 in 2026 → 2020–2026).
    """
    cap = int(through_year) if through_year is not None else date.today().year
    end = min(int(decade) + 9, cap)
    start = int(decade)
    if end < start:
        return ()
    return tuple(range(start, end + 1))


def build_vinyl_decade_year_sort_search_url(
    *,
    decade: int,
    year: int,
    page: int = 1,
    sort_mode: str = "relevance",
    base: str = "https://www.discogs.com/search",
) -> str:
    """
    Vinyl + decade facet + exact ``year``, optional HTML ``sort`` (Discogs uses
    comma-separated values such as ``have,desc`` on the website).
    """
    if sort_mode not in SEARCH_HTML_SORT_MODES:
        raise ValueError(
            f"sort_mode must be one of {SEARCH_HTML_SORT_MODES}, got {sort_mode!r}"
        )
    q: dict[str, str | int] = {
        "type": "release",
        "page": max(1, int(page)),
        "format_exact": "Vinyl",
        "decade": int(decade),
        "year": int(year),
        "limit": 250,
    }
    if sort_mode == "have":
        q["sort"] = "have,desc"
    elif sort_mode == "want":
        q["sort"] = "want,desc"
    elif sort_mode == "trending":
        q["sort"] = "trending,desc"
    return f"{base}?{urlencode(q)}"


def release_id_from_href(
    href: str | None, *, base: str = "https://www.discogs.com"
) -> str | None:
    """
    Return release id from ``href`` (relative ``/release/…`` or absolute Discogs URL).
    """
    if not href:
        return None
    raw = href.strip()
    if "/release/" not in raw.lower():
        return None
    if raw.startswith("//"):
        raw = "https:" + raw
    if raw.startswith(("http://", "https://")):
        path = urlparse(raw).path or ""
    else:
        path = raw.split("#", 1)[0].split("?", 1)[0]
        if not path.startswith("/"):
            path = "/" + path.lstrip("./")
    # Normalise accidental relative double paths
    if not path.startswith("/release/"):
        joined = urljoin(base + "/", raw)
        path = urlparse(joined).path or path
    m = RELEASE_PATH_RE.match(path)
    if not m:
        return None
    return m.group(1)


def extract_release_ids_from_html(
    html: str, *, scope_selector: str | None = "#search_results"
) -> list[str]:
    """
    Parse search (or fragment) HTML; return ordered unique release IDs.

    If ``scope_selector`` matches an element with at least one ``/release/`` link,
    only anchors inside it are used; otherwise the full document is scanned.
    """
    soup = BeautifulSoup(html, "html.parser")
    root = soup.select_one(scope_selector) if scope_selector else None
    anchors = []
    if root is not None:
        anchors = root.select('a[href*="/release/"]')
    if not anchors:
        root = soup.body or soup
        anchors = root.select('a[href*="/release/"]')
    out: list[str] = []
    seen: set[str] = set()
    for a in anchors:
        href = a.get("href")
        rid = release_id_from_href(href)
        if rid is None or rid in seen:
            continue
        seen.add(rid)
        out.append(rid)
    if not out:
        for m in RELEASE_ID_IN_MARKUP_RE.finditer(html):
            rid = m.group(1)
            if rid in seen:
                continue
            seen.add(rid)
            out.append(rid)
    return out


def js_collect_release_ids(*, scope_selector: str | None = "#search_results") -> str:
    """
    Return JavaScript source (IIFE) evaluated in the browser: ``string[]`` of IDs.

    Handles **absolute** ``https://www.discogs.com/release/…`` and **relative**
    ``/release/…`` hrefs (live search often uses absolute URLs).
    """
    scope_js = "null" if not scope_selector else json.dumps(str(scope_selector))
    return f"""
(function () {{
  function idFromHref(raw) {{
    const href = (raw || "").trim();
    if (!href || href.indexOf("/release/") < 0) return null;
    let path = "";
    try {{
      const u = new URL(href, window.location.href || "https://www.discogs.com/");
      path = (u.pathname || "").split("#")[0];
    }} catch (e) {{
      let h = href.split("#")[0].split("?")[0];
      if (!h.startsWith("/")) h = "/" + h;
      path = h;
    }}
    const m = /^\\/release\\/(\\d+)/i.exec(path);
    return m ? m[1] : null;
  }}
  const scopeSel = {scope_js};
  const sel = 'a[href*="/release/"]';
  let nodes = [];
  if (scopeSel) {{
    const sr = document.querySelector(scopeSel);
    if (sr) {{
      const inside = sr.querySelectorAll(sel);
      if (inside.length) nodes = Array.from(inside);
    }}
  }}
    if (!nodes.length) {{
    for (const cand of [
      "#search_results",
      '[data-testid="search-results"]',
      "#__next",
      '[role="main"]',
      "main",
    ]) {{
      try {{
        const el = document.querySelector(cand);
        if (el) {{
          const inside = el.querySelectorAll(sel);
          if (inside.length) {{ nodes = Array.from(inside); break; }}
        }}
      }} catch (e) {{}}
    }}
  }}
  if (!nodes.length) nodes = Array.from(document.querySelectorAll(sel));
  const out = [];
  const seen = new Set();
  for (const a of nodes) {{
    const id = idFromHref(a.getAttribute("href"));
    if (!id) continue;
    if (seen.has(id)) continue;
    seen.add(id);
    out.push(id);
  }}
  if (!out.length) {{
    const chunks = [];
    const grab = (el) => {{
      try {{
        if (el && el.innerHTML) chunks.push(el.innerHTML);
      }} catch (e) {{}}
    }};
    grab(document.getElementById("__next"));
    grab(document.querySelector("main"));
    grab(document.getElementById("search_results"));
    grab(document.body);
    const blob = chunks.join("\\n");
    const re = /\\/release\\/(\\d+)/gi;
    let m;
    while ((m = re.exec(blob)) !== null) {{
      const id = m[1];
      if (seen.has(id)) continue;
      seen.add(id);
      out.push(id);
    }}
  }}
  return out;
}})()
"""
