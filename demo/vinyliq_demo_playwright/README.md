# vinyliq_demo_playwright

End-to-end demo automation for the [VinylIQ extension](../../vinyliq-extension/)
running against the GKE-deployed price + grader APIs.

**Playback order:** the recorder tab opens **`https://www.discogs.com/`** (**`DEMO_START_URL`**) whenever **`DEMO_CATALOG_UX=1`** (default). Scripted **`inject_extension_storage`** still opens **`popup.html`** in a **different** browser tab (**not** toolbar). **`goto_seller_listing`** then runs **home → **`search_query`** → masters search (**`type=master`**) → **`/master/DEMO_MASTER_ID`** → **`/release/DEMO_RELEASE_ID`** → **Sell copy** → **`/sell/post/…`** (see **`fixtures/discogs_navigation.ts`**). **`SELL_POST_URL`** still anchors assertions and deep-link mode. **`npm run test:ci`** passes **`DEMO_CATALOG_UX=0`** to jump straight **`SELL_POST_URL`**. **`#vinyliq-sell-dock`** gates grading/estimate UI.

1. **Catalog → seller draft**: Navbar search (**`search_query`**), Masters facet, golden **master**, pick the curated **release**, **Sell**, then **`#vinyliq-sell-dock`** + golden comments → **Grade** / **Estimate** wedges.

The golden file [`grader/demo/golden_predict_demo.json`](../../grader/demo/golden_predict_demo.json)
supplies **`search_query`**, **`demo_master_id`** (Discogs Master for master-first navigation), **`demo_release_id`**, **`sell_post_url`**, graded comments, expected condition ladders, **`MIN_PRICE_DELTA_USD`**, and optional narration fields like **`release_description`**.

## Why this is its own project

It lives under `demo/` and uses npm rather than uv because Playwright
itself is a Node ecosystem tool. No Python is involved here.

## Prerequisites

- Node 20+
- A pre-authenticated Chrome profile for the project's Discogs account
  (logging in via Playwright is fragile; reuse a manually-prepared
  profile dir).
- An accessible seller listing URL on that account
  (`https://www.discogs.com/sell/post/<id>`).
- The GKE demo deploy reachable (or local uvicorn on the default ports).

## Setup

```bash
cd demo/vinyliq_demo_playwright
npm install
npm run install-browsers      # downloads Playwright's chromium build
```

## Configuration (env)

Set via repo-root `.env` and `set -a && source .env && set +a`, or via a
local `.env` in this folder. The recording runbook in `RECORDING.md`
lists the canonical defaults.

| Var | Default | Notes |
| --- | --- | --- |
| `CHROME_PROFILE_DIR` | _(required)_ bundled | Stable profile dir (Discogs logged in). **Unused** when **`PLAYWRIGHT_CONNECT_CDP`** is set—you control the profile inside Google Chrome itself |
| `EXTENSION_PATH` | _(required)_ bundled | Repo path **`vinyliq-extension/`** for Playwright unpacked load. With **`PLAYWRIGHT_CONNECT_CDP`**, VinylIQ **must still be loaded manually** in Chrome, but **`EXTENSION_PATH`** is **recommended**: Playwright derives the unpacked extension id from the canonical absolute path and skips flaky MV3-over-CDP service-worker discovery (**`VINYLIQ_EXTENSION_ID`** still overrides when Chrome’s id differs) |
| `PRICE_API_BASE` | `http://127.0.0.1:8801` | Price API base URL (no trailing slash) |
| `GRADER_API_BASE` | `http://127.0.0.1:8090` | Grader API base URL (no trailing slash) |
| `SELL_POST_URL` | from golden file | Target seller draft (**deep-link fallback** path when **`DEMO_CATALOG_UX=0`**); asserted against **`sell_post_url`** |
| `DEMO_START_URL` | **`https://www.discogs.com/`** or **`SELL_POST_URL`** | With **`DEMO_CATALOG_UX=1`**, first paint is the Discogs homepage (override if you need a different **`discogs.com`** entry); with **`DEMO_CATALOG_UX=0`**, defaults to **`sellPostUrl`** |
| `DEMO_CATALOG_UX` | `1` (**`npm run test:ci` sets `0`**) | **`1`** = home → search → **`type=master`** → **`/master/{demo_master_id}`** → **`/release`** → Sell → **`/sell/post`**; **`0`** = skip straight to **`sell_post`** |
| `DEMO_MASTER_ID` | from golden **`demo_master_id`** | Overrides Discogs Master id when probing another pressing without editing JSON |
| `DEMO_DEEP_LINK_SELL_POST` | `0` | Shorthand **`1`** = **`DEMO_CATALOG_UX=0`** semantics |
| `DEMO_RELEASE_ID` | from golden file | Discogs release ID; defaults to 456663 |
| `MIN_PRICE_DELTA_USD` | from golden file | Minimum absolute delta required between paired estimates (golden examples A vs B) |
| `GOLDEN_FILE` | `../../grader/demo/golden_predict_demo.json` | Golden file path |
| `VINYLIQ_API_KEY` | _(empty)_ | Sent as `X-API-Key` to both services |
| `PLAYWRIGHT_CONNECT_CDP` | _(empty)_ | Debugger URL (e.g. **`http://127.0.0.1:9222`**) → attach to **your** Google Chrome (see subsection below); bypasses bundled Chromium when Cloudflare never clears. Harness uses **`chromiumContextsForMv3`** in **`fixtures/extension.ts`** when probing MV3 visibility. |
| `VINYLIQ_EXTENSION_ID` | _(auto)_ | Chrome extension hostname (under **Details** on **`chrome://extensions`**); overrides MV3 detection when using CDP. **Reference unpacked ID for this repo’s VinylIQ listing:** **`bhhebpplkmapokijgbmeejhamdlkbcao`** — yours may differ if **`vinyliq-extension/`** is loaded from another absolute path; confirm under **chrome://extensions** |
| `PLAYWRIGHT_CDP_MV3_PROBE_MS` | `5000` | **CDP + pinned id:** ms to probe for enumerable MV3 before dock fallback |
| `PLAYWRIGHT_CDP_DOCK_FALLBACK` | `1` | **CDP + pinned id:** if still no SW enumeration, attach anyway and rely on `#vinyliq-sell-dock` + internal `popup.html` tab hops |
| `PLAYWRIGHT_SELL_DOCK_TIMEOUT_MS` | `240000` | Max wait for `#vinyliq-sell-dock` after navigating to **`SELL_POST_URL`** |
| `PLAYWRIGHT_EXTENSION_SW_TIMEOUT_MS` | `120000` | How long to wait for the extension MV3 service worker after launch |
| `PLAYWRIGHT_REDUCE_AUTOMATION_SIGNALS` | `1` | When enabled, drops `--enable-automation` and masks `navigator.webdriver` via `--disable-blink-features=AutomationControlled` (helps Cloudflare re-challenges a bit); set `0` to revert to plain Playwright defaults |
| `WARM_PROFILE_INCLUDE_SELL_POST` | _(off)_ | Set **`1`** to visit **`SELL_POST_URL`** / golden **`sell_post_url`** during **`warm-profile`** after Login (often a **second Cloudflare solve**); omit for login-only warmup |
| `PLAYWRIGHT_WARM_NAV_SETTLE_MS` | `3500` | After each **`warm-profile`** navigation: wait **n** ms (then prompts) so Cloudflare tokens can persist; **`0`** disables |
| `WARM_PROFILE_URLS` | _(built-in list)_ | When set, visit those URLs **in order** (comma-separated or pipe-separated). When unset: default login URL only (**`https://www.discogs.com/login`** or **`WARM_PROFILE_START_URL`**). Seller hop requires **`WARM_PROFILE_INCLUDE_SELL_POST=1`** unless you spelled every URL explicitly here |
| `DEMO_VIDEO_CHAPTERS` | `1` | Full-frame text cards on the seller tab explaining each scripted phase (**`fixtures/demo_video_ann.ts`**). Set **`0`** for silent runs (recommended with **`npm run test:ci`**) |
| `DEMO_VIDEO_CHAPTER_MS` | **`12400`** | Fullscreen chapter cards (ms, 800..44000); subtitles are long—raise this if viewers still rush |
| `DEMO_CATALOG_SEARCH_INTRO_CHAPTER_MS` | **`4000`** | Before navbar search (**“Let's find the right release…”**); **`700`..`20000`** |
| `DEMO_RELEASE_CONFIRMED_CHAPTER_MS` | **`5000`** | **“The details match…”** on `/release/` before Sell; **`700`..`20000`** |
| `DEMO_RELEASE_HMM_CHAPTER_MS` | **`2000`** | Headline-only “Hmm…” `/release/` card only (ms, `700`..`20000`); longer chapter subtitles still use ``DEMO_VIDEO_CHAPTER_MS`` |
| `DEMO_RELEASE_AFTER_HMM_SCROLL_MS` | **`9800`** | After “Hmm…” + nudge-scroll on `/release/` — bare dwell before the confirmation chapter so you can read tracklist / identifiers (`600`..`45000`) |
| `DEMO_SELLER_STRIP_READ_MS` | **`7200`** | After the bottom insight strip appears, wait this long before keystrokes / next action (ms, 400..28000) |
| `DEMO_SELLER_STRIP_MS` | **`26000`** | How long the gradient strip stays mounted (ms, 2800..48000); keep ≥ read delay so copy stays visible while the script waits |
| `DEMO_STRIP_ABOVE_DOCK_GAP_PX` | **`18`** | Gap (px, 0..80) between **`#vinyliq-sell-dock`** and the narration strip (**`fixtures/demo_video_ann.ts`**) |
| `DEMO_SEGUE_STRIP_MS` | **`24000`** | Bridge strips after grades / **`after_first_estimate`** (**`fixtures/demo_video_ann.ts`**; **`4500`..`52000`**) |
| `DEMO_SEGUE_STRIP_READ_MS` | **`7600`** | Post-mount pause before continuing after segue strips (`700..28000`) |
| `DEMO_SEGUE_AFTER_FIRST_ESTIMATE_MS` | **`8000`** | **“That's Copy A's number…”** strip only (`900`..`30000`); other segues use ``DEMO_SEGUE_STRIP_*`` |
| `DEMO_SEGUE_AFTER_FIRST_GRADE_TRIM_MS` | **`0`** | **`after_first_grade`** / **`after_second_grade`** optional trim (`0`..`12000`; floors `4500` / `700` vs ``DEMO_SEGUE_STRIP_*``) |
| `DEMO_AFTER_FIRST_ESTIMATE_BARE_MS` | **`9800`** | Silent dwell before **That's Copy A's number…** and before the Copy **B** session **outro** (same knob; default ~Copy A strip remainder + ~3 s at stock **`DEMO_SELLER_STRIP_READ_MS`** + **12 s** Copy A strip; lengthen e.g. **`19000`** for a slower Copy **B** outro) (`0..72000`) |
| `DEMO_SKIP_SEGUES` | _(off)_ | **`1`** → skip **That's Copy A's number…** and session **outro** after Copy B. **Grade** segues (**after_first_grade**, **after_second_grade**) still show when **`DEMO_VIDEO_CHAPTERS=1`** |
| `DEMO_COMMENT_TYPING_DELAY_MS` | **`38`** | Ms between keystrokes for seller condition comments (**`0`** → instant **`fill`**). Stacks with **`playwright.config.ts`** **`slowMo`** on each Playwright primitive — both show up **1×** in **`recordVideo`**, not slowed again in export |
| `DEMO_HYBRID` | `0` | **`1`** = **hybrid operator** run: chapter/strip annotations + typing golden comments stay automated; **you** drive Discogs navigation (**`waitForURL`** **`/sell/post/…`** matches golden), **Grade condition**, and **Get estimate** (see **`tests/demo.spec.ts`**) |
| `DEMO_HYBRID_NAV_TIMEOUT_MS` | same as step budget | Max wait (ms, ≥60 000) for **`/sell/post/{id}`** to appear in hybrid mode |
| `DEMO_HYBRID_STEP_TIMEOUT_MS` | **`1800000`** | Max wait (ms, ≥30 000) per human beat — grade dropdowns settle, overlay estimate, etc. Test timeout scales with **`demoHybridSuggestedTestTimeoutMs`** |
| `DEMO_HYBRID_PAUSE` | `0` | Hybrid only: **`1`** → **`page.pause()`** after each scripted comment (Playwright Inspector resume) |
| `DEMO_FULL_WALKTHROUGH` | _(empty)_ | Legacy alias; **`1`** forces **`DEMO_CATALOG_UX`** on even if overridden elsewhere |

Bundled **`npm test`** **`recordVideo`** also enables Playwright **`showActions`**: outlines + short captions on each atomic interaction (fills, waits, etc.). **`PLAYWRIGHT_CONNECT_CDP`** bypasses **`recordVideo`** (see **`fixtures/extension.ts`**) unless you separately wire Chrome screencast.

## Run

```bash
set -a && source ../../.env && set +a   # or source the local .env
npm test
```

### Cannot pass Cloudflare in Playwright’s Chromium?

Discogs may **never** trust Playwright’s automation Chromium. Attach over **Chrome
DevTools** so you authenticate in **real Google Chrome**, then automate the session
already cleared by Turnstile.

**Quit Chrome windows that reuse the chosen `--user-data-dir`**. macOS bootstrap:

```bash
PROFILE="$HOME/chrome-profiles/vinyliq-demo-discogs"
/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome \
  --user-data-dir="$PROFILE" \
  --remote-debugging-port=9222
```

In that window: **`chrome://extensions`** → Developer mode → **Load unpacked** → repo
folder **`vinyliq-extension/`** → browse **`https://www.discogs.com/login`** manually and finish
human checks / login while Chrome stays open.

```bash
export PLAYWRIGHT_CONNECT_CDP="http://127.0.0.1:9222"
export EXTENSION_PATH="/absolute/path/to/vinyl_management_system/vinyliq-extension"
# Recommended with CDP: set EXTENSION_PATH for pinned id (+ optional MV3 probes); demo uses dock readiness, not toolbar popup.
# Fallback: export VINYLIQ_EXTENSION_ID="bhhebpplkmapokijgbmeejhamdlkbcao"
#             (matches this repo's reference unpack — copy from chrome://extensions if yours differs)
# CHROME_PROFILE_DIR is ignored — unset from .env when using CDP
npm test
```

Teardown disconnects Playwright (**`browser.close()` on a CDP link drops the websocket;
Chrome stays running**). Omit **`PLAYWRIGHT_CONNECT_CDP`** to revert to bundled
Chromium (**`EXTENSION_PATH`** + **`CHROME_PROFILE_DIR`** required again).

Without **`EXTENSION_PATH`** (or **`VINYLIQ_EXTENSION_ID`**), Playwright polls for an
MV3 **`chrome-extension://`** service worker (**often absent over CDP — logs `ext_sw=0`**).

**Recommended:** **`export EXTENSION_PATH=…/vinyliq-extension`** (deterministic unpacked id). Launch uses brief **automated navigations** to **`popup.html`** in a normal browser tab — **that is internal storage/MV3 wake logic, not expecting you to click the extension icon.**

**Fallback:** **`export VINYLIQ_EXTENSION_ID=…`** from **chrome://extensions → VinylIQ → Id**
when derivation disagrees with what Chrome displays.

### If Cloudflare challenges you on every **bundled** run

The demo launches **Playwright’s bundled Chromium**, not Google Chrome.
Cookies/session data you bake only in Chrome often **fail to decrypt** in
another Chromium embed, so Cloudflare keeps treating each `npm test` as a cold
visitor. Do this once:

```bash
export CHROME_PROFILE_DIR="/absolute/path/to/a/fresh/playwright/discogs/profile"
npm run warm-profile   # Chromium only (no unpacked extension → fewer scripted hits on Discogs/CF)

export EXTENSION_PATH="/absolute/path/to/vinyliq-extension"   # required for npm test only
npm test
```

`warm-profile` defaults to **login only** (**`https://www.discogs.com/login`**
unless **`WARM_PROFILE_START_URL`** overrides). After each navigation it waits
**`PLAYWRIGHT_WARM_NAV_SETTLE_MS`** (default 3500 ms) so Cloudflare cookies can persist
before prompts or chained navigations.

**`/sell/post` is skipped unless** **`WARM_PROFILE_INCLUDE_SELL_POST=1`** (or you enumerate
everything in **`WARM_PROFILE_URLS`**) — that path often retriggers shields even mid-session.

**Why not load the extension here?** The MV3 service worker/content scripts fetch
against Discogs; Cloudflare can treat that traffic as suspicious and hammer you
with back-to-back interstitials inside the **same** `npm run warm-profile`. Seeding a
bare Chromium profile avoids that; **`npm test` may still flash a challenge once**
when VinylIQ attaches.

After that profile is warm, **`npm run install-browsers` still updates Chromium**;
after a Playwright-major Chrome revision bump you may need to revisit Discogs once.

An **`npm test`** run (bundled Chromium with **`EXTENSION_PATH`**) emits a **`recordVideo`**
artifact under **`demo/vinyliq_demo_playwright/recordings/`** (Playwright assigns the
exact **`.webm`** filename under that directory when the MV3-backed context shuts down).
Interactive highlights come from **`recordVideo.showActions`**, while full-frame narration
slides are controlled via **`DEMO_VIDEO_CHAPTERS`** / **`DEMO_VIDEO_CHAPTER_MS`**
(**`fixtures/demo_video_ann.ts`**). **`PLAYWRIGHT_CONNECT_CDP`** attaches to Chrome that
normally **does not** record through Playwright (**`RECORDING.md` Phase 7b** stays the
fallback for glossy audio/video). The [`RECORDING.md`](RECORDING.md) runbook walks
finding the freshest **`.webm`**, ffmpeg → **`demo.mp4`**, and uploading to GitHub.

## Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `Missing required env var: CHROME_PROFILE_DIR` | profile not exported | Only for **bundled** Chromium (`PLAYWRIGHT_CONNECT_CDP` unset): `export CHROME_PROFILE_DIR=...` |
| `Browser.getWindowForTarget` / `Browser window not found`, browser exits instantly | **`PLAYWRIGHT_USE_SYSTEM_CHROME=1`** (stock Chrome does not reliably side-load unpacked extensions) or profile **SingletonLock** (another Chromium/Chrome holds `CHROME_PROFILE_DIR`) | Unset **`PLAYWRIGHT_USE_SYSTEM_CHROME`**; quit all browsers using that profile dir before `npm test` |
| Test hangs at `await commentEl.waitFor` | Discogs page loaded with a new layout | re-discover selectors interactively, update `vinyliq-extension/listing_dom.js` (+ `fixtures/demo_runner.ts`) |
| `Extension service worker not found` … | Older harness or **`PLAYWRIGHT_CDP_DOCK_FALLBACK=0`** | Default CDP+dock runner does **not** require Playwright-visible MV3. If you forced strict SW polls: **`EXTENSION_PATH`** / **`VINYLIQ_EXTENSION_ID`**, ensure VinylIQ is enabled |
| Estimate returns 401 | `VINYLIQ_API_KEY` mismatch | unset on server side, or set the same value here |
| Forever stuck verifying in automation Chromium | Cloudflare classifies bundled Playwright as a bot | **`PLAYWRIGHT_CONNECT_CDP`**: bootstrap Google Chrome manually, finish Turnstile, leave Chrome running → `npm test` |
| `expect.poll` times out on the seller selectors | grader confidence on golden text is low and rules changed the grade | re-curate the golden file rows against the live grader |
| Cloudflare repeats **during** **`npm run warm-profile`** itself | Scripted **`login` → `/sell/post`** commonly triggers separate Cloudflare wall checks even in one Chromium session | Default warm stays **login-only**; enable **`WARM_PROFILE_INCLUDE_SELL_POST=1`** only when you need that URL baked ahead of **`npm test`**. **`PLAYWRIGHT_WARM_NAV_SETTLE_MS`** pauses between hops so clearance cookies persist |
| Cloudflare loop on **every bundled** `npm test`, even though you passed it in desktop Chrome once | Profiles differ: warmed cookies for Google Chrome seldom decode in bundled Chromium—or Playwright upgraded Chromium revisions | Prefer **`warm-profile`** for bundles; **`PLAYWRIGHT_CONNECT_CDP`** is the workaround when Discogs never validates automation Chromium |
| Test times out at seller or release page | Discogs **Cloudflare** or slow DOM after Turnstile | Try **`warm-profile`** + bundled Chromium; if shields never disappear, **`PLAYWRIGHT_CONNECT_CDP`**. Quit duplicate Chrome instances touching the same **`--user-data-dir`**. VPN/datacenter IPs worsen checks |

## How automation differs from a manual click tour

Playwright cannot open Chrome’s MV3 toolbar bubble
([upstream issue](https://github.com/microsoft/playwright/issues/5593)).

The recorder tab **`goto`**s **Discogs** before scripted steps: **`https://www.discogs.com/`** when **`DEMO_CATALOG_UX=1`** (default), **`SELL_POST_URL`** when **`DEMO_CATALOG_UX=0`**. Seeds API URLs by opening **`popup.html` in another tab** (**not** the toolbar). **`goto_seller_listing`** either runs **catalog UX** (`fixtures/discogs_navigation.ts`) or skips straight **`goto`** **`SELL_POST_URL`**, then waits **`#vinyliq-sell-dock`**.

If you need a purely human recording of the toolbar, use **`RECORDING.md`** (“Phase 7b”) manual capture and splice.
