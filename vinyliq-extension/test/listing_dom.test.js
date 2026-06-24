/**
 * DOM contract tests for listing_dom.js (Discogs page snapshots).
 */
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import { JSDOM } from "jsdom";

const __dirname = dirname(fileURLToPath(import.meta.url));
const EXT_ROOT = join(__dirname, "..");
const FIXTURES = join(EXT_ROOT, "fixtures");

function stubRenderableElements(window) {
  window.Element.prototype.getBoundingClientRect = function getBoundingClientRect() {
    return {
      width: 120,
      height: 28,
      top: 0,
      left: 0,
      right: 120,
      bottom: 28,
      x: 0,
      y: 0,
      toJSON() {
        return {};
      },
    };
  };
  if (typeof window.Element.prototype.checkVisibility !== "function") {
    window.Element.prototype.checkVisibility = () => true;
  }
}

function loadListingDom(html, url) {
  const dom = new JSDOM(html, { url, runScripts: "outside-only" });
  const { window } = dom;
  stubRenderableElements(window);
  const script = readFileSync(join(EXT_ROOT, "listing_dom.js"), "utf8");
  window.eval(script);
  const api = window.__vinyliqListingDom;
  assert.ok(api, "listing_dom.js must attach globalThis.__vinyliqListingDom");
  return api;
}

test("parseReleaseIdAndSurface on release page", () => {
  const html = readFileSync(join(FIXTURES, "release_page.html"), "utf8");
  const api = loadListingDom(html, "https://www.discogs.com/release/37091274");
  const ctx = api.parseReleaseIdAndSurface();
  assert.equal(ctx.releaseId, "37091274");
  assert.equal(ctx.surface, "release");
});

test("parseReleaseIdAndSurface on sell post page", () => {
  const html = readFileSync(join(FIXTURES, "sell_post_page.html"), "utf8");
  const api = loadListingDom(html, "https://www.discogs.com/sell/post/456663");
  const ctx = api.parseReleaseIdAndSurface();
  assert.equal(ctx.releaseId, "456663");
  assert.equal(ctx.surface, "sell_post");
});

test("findFirstCommentTextarea on sell post fixture", () => {
  const html = readFileSync(join(FIXTURES, "sell_post_page.html"), "utf8");
  const api = loadListingDom(html, "https://www.discogs.com/sell/post/456663");
  const ta = api.findFirstCommentTextarea();
  assert.ok(ta);
  assert.equal(ta.id, "comments");
});

test("readMediaSleeveValues on sell post fixture", () => {
  const html = readFileSync(join(FIXTURES, "sell_post_page.html"), "utf8");
  const api = loadListingDom(html, "https://www.discogs.com/sell/post/456663");
  const grades = api.readMediaSleeveValues();
  assert.ok(grades);
  assert.match(grades.media, /Very Good/i);
  assert.match(grades.sleeve, /Good/i);
});
