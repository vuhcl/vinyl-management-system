import * as fs from "fs";
import * as path from "path";

export interface GoldenExample {
  id: string;
  text: string;
  expected_media_condition: string;
  expected_sleeve_condition: string;
  notes?: string;
}

/** Shape of ``grader/demo/golden_predict_demo.json`` used by demo + warm-profile */
export interface GoldenPredictDemoJson {
  demo_release_id: number | string;
  /**
   * Discogs catalogue **Master** numeric id (`/master/{id}` web path segment).
   * Required with catalog UX: home search → masters results → this master → chosen ``demo_release_id``.
   * Override env: **`DEMO_MASTER_ID`**.
   */
  demo_master_id?: number | string;
  release_description?: string;
  sell_post_url: string;
  /** Required when ``DEMO_CATALOG_UX`` (default ``1``): navbar search phrase before picking ``demo_master_id``. */
  search_query?: string;
  min_price_delta_usd?: number;
  examples?: GoldenExample[];
}

export function goldenPredictDemoPath(): string {
  const fallback = path.resolve(
    __dirname,
    "..",
    "..",
    "..",
    "grader",
    "demo",
    "golden_predict_demo.json",
  );
  const fromEnv = process.env.GOLDEN_FILE?.trim();
  if (fromEnv && fromEnv.length > 0) {
    return path.resolve(fromEnv);
  }
  return fallback;
}

/** ``minExamples``: when set, require at least that many golden examples */
export function readGoldenPredictDemo(
  opts: { minExamples?: number } = {},
): GoldenPredictDemoJson {
  const file = goldenPredictDemoPath();
  const raw = fs.readFileSync(file, "utf8");
  const parsed = JSON.parse(raw) as GoldenPredictDemoJson;
  const min = opts.minExamples;
  if (min != null) {
    if (!parsed.examples || parsed.examples.length < min) {
      throw new Error(
        `Golden file ${file} must contain at least ${min} examples.`,
      );
    }
  }
  return parsed;
}
