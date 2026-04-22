#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH=.

MASTER_JSON="${MASTER_JSON:-artifacts/discogs_master_to_aoty.json}"
RELEASE_JSON="${RELEASE_JSON:-artifacts/discogs_release_to_aoty.json}"
STATS_PARQUET="${STATS_PARQUET:-artifacts/discogs_master_stats.parquet}"
FEATURE_DB="${FEATURE_DB:-price_estimator/data/feature_store.sqlite}"
MARKETPLACE_DB="${MARKETPLACE_DB:-price_estimator/data/cache/marketplace_stats.sqlite}"

SKIP_RELEASE=0
SKIP_STATS=0
for arg in "$@"; do
  case "$arg" in
    --skip-release) SKIP_RELEASE=1 ;;
    --skip-stats) SKIP_STATS=1 ;;
    -h|--help)
      cat <<'EOF'
Refresh Discogs→AOTY downstream artifacts after updating
artifacts/discogs_master_to_aoty.json (e.g. dump-join or phase-A).

  1. Phase B: compose release→AOTY JSON (+ Mongo discogs_release_aoty upserts).
  2. Reranker: rebuild discogs_master_stats.parquet from the feature store.

Requires Mongo with discogs_release_master populated (same as
build_discogs_release_to_aoty_artifact.py). Run from repo root or anywhere.

Usage:
  ./scripts/refresh_discogs_aoty_artifacts.sh
  MASTER_JSON=artifacts/foo.json ./scripts/refresh_discogs_aoty_artifacts.sh --skip-stats

Options:
  --skip-release   Only rebuild stats (skip Phase B).
  --skip-stats     Only run Phase B (skip stats parquet).

Env overrides (optional):
  MASTER_JSON, RELEASE_JSON, STATS_PARQUET, FEATURE_DB, MARKETPLACE_DB
  MONGO_URI, MONGO_DB (Python defaults; set if not localhost/music)

Afterward: re-run ingest + preprocess if interactions.parquet should use the new map.
EOF
      exit 0
      ;;
    *)
      echo "Unknown option: $arg (try --help)" >&2
      exit 1
      ;;
  esac
done

if [[ ! -f "$MASTER_JSON" ]]; then
  echo "Master map not found: $MASTER_JSON" >&2
  exit 1
fi

if [[ "$SKIP_RELEASE" -eq 0 ]]; then
  echo "==> Phase B: release → AOTY ($RELEASE_JSON)"
  uv run python scripts/build_discogs_release_to_aoty_artifact.py \
    --master-json "$MASTER_JSON" \
    --output "$RELEASE_JSON"
else
  echo "==> Skipping Phase B (--skip-release)"
fi

if [[ "$SKIP_STATS" -eq 0 ]]; then
  echo "==> Reranker stats: $STATS_PARQUET"
  MP_ARGS=()
  if [[ -f "$MARKETPLACE_DB" ]]; then
    MP_ARGS=(--marketplace-db "$MARKETPLACE_DB")
  else
    echo "    (marketplace DB missing: $MARKETPLACE_DB — Tier B will be zero)" >&2
  fi
  uv run python scripts/build_discogs_master_stats_artifact.py \
    --master-json "$MASTER_JSON" \
    --feature-db "$FEATURE_DB" \
    "${MP_ARGS[@]}" \
    --output "$STATS_PARQUET"
else
  echo "==> Skipping stats parquet (--skip-stats)"
fi

echo "Done. Re-run ingest + preprocess if interactions.parquet should use the new release map."
