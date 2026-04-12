#!/usr/bin/env bash
# Apply a GCS lifecycle JSON to a bucket (additive: merges with existing rules
# only if you use gcloud correctly — see Google docs; this replaces lifecycle
# config for the bucket with the contents of the given file).
#
# Usage:
#   ./scripts/mlflow_retention/apply_gcs_bucket_lifecycle.sh gs://my-mlflow-bucket \
#     scripts/mlflow_retention/gcs_lifecycle_abort_incomplete.json
#
set -euo pipefail
if [[ "${1:-}" == "" || "${2:-}" == "" ]]; then
  echo "Usage: $0 gs://BUCKET /path/to/lifecycle.json" >&2
  exit 1
fi
BUCKET="$1"
FILE="$2"
if [[ ! -f "$FILE" ]]; then
  echo "Lifecycle file not found: $FILE" >&2
  exit 1
fi
gcloud storage buckets update "$BUCKET" --lifecycle-file="$FILE"
echo "Updated lifecycle on $BUCKET from $FILE"
