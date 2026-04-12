#!/usr/bin/env bash
# Hard-delete MLflow runs that are already soft-deleted (lifecycle_stage=deleted)
# and optionally remove their artifacts from cloud storage.
#
# Set:
#   MLFLOW_TRACKING_URI   — tracking server (for client; gc still needs DB URI)
#   MLFLOW_BACKEND_STORE_URI — SQLAlchemy URI for the tracking DB (Postgres/MySQL)
#   MLFLOW_ARTIFACTS_DESTINATION — e.g. gs://bucket/mlflow-artifacts
#
# Example (adjust URIs):
#   export MLFLOW_TRACKING_URI=https://mlflow.example.com
#   export MLFLOW_BACKEND_STORE_URI=postgresql://mlflow:pass@host:5432/mlflow
#   export MLFLOW_ARTIFACTS_DESTINATION=gs://my-mlflow-artifacts
#   ./scripts/mlflow_retention/mlflow_gc_deleted.example.sh
#
# See: https://mlflow.org/docs/latest/cli.html#mlflow-gc
#
set -euo pipefail
: "${MLFLOW_BACKEND_STORE_URI:?Set MLFLOW_BACKEND_STORE_URI}"
: "${MLFLOW_ARTIFACTS_DESTINATION:?Set MLFLOW_ARTIFACTS_DESTINATION}"
EXTRA=(--older-than 90d)
if [[ "${MLFLOW_GC_RUN_IDS:-}" != "" ]]; then
  EXTRA=(--run-ids "$MLFLOW_GC_RUN_IDS")
fi
exec uv run mlflow gc \
  --backend-store-uri "$MLFLOW_BACKEND_STORE_URI" \
  --artifacts-destination "$MLFLOW_ARTIFACTS_DESTINATION" \
  "${EXTRA[@]}"
