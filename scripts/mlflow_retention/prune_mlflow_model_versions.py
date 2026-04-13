#!/usr/bin/env python3
"""
Delete oldest MLflow Model Registry versions so at most ``--keep`` remain.

Uses the tracking API (``MLFLOW_TRACKING_URI``). Production and Staging
versions are never deleted unless ``--force-delete-staged`` is passed.

Examples::

    uv run python scripts/mlflow_retention/prune_mlflow_model_versions.py \\
        --model VinylGrader --keep 20 --dry-run

    uv run python scripts/mlflow_retention/prune_mlflow_model_versions.py \\
        --model VinylGrader --keep 20
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from shared.project_env import load_project_dotenv

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model",
        required=True,
        help="Registered model name (e.g. VinylGrader from grader.yaml).",
    )
    p.add_argument(
        "--keep",
        type=int,
        default=25,
        help="Maximum number of newest versions to retain (default: 25).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions only; do not delete.",
    )
    p.add_argument(
        "--force-delete-staged",
        action="store_true",
        help="Allow deleting Staging/Production versions (dangerous).",
    )
    args = p.parse_args()
    if args.keep < 1:
        p.error("--keep must be >= 1")

    load_project_dotenv()
    uri = os.environ.get("MLFLOW_TRACKING_URI", "").strip()
    if not uri:
        logger.error(
            "MLFLOW_TRACKING_URI is unset after load_project_dotenv(). "
            "Set it in repo-root .env (see .env.template) or export it; "
            "remote example: https://mlflow.your-internal.host"
        )
        return 1

    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=uri)
    try:
        versions = client.search_model_versions(f"name='{args.model}'")
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to list versions: %s", exc)
        return 1

    if not versions:
        logger.info("No versions found for model %r.", args.model)
        return 0

    by_version_desc = sorted(
        versions, key=lambda v: int(v.version), reverse=True
    )
    if len(by_version_desc) <= args.keep:
        logger.info(
            "Model %r has %d version(s) (<= keep=%d); nothing to prune.",
            args.model,
            len(by_version_desc),
            args.keep,
        )
        return 0

    to_remove = by_version_desc[args.keep:]
    to_remove = sorted(to_remove, key=lambda v: int(v.version))

    protected = {"Production", "Staging"}
    removed = 0
    skipped = 0
    for mv in to_remove:
        stage = (mv.current_stage or "").strip() or "None"
        if stage in protected and not args.force_delete_staged:
            logger.warning(
                "Skip version %s (stage=%r); "
                "use --force-delete-staged to remove.",
                mv.version,
                stage,
            )
            skipped += 1
            continue
        if args.dry_run:
            logger.info(
                "Would delete version %s (stage=%r).",
                mv.version,
                stage,
            )
            removed += 1
            continue
        try:
            client.delete_model_version(args.model, mv.version)
            logger.info("Deleted version %s (was stage=%r).", mv.version, stage)
            removed += 1
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "delete_model_version failed for %s: %s",
                mv.version,
                exc,
            )
            skipped += 1

    logger.info(
        "Done: removed=%d skipped_or_failed=%d remaining_cap=%d",
        removed,
        skipped,
        args.keep,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
