"""
Load monorepo-root `.env` into the process environment.

Used by recommender CLIs and other entrypoints so `DISCOGS_*`, `MONGO_*`,
etc. can live in a repo-root `.env` without manual `export`.

`python-dotenv` is optional at import time but listed in project dependencies.
Variables already set in the environment are not overwritten.
"""

from __future__ import annotations

import os
from pathlib import Path


def load_project_dotenv() -> None:
    """
    Load monorepo-root ``.env`` first, then any ``.env`` found walking upward
    from the current working directory.

    Root-first avoids a ``.env`` in the home directory (cwd ``~``) shadowing
    ``MLFLOW_TRACKING_URI`` and other keys that only exist in this repo's
    ``.env``.
    """
    try:
        from dotenv import dotenv_values, load_dotenv
    except ImportError:
        return

    repo_root = Path(__file__).resolve().parent.parent
    repo_env = repo_root / ".env"
    if repo_env.is_file():
        load_dotenv(repo_env)
        # Shell may export MLFLOW_TRACKING_URI= (empty); dotenv won't override
        # by default — still apply a non-empty value from the file.
        vals = dotenv_values(repo_env) or {}
        file_uri = (vals.get("MLFLOW_TRACKING_URI") or "").strip()
        env_uri = (os.environ.get("MLFLOW_TRACKING_URI") or "").strip()
        if file_uri and not env_uri:
            os.environ["MLFLOW_TRACKING_URI"] = file_uri

    def walk_for_dotenv(start: Path) -> bool:
        cur = start.resolve()
        for _ in range(20):
            candidate = cur / ".env"
            if candidate.is_file():
                if candidate.resolve() != repo_env.resolve():
                    load_dotenv(candidate)
                return True
            if cur.parent == cur:
                break
            cur = cur.parent
        return False

    if walk_for_dotenv(Path.cwd()):
        return
    if not repo_env.is_file():
        walk_for_dotenv(repo_root)
