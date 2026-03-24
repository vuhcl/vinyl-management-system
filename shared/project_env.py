"""
Load monorepo-root `.env` into the process environment.

Used by recommender CLIs and other entrypoints so `DISCOGS_*`, `MONGO_*`,
etc. can live in a repo-root `.env` without manual `export`.

`python-dotenv` is optional at import time but listed in project dependencies.
Variables already set in the environment are not overwritten.
"""

from __future__ import annotations

from pathlib import Path


def load_project_dotenv() -> None:
    """
    Search upward from cwd, then from `shared/`, for a `.env` file and load it.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    def walk_for_dotenv(start: Path) -> bool:
        cur = start.resolve()
        for _ in range(20):
            candidate = cur / ".env"
            if candidate.is_file():
                load_dotenv(candidate)
                return True
            if cur.parent == cur:
                break
            cur = cur.parent
        return False

    if walk_for_dotenv(Path.cwd()):
        return
    walk_for_dotenv(Path(__file__).resolve().parent.parent)
