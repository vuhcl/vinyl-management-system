"""
Load monorepo-root `.env` into the process environment.

Implementation lives in ``shared.project_env``; this module re-exports it for
existing ``from grader.src.project_env import load_project_dotenv`` imports.
"""

from __future__ import annotations

from shared.project_env import load_project_dotenv

__all__ = ["load_project_dotenv"]
