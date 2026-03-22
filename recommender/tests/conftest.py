"""
Put the monorepo root on sys.path so imports like `shared.*` work under pytest.

`pyproject.toml` also sets `pythonpath = ["."]` (pytest 7+).
This file is a fallback if that option is not applied.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_root_str = str(_ROOT)
if _root_str not in sys.path:
    sys.path.insert(0, _root_str)
