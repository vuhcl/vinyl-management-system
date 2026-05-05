"""Label-audit LLM queue runner (split: lib + main)."""

from __future__ import annotations

import sys
from types import ModuleType

from . import lib as _lib
from .main import main

# Re-export all public and private names from lib (``import *`` skips ``_*``).
_m: ModuleType = sys.modules[__name__]
for _k in dir(_lib):
    if _k.startswith("__"):
        continue
    setattr(_m, _k, getattr(_lib, _k))

del _m, _k

__all__ = [n for n in dir(_lib) if not n.startswith("__")] + ["main"]
