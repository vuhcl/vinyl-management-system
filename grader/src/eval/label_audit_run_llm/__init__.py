"""Label-audit LLM queue runner (split: lib + main).

PEP 562 ``__getattr__`` delegates missing attributes to :mod:`.lib` so
monkeypatches on ``grader.src.eval.label_audit_run_llm.lib`` are visible
through ``import … label_audit_run_llm`` without duplicating bindings.
"""

from __future__ import annotations

from typing import Any

from . import lib
from .main import main

__all__ = [n for n in dir(lib) if not n.startswith("__")] + ["lib", "main"]


def __getattr__(name: str) -> Any:
    return getattr(lib, name)


def __dir__() -> list[str]:
    return sorted(set(dir(lib)) | {"lib", "main"})
