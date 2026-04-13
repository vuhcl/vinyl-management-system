"""
MLflow models-from-code entry for the vinyl grader pyfunc bundle.

Loaded at ``mlflow.pyfunc.load_model`` time; do not import for training logic.
Resolves ``grader_pyfunc.py`` whether it was packed next to this file or under
``code/`` (MLflow ``code_paths`` layout).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from mlflow.models import set_model

_here = Path(__file__).resolve().parent
_impl = _here / "code" / "grader_pyfunc.py"
if not _impl.is_file():
    _impl = _here / "grader_pyfunc.py"
if not _impl.is_file():
    raise FileNotFoundError(
        "Expected grader_pyfunc.py beside this entry or under code/; "
        f"checked {_here}"
    )
_spec = importlib.util.spec_from_file_location(
    "_vinyl_grader_pyfunc_impl", _impl
)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Cannot load model implementation from {_impl}")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
set_model(_mod.VinylGraderModel())
