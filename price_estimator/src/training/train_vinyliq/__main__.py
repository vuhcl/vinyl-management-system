"""``python -m price_estimator.src.training.train_vinyliq`` entrypoint."""

from __future__ import annotations

from .cli import _cli_main

if __name__ == "__main__":
    raise SystemExit(_cli_main())
