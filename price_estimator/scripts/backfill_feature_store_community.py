#!/usr/bin/env python3
"""
Deprecated (plan §1b): ``releases_features`` no longer stores ``want_count`` /
``have_count``. Community counts live in ``marketplace_stats.sqlite`` via
``collect_marketplace_stats.py`` / ``GET /releases``.
"""
from __future__ import annotations

import sys


def main() -> int:
    print(
        "This script is retired: use marketplace_stats + collect_marketplace_stats.py.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
