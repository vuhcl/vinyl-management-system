"""Shared ordinal / decade sentinels for cross-grade (NM vs VG+) sale pooling."""
from __future__ import annotations

# Effective-grade NM gate (min(media, sleeve) at or above this → treated as NM tier).
ORDINAL_NM: float = 7.0

# Strict VG+ box for symmetric NM−VG+ contrast (open upper bound).
ORDINAL_VG_PLUS_LO: float = 6.0
ORDINAL_VG_PLUS_HI: float = 7.0

# Decade bin when release year is missing / non-finite (matches ``add_anchor_decade_bins``).
MISSING_YEAR_DECADE_SENTINEL: float = -9990.0
