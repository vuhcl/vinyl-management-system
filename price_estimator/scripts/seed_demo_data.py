#!/usr/bin/env python3
"""Create small synthetic marketplace + feature DBs for local testing without Discogs."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    sys.path.insert(0, str(ROOT.parent))
    from price_estimator.src.storage.feature_store import FeatureStoreDB
    from price_estimator.src.storage.marketplace_db import MarketplaceStatsDB

    mp = ROOT / "data" / "cache" / "marketplace_stats.sqlite"
    fs = ROOT / "data" / "feature_store.sqlite"
    mdb = MarketplaceStatsDB(mp)
    fdb = FeatureStoreDB(fs)

    demo = [
        ("100", 19.99, 120, 400, "Rock", 1975, "US", "Vinyl, LP", "100", 1975),
        ("101", 24.50, 80, 200, "Jazz", 1960, "US", "Vinyl, LP", "101", 1960),
        ("102", 8.00, 300, 2000, "Rock", 1995, "UK", "CD", "102", 1995),
        ("103", 45.00, 50, 80, "Electronic", 2000, "DE", "Vinyl, 12", "90", 1998),
        ("104", 12.00, 200, 900, "Rock", 1988, "US", "Vinyl, LP", "104", 1988),
        ("105", 60.00, 30, 40, "Jazz", 1965, "US", "Vinyl, LP", "105", 1965),
        ("106", 15.50, 90, 350, "Punk", 1979, "UK", "Vinyl, 7", "106", 1979),
        ("107", 22.00, 100, 300, "Rock", 1972, "US", "Vinyl, LP", "107", 1972),
        ("108", 9.99, 250, 1500, "Pop", 1985, "US", "Vinyl, LP", "108", 1985),
        ("109", 35.00, 70, 150, "Electronic", 1998, "UK", "2xVinyl", "109", 1998),
        ("110", 18.00, 140, 500, "Rock", 1991, "US", "Vinyl, LP", "110", 1991),
        ("111", 28.00, 95, 220, "Jazz", 1970, "US", "Vinyl, LP", "111", 1970),
        ("112", 11.50, 180, 800, "Rock", 2005, "US", "Vinyl, LP", "112", 2005),
        ("113", 55.00, 40, 60, "Jazz", 1959, "US", "Vinyl, LP", "113", 1959),
        ("114", 14.00, 210, 700, "Rock", 1982, "UK", "Vinyl, LP", "114", 1982),
        ("115", 32.00, 85, 180, "Electronic", 2003, "DE", "Vinyl, LP", "115", 2003),
        ("116", 20.00, 110, 380, "Rock", 1978, "US", "Vinyl, LP", "116", 1978),
        ("117", 7.50, 400, 2500, "Rock", 1999, "US", "CD", "117", 1999),
        ("118", 48.00, 55, 90, "Jazz", 1962, "US", "Vinyl, LP", "118", 1962),
        ("119", 16.50, 160, 600, "Rock", 1987, "US", "Vinyl, LP", "119", 1987),
        ("120", 26.00, 88, 240, "Soul", 1973, "US", "Vinyl, LP", "120", 1973),
        ("121", 13.00, 190, 750, "Rock", 1993, "US", "Vinyl, LP", "121", 1993),
        ("122", 38.00, 75, 140, "Electronic", 2001, "UK", "Vinyl, 12", "122", 2001),
        ("123", 21.00, 125, 420, "Rock", 1976, "US", "Vinyl, LP", "123", 1976),
        ("124", 10.00, 280, 1200, "Pop", 1989, "US", "Vinyl, LP", "124", 1989),
    ]
    for rid, median, wants, haves, genre, year, country, fmt, mid, my in demo:
        mdb.upsert(
            rid,
            {
                "lowest_price": {"value": median * 0.9},
                "median_price": {"value": median},
                "num_for_sale": max(1, wants // 40),
            },
        )
        ratio = wants / haves if haves else 0
        genres_j = json.dumps([genre], separators=(",", ":"))
        fdb.upsert_row(
            {
                "release_id": rid,
                "master_id": mid,
                "want_count": wants,
                "have_count": haves,
                "want_have_ratio": ratio,
                "genre": genre,
                "style": None,
                "decade": (year // 10) * 10,
                "year": year,
                "country": country,
                "label_tier": 0,
                "is_original_pressing": 1 if year == my else 0,
                "is_colored_vinyl": 0,
                "is_picture_disc": 0,
                "is_promo": 0,
                "format_desc": fmt,
                "artists_json": None,
                "labels_json": None,
                "genres_json": genres_j,
                "styles_json": None,
                "formats_json": None,
            }
        )
    print(f"Seeded {len(demo)} releases -> {mp} and {fs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
