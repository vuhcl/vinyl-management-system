"""
scripts/verify_api_connections.py

Verifies that Discogs and eBay API credentials work correctly
before running the full ingestion pipeline.

Tests:
  1. Discogs — authenticated request, rate limit headers
  2. Discogs — public seller inventory sample (1 page, 5 rows)
  3. eBay    — OAuth token fetch
  4. eBay    — Browse API search for trusted seller listings

Run with:
    .venv/bin/python scripts/verify_api_connections.py
"""

import base64
import json
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load environment variables from .env
# ---------------------------------------------------------------------------
env_path = Path(__file__).parent.parent / ".env"
if not env_path.exists():
    print(f"ERROR: .env file not found at {env_path}")
    print("Copy .env.template to .env and fill in your credentials.")
    sys.exit(1)

load_dotenv(env_path)

DISCOGS_TOKEN    = os.environ.get("DISCOGS_TOKEN")
EBAY_CLIENT_ID   = os.environ.get("EBAY_CLIENT_ID")
EBAY_CLIENT_SECRET = os.environ.get("EBAY_CLIENT_SECRET")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def print_section(title: str) -> None:
    print(f"\n{'=' * 55}")
    print(f"  {title}")
    print(f"{'=' * 55}")


def print_result(label: str, value, success: bool = True) -> None:
    icon = "✓" if success else "✗"
    print(f"  {icon}  {label}: {value}")


# ---------------------------------------------------------------------------
# Discogs verification
# ---------------------------------------------------------------------------
def verify_discogs() -> bool:
    print_section("DISCOGS API")

    if not DISCOGS_TOKEN or DISCOGS_TOKEN == "your_discogs_token_here":
        print("  ✗  DISCOGS_TOKEN not set in .env")
        return False

    session = requests.Session()
    session.headers.update({
        "Authorization": f"Discogs token={DISCOGS_TOKEN}",
        "User-Agent": "VinylCollectorAI/0.1 +https://github.com/user/vinyl_collector_ai",
    })

    # Test 1 — Identity check (confirms token is valid)
    print("\n  [1/2] Testing authentication...")
    try:
        resp = session.get(
            "https://api.discogs.com/oauth/identity",
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            print_result("Authenticated as", data.get("username", "unknown"))
            print_result(
                "Rate limit",
                f"{resp.headers.get('X-Discogs-Ratelimit-Remaining', '?')}/"
                f"{resp.headers.get('X-Discogs-Ratelimit', '?')} remaining",
            )
        else:
            print_result(
                "Auth failed", f"HTTP {resp.status_code} — {resp.text[:100]}",
                success=False
            )
            return False
    except Exception as e:
        print_result("Connection error", str(e), success=False)
        return False

    # Test 2 — Seller inventory (official API; /marketplace/search returns 404)
    test_seller = os.environ.get("DISCOGS_INVENTORY_TEST_SELLER", "rappcats")
    print(f"\n  [2/2] Testing seller inventory (user={test_seller!r})...")
    time.sleep(1)  # respect rate limit
    try:
        resp = session.get(
            f"https://api.discogs.com/users/{test_seller}/inventory",
            params={
                "status":   "For Sale",
                "per_page": 5,
                "page":     1,
            },
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            total    = data.get("pagination", {}).get("items", 0)
            listings = data.get("listings", [])
            print_result("Items in seller inventory (total)", f"{total:,}")
            print_result("Listings returned", len(listings))

            # Show a sample listing
            if listings:
                sample = listings[0]
                release = sample.get("release", {})
                print(f"\n  Sample listing:")
                print(f"    Artist:    {release.get('artist', 'N/A')}")
                print(f"    Title:     {release.get('title', 'N/A')}")
                print(f"    Sleeve:    {sample.get('sleeve_condition', 'N/A')}")
                print(f"    Media:     {sample.get('condition', 'N/A')}")
                notes = sample.get('comments', '')
                print(f"    Notes:     {notes[:80]}{'...' if len(notes) > 80 else ''}")
        else:
            print_result(
                "Inventory request failed",
                f"HTTP {resp.status_code}",
                success=False,
            )
            return False
    except Exception as e:
        print_result("Connection error", str(e), success=False)
        return False

    print("\n  ✓  Discogs API: OK")
    return True


# ---------------------------------------------------------------------------
# eBay verification
# ---------------------------------------------------------------------------
def verify_ebay() -> bool:
    print_section("EBAY API")

    if not EBAY_CLIENT_ID or EBAY_CLIENT_ID == "your_ebay_client_id_here":
        print("  ✗  EBAY_CLIENT_ID not set in .env")
        return False

    if not EBAY_CLIENT_SECRET or EBAY_CLIENT_SECRET == "your_ebay_client_secret_here":
        print("  ✗  EBAY_CLIENT_SECRET not set in .env")
        return False

    # Test 1 — OAuth token fetch
    print("\n  [1/2] Fetching OAuth token...")
    try:
        credentials = base64.b64encode(
            f"{EBAY_CLIENT_ID}:{EBAY_CLIENT_SECRET}".encode()
        ).decode()

        resp = requests.post(
            "https://api.ebay.com/identity/v1/oauth2/token",
            headers={
                "Authorization": f"Basic {credentials}",
                "Content-Type":  "application/x-www-form-urlencoded",
            },
            data={
                "grant_type": "client_credentials",
                "scope":      "https://api.ebay.com/oauth/api_scope",
            },
            timeout=15,
        )

        if resp.status_code == 200:
            token_data   = resp.json()
            access_token = token_data["access_token"]
            expires_in   = token_data.get("expires_in", 0)
            print_result("Token acquired", f"expires in {expires_in}s")
        else:
            print_result(
                "Token fetch failed",
                f"HTTP {resp.status_code} — {resp.text[:200]}",
                success=False,
            )
            return False
    except Exception as e:
        print_result("Connection error", str(e), success=False)
        return False

    # Test 2 — Browse API search for Face Records
    print("\n  [2/2] Testing Browse API (facerecords seller)...")
    try:
        session = requests.Session()
        session.headers.update({
            "Authorization":           f"Bearer {access_token}",
            "X-EBAY-C-MARKETPLACE-ID": "EBAY_US",
            "Content-Type":            "application/json",
        })

        resp = session.get(
            "https://api.ebay.com/buy/browse/v1/item_summary/search",
            params={
                "q":      "vinyl record",
                "filter": "sellers:{facerecords},itemLocationCountry:JP",
                "limit":  3,
            },
            timeout=15,
        )

        if resp.status_code == 200:
            data  = resp.json()
            total = data.get("total", 0)
            items = data.get("itemSummaries", [])
            print_result("Total facerecords listings", f"{total:,}")
            print_result("Items returned", len(items))

            if items:
                item      = items[0]
                aspects   = {
                    a["name"]: a["value"]
                    for a in item.get("localizedAspects", [])
                    if "name" in a and "value" in a
                }
                print(f"\n  Sample listing (facerecords):")
                print(f"    Title:         {item.get('title', 'N/A')[:60]}")
                print(f"    Sleeve Grading: {aspects.get('Sleeve Grading', 'N/A')}")
                print(f"    Record Grading: {aspects.get('Record Grading', 'N/A')}")
                print(f"    OBI Grading:    {aspects.get('OBI_Grading', 'N/A')}")
                print(f"    Seller Notes:   {item.get('shortDescription', 'N/A')}")
        elif resp.status_code == 404:
            print_result(
                "No listings found",
                "facerecords may have no active vinyl listings",
                success=False,
            )
        else:
            print_result(
                "Search failed",
                f"HTTP {resp.status_code} — {resp.text[:200]}",
                success=False,
            )
            return False

    except Exception as e:
        print_result("Connection error", str(e), success=False)
        return False

    print("\n  ✓  eBay API: OK")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("\nVinyl Collector AI — API Connection Verification")
    print("=" * 55)

    discogs_ok = verify_discogs()
    ebay_ok    = verify_ebay()

    print_section("SUMMARY")
    print_result("Discogs API", "OK" if discogs_ok else "FAILED", discogs_ok)
    print_result("eBay API",    "OK" if ebay_ok    else "FAILED", ebay_ok)

    if discogs_ok and ebay_ok:
        print("\n  All APIs verified. Ready to run ingestion pipeline.")
        print("  Next step: .venv/bin/python -m grader.src.pipeline train --baseline-only")
    else:
        print("\n  Fix the failing APIs before running ingestion.")
        sys.exit(1)


if __name__ == "__main__":
    main()
