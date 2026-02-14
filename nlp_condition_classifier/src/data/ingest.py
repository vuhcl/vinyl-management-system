"""
Data ingestion for the NLP condition classifier.

- **Discogs**: Uses the shared Discogs API (discogs_api) to fetch listing/seller notes
  and condition labels when use_api is enabled.
- **CSV fallback**: Labeled data (item_id, seller_notes, sleeve_condition, media_condition, ...)
  from paths.raw_data for supervised training.
"""
from pathlib import Path
from typing import Any

import pandas as pd

# Condition grades used by Discogs and this project
CONDITION_GRADES = [
    "Mint",
    "Near Mint",
    "Very Good Plus",
    "Very Good",
    "Good",
]


def _normalize_condition(s: str | None) -> str | None:
    """Normalize condition string to one of CONDITION_GRADES."""
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return None
    s = str(s).strip()
    if not s:
        return None
    for grade in CONDITION_GRADES:
        if grade.lower() == s.lower():
            return grade
    return None


def load_labeled_from_csv(
    path: Path | None = None,
    data_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Load labeled condition data from CSV.

    Expected columns (names case-insensitive):
    - item_id (or id)
    - seller_notes (or notes, description)
    - sleeve_condition
    - media_condition
    Optional: artist, genre, release_year
    """
    if path is None and data_dir is None:
        return pd.DataFrame(
            columns=[
                "item_id",
                "seller_notes",
                "sleeve_condition",
                "media_condition",
            ]
        )
    p = path if path is not None else (data_dir / "condition_labeled.csv" if data_dir else None)
    if p is None or not p.exists():
        return pd.DataFrame(
            columns=[
                "item_id",
                "seller_notes",
                "sleeve_condition",
                "media_condition",
            ]
        )
    df = pd.read_csv(p)
    df = df.rename(columns={c: c.lower().replace(" ", "_") for c in df.columns})
    # Normalize column names
    col_map = {}
    if "id" in df.columns and "item_id" not in df.columns:
        col_map["id"] = "item_id"
    for alias in ["notes", "description"]:
        if alias in df.columns and "seller_notes" not in df.columns:
            col_map[alias] = "seller_notes"
    df = df.rename(columns=col_map)
    # Normalize condition labels
    for col in ["sleeve_condition", "media_condition"]:
        if col in df.columns:
            df[col] = df[col].map(_normalize_condition)
    df = df.dropna(subset=["item_id", "seller_notes"])
    df["item_id"] = df["item_id"].astype(str)
    return df


def fetch_listings_from_discogs(
    discogs_token: str | None = None,
    release_id: str | None = None,
    max_listings: int = 500,
) -> pd.DataFrame:
    """
    Fetch marketplace listings from Discogs API (seller notes + condition).

    Uses the shared discogs_api client. When release_id is set, fetches listings
    for that release; otherwise can be extended for user inventory/listings.
    """
    try:
        import sys
        from pathlib import Path as P
        root = P(__file__).resolve().parents[3]
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from discogs_api.client import DiscogsClient
    except ImportError:
        return pd.DataFrame(
            columns=["item_id", "seller_notes", "sleeve_condition", "media_condition"]
        )
    if not discogs_token:
        return pd.DataFrame(
            columns=["item_id", "seller_notes", "sleeve_condition", "media_condition"]
        )
    client = DiscogsClient(user_token=discogs_token)
    rows: list[dict[str, Any]] = []
    try:
        if release_id:
            data = client._get(f"/marketplace/listings", {"release_id": release_id, "per_page": 100})
            if isinstance(data, dict):
                listings = data.get("listings", [])
            else:
                listings = data if isinstance(data, list) else []
        else:
            # User's inventory: /users/{username}/inventory
            username = client.get_username()
            if not username:
                return pd.DataFrame(
                    columns=["item_id", "seller_notes", "sleeve_condition", "media_condition"]
                )
            listings = client._paginate(
                f"/users/{username}/inventory",
                key="listings",
                per_page=100,
            )
        for i, listing in enumerate(listings):
            if i >= max_listings:
                break
            item_id = str(listing.get("id", ""))
            notes = listing.get("condition", "") or ""
            sleeve = listing.get("sleeve_condition") or listing.get("condition")
            media = listing.get("media_condition") or listing.get("condition")
            sleeve_n = _normalize_condition(sleeve)
            media_n = _normalize_condition(media)
            if item_id and (sleeve_n or media_n):
                rows.append({
                    "item_id": item_id,
                    "seller_notes": notes,
                    "sleeve_condition": sleeve_n,
                    "media_condition": media_n,
                })
    except Exception:
        pass
    if not rows:
        return pd.DataFrame(
            columns=["item_id", "seller_notes", "sleeve_condition", "media_condition"]
        )
    return pd.DataFrame(rows)


def load_labeled_condition_data(
    path: Path | None = None,
    data_dir: Path | None = None,
    *,
    from_discogs: bool = False,
    discogs_token: str | None = None,
) -> pd.DataFrame:
    """
    Load labeled condition data. Prefer Discogs API when from_discogs and token set.
    Otherwise load from CSV (path or data_dir/condition_labeled.csv).
    """
    if from_discogs and discogs_token:
        return fetch_listings_from_discogs(discogs_token=discogs_token)
    return load_labeled_from_csv(path=path, data_dir=data_dir)
