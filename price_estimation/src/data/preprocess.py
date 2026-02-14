"""
Preprocess price estimation data.

- Filter valid sale prices and dates
- Merge sales with metadata
- Optional: filter to last N years
- Normalize condition labels for encoding
"""
from pathlib import Path
from typing import Any

import pandas as pd


def preprocess_price_data(
    sales: pd.DataFrame,
    metadata: pd.DataFrame | None = None,
    *,
    min_price: float = 0.01,
    max_price: float = 10_000.0,
    max_years_back: int | None = 3,
    drop_na_condition: bool = False,
) -> pd.DataFrame:
    """
    Clean and merge sales with metadata. Optionally restrict to recent years.

    - Drops rows with sale_price outside [min_price, max_price]
    - Drops rows with invalid sale_date
    - If max_years_back is set, keeps only sales within that many years from latest date
    - Merges metadata on item_id (left join from sales)
    """
    df = sales.copy()
    if df.empty:
        return df

    df = df[df["sale_price"].between(min_price, max_price)]
    df = df.dropna(subset=["sale_date"])
    if df.empty:
        return df

    if max_years_back is not None:
        max_date = df["sale_date"].max()
        cutoff = max_date - pd.DateOffset(years=max_years_back)
        df = df[df["sale_date"] >= cutoff]

    if metadata is not None and not metadata.empty:
        meta_cols = [c for c in metadata.columns if c != "item_id"]
        df = df.merge(
            metadata[["item_id"] + meta_cols].drop_duplicates(subset=["item_id"]),
            on="item_id",
            how="left",
        )

    if drop_na_condition:
        for col in ["sleeve_condition", "media_condition"]:
            if col in df.columns:
                df = df.dropna(subset=[col])

    return df.reset_index(drop=True)
