"""Great Expectations validation for VinylIQ monitoring Parquet exports."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import uuid

import pandas as pd
import great_expectations as gx
from great_expectations.expectations import (
    ExpectColumnValuesToBeBetween,
    ExpectColumnValuesToBeUnique,
    ExpectColumnValuesToNotBeNull,
    ExpectTableColumnsToMatchOrderedList,
)

from price_estimator.src.storage.feature_store import RELEASES_FEATURES_COLUMNS


def _batch_for(df: pd.DataFrame):
    context = gx.get_context(mode="ephemeral")
    name = f"pandas_{uuid.uuid4().hex[:12]}"
    src = context.data_sources.add_pandas(name=name)
    return src.read_dataframe(df, asset_name="releases")


def _expectation_type_from_result(result) -> str:
    cfg = result.expectation_config
    if isinstance(cfg, dict):
        return str(cfg.get("type", "unknown"))
    t = getattr(cfg, "type", None)
    return str(t) if t is not None else "unknown"


def expect_schema_ordered(df: pd.DataFrame, columns: list[str]) -> bool:
    batch = _batch_for(df)
    res = batch.validate(
        ExpectTableColumnsToMatchOrderedList(column_list=columns),
    )
    return bool(res.success)


def expect_schema_releases_features(df: pd.DataFrame) -> bool:
    """Full canonical column list + ordered match."""
    return expect_schema_ordered(df, list(RELEASES_FEATURES_COLUMNS))


@dataclass
class IntegrityResult:
    success: bool
    failed_expectations: list[str]


def validate_integrity(
    df: pd.DataFrame, *, label_tier_max: int = 10
) -> IntegrityResult:
    """Bounds: year range, unique release_id, label_tier, null caps."""
    failed: list[str] = []
    batch = _batch_for(df)
    expectations = [
        ExpectColumnValuesToNotBeNull(column="release_id"),
        ExpectColumnValuesToBeUnique(column="release_id"),
        ExpectColumnValuesToBeBetween(
            column="year",
            min_value=1900,
            max_value=2035,
            mostly=0.98,
        ),
        ExpectColumnValuesToBeBetween(
            column="label_tier",
            min_value=0,
            max_value=label_tier_max,
            mostly=1.0,
        ),
    ]
    for exp in expectations:
        r = batch.validate(exp)
        if not r.success:
            failed.append(_expectation_type_from_result(r))

    y = pd.to_numeric(df["year"], errors="coerce")
    dec = pd.to_numeric(df["decade"], errors="coerce")
    mask = y.notna() & dec.notna()
    bad_decade = int((dec[mask] != (y[mask].astype(int) // 10) * 10).sum())
    if bad_decade > 0:
        failed.append(f"decade_year_mismatch:{bad_decade}")

    genre_nf = (
        float(df["genre"].isna().mean()) if "genre" in df.columns else 0.0
    )
    country_nf = (
        float(df["country"].isna().mean()) if "country" in df.columns else 0.0
    )
    if genre_nf > 0.5 or country_nf > 0.5:
        failed.append("high_null_genre_or_country")

    ok = len(failed) == 0
    return IntegrityResult(success=ok, failed_expectations=failed)


def validate_parquet_file(
    path: Path, *, integrity: bool = True
) -> IntegrityResult:
    df = pd.read_parquet(path)
    cols_ok = expect_schema_releases_features(df)
    if not cols_ok:
        failed = ["expect_table_columns_to_match_ordered_list"]
        return IntegrityResult(False, failed)
    if not integrity:
        return IntegrityResult(True, [])
    return validate_integrity(df)
