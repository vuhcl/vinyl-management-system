"""Great Expectations validation for VinylIQ monitoring Parquet exports."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import great_expectations as gx

from price_estimator.src.storage.feature_store import RELEASES_FEATURES_COLUMNS


def _validator_for(df: pd.DataFrame):
    context = gx.get_context(mode="ephemeral")
    # Fresh datasource name avoids collisions across validations in-process.
    import uuid

    name = f"pandas_{uuid.uuid4().hex[:12]}"
    src = context.sources.add_pandas(name=name)
    return src.read_dataframe(df, asset_name="releases")


def expect_schema_ordered(df: pd.DataFrame, columns: list[str]) -> bool:
    v = _validator_for(df)
    res = v.expect_table_columns_to_match_ordered_list(column_list=columns)
    return bool(res.success)


def expect_schema_releases_features(df: pd.DataFrame) -> bool:
    """Full canonical column list + ordered match."""
    return expect_schema_ordered(df, list(RELEASES_FEATURES_COLUMNS))


@dataclass
class IntegrityResult:
    success: bool
    failed_expectations: list[str]


def validate_integrity(df: pd.DataFrame, *, label_tier_max: int = 10) -> IntegrityResult:
    """Bounds: year range, unique release_id, label_tier, null caps."""
    failed: list[str] = []
    v = _validator_for(df)
    checks = [
        v.expect_column_values_to_not_be_null("release_id"),
        v.expect_column_values_to_be_unique("release_id"),
        v.expect_column_values_to_be_between(
            "year",
            min_value=1900,
            max_value=2035,
            mostly=0.98,
        ),
        v.expect_column_values_to_be_between(
            "label_tier",
            min_value=0,
            max_value=label_tier_max,
            mostly=1.0,
        ),
    ]
    for r in checks:
        if not r.success:
            failed.append(str(r.expectation_config.type))

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


def validate_parquet_file(path: Path, *, integrity: bool = True) -> IntegrityResult:
    df = pd.read_parquet(path)
    cols_ok = expect_schema_releases_features(df)
    if not cols_ok:
        failed = ["expect_table_columns_to_match_ordered_list"]
        return IntegrityResult(False, failed)
    if not integrity:
        return IntegrityResult(True, [])
    return validate_integrity(df)
