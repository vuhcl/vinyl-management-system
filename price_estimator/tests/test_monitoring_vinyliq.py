"""VinylIQ data and model monitoring (GE + drift_stats)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from price_estimator.src.monitoring.drift_stats import (
    binary_two_sample_psi_and_ztest,
    categorical_psi_topk,
    chi_square_independence,
    drift_numeric_column,
    drift_predictions,
)
from price_estimator.src.monitoring.export_features import export_releases_features_to_parquet
from price_estimator.src.monitoring.ge_suite import (
    expect_schema_releases_features,
    validate_integrity,
)
from price_estimator.src.monitoring.thresholds import load_thresholds
from price_estimator.src.storage.feature_store import (
    RELEASES_FEATURES_COLUMNS,
    FeatureStoreDB,
)

pytestmark = pytest.mark.monitoring


def _rng() -> np.random.Generator:
    return np.random.default_rng(42)


def _sample_row(i: int, rng: np.random.Generator) -> dict:
    year = int(rng.integers(1965, 2021))
    decade = (year // 10) * 10
    return {
        "release_id": f"rel_{i:06d}",
        "master_id": f"m{i % 5000}",
        "genre": str(rng.choice(["Rock", "Jazz", "Electronic", "Hip Hop", "Classical"])),
        "style": str(rng.choice(["Alternative", "Soul", "Ambient", ""])),
        "decade": decade,
        "year": year,
        "country": str(rng.choice(["US", "UK", "DE", "JP", "FR"])),
        "label_tier": int(rng.integers(0, 4)),
        "is_original_pressing": int(rng.choice([0, 1])),
        "is_colored_vinyl": int(rng.choice([0, 1], p=[0.85, 0.15])),
        "is_picture_disc": int(rng.choice([0, 1], p=[0.95, 0.05])),
        "is_promo": int(rng.choice([0, 1], p=[0.9, 0.1])),
        "format_desc": "Vinyl, LP",
        "artists_json": "[]",
        "labels_json": "[]",
        "genres_json": "[]",
        "styles_json": "[]",
        "formats_json": "[]",
    }


def make_releases_features_df(n: int = 250, *, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = [_sample_row(i, rng) for i in range(n)]
    return pd.DataFrame(rows)


@pytest.fixture
def reference_df() -> pd.DataFrame:
    return make_releases_features_df(250, seed=42)


@pytest.fixture
def thresholds() -> dict:
    return load_thresholds()


def test_monitoring_ge_schema_releases_features(reference_df: pd.DataFrame) -> None:
    assert expect_schema_releases_features(reference_df)


def test_monitoring_ge_integrity_bounds(reference_df: pd.DataFrame, thresholds: dict) -> None:
    lt_max = int(thresholds["integrality"]["label_tier_max"])
    res = validate_integrity(reference_df, label_tier_max=lt_max)
    assert res.success, res.failed_expectations


def test_monitoring_ge_rejects_corrupt_parquet(reference_df: pd.DataFrame) -> None:
    bad = reference_df.drop(columns=["genre"])
    assert not expect_schema_releases_features(bad)


def test_monitoring_no_drift_reference_vs_reference(
    reference_df: pd.DataFrame, thresholds: dict
) -> None:
    ref = reference_df
    cur = reference_df.copy()
    num = thresholds["numeric"]
    cat = thresholds["categorical"]
    bin_th = thresholds["binary"]

    for col, kind in (
        ("year", "year"),
        ("decade", "decade"),
        ("label_tier", "label_tier"),
    ):
        d = drift_numeric_column(ref, cur, col, kind=kind)
        assert d.psi <= float(num[col]["max_psi"])
        assert d.ks_pvalue >= float(num[col]["min_ks_pvalue"])

    k = int(cat["genre"]["top_k"])
    psi_g = categorical_psi_topk(ref["genre"], cur["genre"], k=k)
    chi2_p = chi_square_independence(ref["genre"], cur["genre"], k=k)[1]
    assert psi_g <= float(cat["genre"]["max_psi_topk"])
    assert chi2_p >= float(cat["genre"]["min_chi2_pvalue"])

    psi_co = categorical_psi_topk(ref["country"], cur["country"], k=k)
    chi2_co = chi_square_independence(ref["country"], cur["country"], k=k)[1]
    assert psi_co <= float(cat["country"]["max_psi_topk"])
    assert chi2_co >= float(cat["country"]["min_chi2_pvalue"])

    for bcol in ("is_promo", "is_colored_vinyl"):
        psi_b, z_p = binary_two_sample_psi_and_ztest(ref[bcol], cur[bcol])
        assert psi_b <= float(bin_th[bcol]["max_psi"])
        assert z_p >= float(bin_th[bcol]["min_z_pvalue"])


def test_monitoring_detects_drift_challenge_fixture(
    reference_df: pd.DataFrame, thresholds: dict
) -> None:
    ref = reference_df
    ch = reference_df.copy()
    # Strong temporal shift + genre concentration
    ch["year"] = (ch["year"].astype(int) + 14).clip(1900, 2035)
    ch["decade"] = (ch["year"] // 10) * 10
    ch["genre"] = "ObscureGenre"

    cdetect = thresholds["challenge_detect"]
    dy = drift_numeric_column(ref, ch, "year", kind="year")
    assert dy.psi >= float(cdetect["numeric"]["min_psi"])
    assert dy.ks_pvalue <= float(cdetect["numeric"]["max_ks_pvalue"])

    k = int(thresholds["categorical"]["genre"]["top_k"])
    chi2_p = chi_square_independence(ref["genre"], ch["genre"], k=k)[1]
    psi_g = categorical_psi_topk(ref["genre"], ch["genre"], k=k)
    assert chi2_p <= float(cdetect["categorical"]["max_chi2_pvalue"])
    assert psi_g >= float(cdetect["categorical"]["min_psi_topk"])


def test_monitoring_sqlite_export_columns_match_canonical(
    tmp_path, reference_df: pd.DataFrame
) -> None:
    db_path = tmp_path / "fs.sqlite"
    pq_path = tmp_path / "out.parquet"
    fs = FeatureStoreDB(db_path)
    for rec in reference_df.to_dict(orient="records"):
        fs.upsert_row(rec)
    n = export_releases_features_to_parquet(db_path, pq_path)
    assert n == len(reference_df)
    back = pd.read_parquet(pq_path)
    assert list(back.columns) == list(RELEASES_FEATURES_COLUMNS)


def test_monitoring_prediction_distribution_model_outputs(thresholds: dict) -> None:
    rng = _rng()
    n = 250
    rid = [f"rel_{i:06d}" for i in range(n)]
    base = rng.normal(2.5, 0.4, size=n)
    ref = pd.DataFrame({"release_id": rid, "pred_log1p_usd": base})
    light = pd.DataFrame({"release_id": rid, "pred_log1p_usd": base + rng.normal(0, 0.001, size=n)})
    shifted = pd.DataFrame({"release_id": rid, "pred_log1p_usd": base + 0.55})

    num = thresholds["numeric"]["pred_log1p_usd"]
    ok = drift_predictions(ref["pred_log1p_usd"], light["pred_log1p_usd"])
    assert ok.psi <= float(num["max_psi"]) and ok.ks_pvalue >= float(num["min_ks_pvalue"])

    bad = drift_predictions(ref["pred_log1p_usd"], shifted["pred_log1p_usd"])
    cd = thresholds["challenge_detect"]["model_shift"]
    assert bad.psi >= float(cd["min_psi"])
    assert bad.ks_pvalue <= float(cd["max_ks_pvalue"])


def test_monitoring_threshold_yaml_loads(thresholds: dict) -> None:
    assert "year" in thresholds["numeric"]
    assert "genre" in thresholds["categorical"]
    assert thresholds["numeric"]["pred_log1p_usd"]["max_psi"] == pytest.approx(0.10)


def test_monitoring_evidently_export_smoke(tmp_path, reference_df: pd.DataFrame) -> None:
    from price_estimator.src.monitoring.evidently_export import write_data_drift_html

    out = tmp_path / "drift.html"
    write_data_drift_html(reference_df, reference_df.copy(), out)
    assert out.is_file() and out.stat().st_size > 100
