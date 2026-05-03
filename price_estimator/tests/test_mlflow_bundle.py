"""Unit tests for MLflow champion bundle resolution (no live server)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import mlflow
import pytest

import mlflow.tracking

from price_estimator.src.inference.mlflow_bundle import (
    _normalize_runs_bundle_uri,
    download_vinyliq_bundle_to_cache,
    resolve_downloaded_vinyliq_bundle_root,
    resolve_vinyliq_bundle_artifact_uri,
)


def test_normalize_runs_run_id_only_appends_artifacts() -> None:
    assert _normalize_runs_bundle_uri("runs:/abc") == "runs:/abc/vinyliq_artifacts"


def test_normalize_runs_swaps_pyfunc_suffix() -> None:
    assert (
        _normalize_runs_bundle_uri("runs:/run1/vinyliq_model")
        == "runs:/run1/vinyliq_artifacts"
    )


def test_normalize_runs_explicit_artifacts_unchanged() -> None:
    u = "runs:/z/vinyliq_artifacts"
    assert _normalize_runs_bundle_uri(u) == u


def test_resolve_models_alias_maps_pyfunc_source(monkeypatch: pytest.MonkeyPatch) -> None:
    class MV:
        name = "VinylIQPrice"
        version = "3"
        source = "runs:/rid999/vinyliq_model"

    class FakeClient:
        def get_model_version_by_alias(self, name: str, alias: str) -> MV:
            assert name == "VinylIQPrice"
            assert alias == "production"
            return MV()

    monkeypatch.setattr(
        mlflow.tracking,
        "MlflowClient",
        lambda: FakeClient(),
    )
    monkeypatch.setattr(mlflow, "set_tracking_uri", lambda *a, **k: None)
    au, stamp = resolve_vinyliq_bundle_artifact_uri(
        "models:/VinylIQPrice@production",
        tracking_uri="sqlite:///tmp/mlflow.db",
    )
    assert au == "runs:/rid999/vinyliq_artifacts"
    assert "VinylIQPrice:3" in stamp


def test_resolve_models_version(monkeypatch: pytest.MonkeyPatch) -> None:
    class MV:
        name = "VinylIQPrice"
        version = "12"
        source = "runs:/aa/vinyliq_model"

    class FakeClient:
        def get_model_version(self, name: str, version: str) -> MV:
            assert (name, version) == ("VinylIQPrice", "12")
            return MV()

    monkeypatch.setattr(
        mlflow.tracking,
        "MlflowClient",
        lambda: FakeClient(),
    )
    monkeypatch.setattr(mlflow, "set_tracking_uri", lambda *a, **k: None)
    au, _ = resolve_vinyliq_bundle_artifact_uri(
        "models:/VinylIQPrice/12",
        tracking_uri=None,
    )
    assert au == "runs:/aa/vinyliq_artifacts"


def test_resolve_models_latest_uses_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class MV:
        name = "VinylIQPrice"
        version = "9"
        source = "runs:/bb/vinyliq_model"

    class FakeClient:
        def search_model_versions(
            self,
            *,
            filter_string: str,
            order_by: list[str],
            max_results: int,
        ):
            assert filter_string == "name='VinylIQPrice'"
            assert order_by == ["version_number DESC"]
            assert max_results == 1
            return [MV()]

    monkeypatch.setattr(
        mlflow.tracking,
        "MlflowClient",
        lambda: FakeClient(),
    )
    monkeypatch.setattr(mlflow, "set_tracking_uri", lambda *a, **k: None)
    au, stamp = resolve_vinyliq_bundle_artifact_uri(
        "models:/VinylIQPrice/latest",
        tracking_uri=None,
    )
    assert au == "runs:/bb/vinyliq_artifacts"
    assert stamp.startswith("VinylIQPrice:9:")


def test_resolve_downloaded_bundle_peels_vinyliq_artifacts_subdir(
    tmp_path: Path,
) -> None:
    outer = tmp_path / "cache" / "vinyliq_bundle_xx"
    inner = outer / "vinyliq_artifacts"
    inner.mkdir(parents=True)
    (inner / "model_manifest.json").write_text("{}")
    assert resolve_downloaded_vinyliq_bundle_root(outer) == inner


def test_download_uses_cache_when_stamp_matches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        mlflow.tracking,
        "MlflowClient",
        lambda: MagicMock(),
    )
    monkeypatch.setattr(mlflow, "set_tracking_uri", lambda *a, **k: None)

    calls: list[str] = []

    def fake_download(*, artifact_uri: str, dst_path: str, tracking_uri: str | None):
        calls.append(artifact_uri)
        Path(dst_path).mkdir(parents=True, exist_ok=True)
        (Path(dst_path) / "model_manifest.json").write_text("{}")
        return dst_path

    monkeypatch.setattr(
        "mlflow.artifacts.download_artifacts",
        fake_download,
    )
    p1 = download_vinyliq_bundle_to_cache(
        model_uri="runs:/onlyrun",
        tracking_uri=None,
        cache_root=tmp_path,
        force=False,
    )
    p2 = download_vinyliq_bundle_to_cache(
        model_uri="runs:/onlyrun",
        tracking_uri=None,
        cache_root=tmp_path,
        force=False,
    )
    assert p1 == p2
    assert len(calls) == 1
    assert calls[0] == "runs:/onlyrun/vinyliq_artifacts"


def test_download_nested_artifact_returns_inner_root_for_inference(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(mlflow.tracking, "MlflowClient", lambda: MagicMock())
    monkeypatch.setattr(mlflow, "set_tracking_uri", lambda *a, **k: None)

    def fake_download(*, artifact_uri: str, dst_path: str, tracking_uri: str | None):
        d = Path(dst_path)
        d.mkdir(parents=True, exist_ok=True)
        nested = d / "vinyliq_artifacts"
        nested.mkdir()
        (nested / "model_manifest.json").write_text('{"backend":"xgboost"}')
        return str(d)

    monkeypatch.setattr("mlflow.artifacts.download_artifacts", fake_download)
    out = download_vinyliq_bundle_to_cache(
        model_uri="runs:/nestedrun",
        tracking_uri=None,
        cache_root=tmp_path,
        force=False,
    )
    assert out.name == "vinyliq_artifacts"
    assert (out / "model_manifest.json").is_file()


def test_bad_scheme_raises() -> None:
    with pytest.raises(ValueError, match="runs:/"):
        resolve_vinyliq_bundle_artifact_uri("s3://bucket/path", tracking_uri=None)
