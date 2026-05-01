"""Smoke-import Postgres storage backends (no live DB required)."""


def test_postgres_storage_importable():
    from price_estimator.src.storage.postgres_feature_store import PostgresFeatureStore
    from price_estimator.src.storage.postgres_marketplace_stats import PostgresMarketplaceStats

    assert PostgresFeatureStore.__name__ == "PostgresFeatureStore"
    assert PostgresMarketplaceStats.__name__ == "PostgresMarketplaceStats"
