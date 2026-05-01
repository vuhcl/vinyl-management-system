-- Cloud SQL schema for VinylIQ price-api (releases_features + marketplace_stats).
-- Apply with: psql "$DATABASE_URL" -f k8s/demo/schema.sql

CREATE TABLE IF NOT EXISTS releases_features (
    release_id           TEXT PRIMARY KEY,
    master_id            TEXT,
    genre                TEXT,
    style                TEXT,
    decade               INTEGER,
    year                 INTEGER,
    country              TEXT,
    label_tier           INTEGER,
    is_original_pressing INTEGER,
    is_colored_vinyl     INTEGER,
    is_picture_disc      INTEGER,
    is_promo             INTEGER,
    format_desc          TEXT,
    artists_json         TEXT,
    labels_json          TEXT,
    genres_json          TEXT,
    styles_json          TEXT,
    formats_json         TEXT
);

CREATE TABLE IF NOT EXISTS marketplace_stats (
    release_id              TEXT PRIMARY KEY,
    fetched_at              TEXT NOT NULL,
    num_for_sale            INTEGER,
    blocked_from_sale       INTEGER,
    raw_json                TEXT NOT NULL,
    release_raw_json        TEXT,
    price_suggestions_json  TEXT,
    release_lowest_price    DOUBLE PRECISION,
    release_num_for_sale    INTEGER,
    community_want          INTEGER,
    community_have          INTEGER
);
