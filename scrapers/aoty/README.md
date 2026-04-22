# AOTY scrapers (`scrapers/aoty/`)

This directory is the **canonical home** for **Album of the Year** (albumoftheyear.org) scraper scripts you run locally. Nothing here is required for CI or tests; the recommender only needs the **CSV output** on disk.

## Purpose

Provide **`ratings.csv`** and **`albums.csv`** so the recommender can build content features and optional user–item signals. See [recommender/README.md](../../recommender/README.md) → *Data sources* for how ingest picks up these files.

## Where to put scripts

- Place your scrapers here (e.g. Botasaurus-based jobs). They are typically **personal tools** and may stay **untracked** in git.
- For a reference pattern used elsewhere in the monorepo, Discogs marketplace scraping uses Botasaurus under **`price_estimator/scripts/`** (`collect_sale_history_botasaurus.py`, `collect_discogs_search_release_ids_botasaurus.py`). AOTY scraping is separate and lives here so it stays next to the recommender’s data contract without pulling AOTY deps into `vinyl-price-estimator`.

## Output contract

Write files into the directory set by **`aoty_scraped.dir`** in [`recommender/configs/base.yaml`](../../recommender/configs/base.yaml) (common choice: `recommender/data/aoty_scraped/`).

| File | Required columns |
|------|-------------------|
| **`ratings.csv`** | `user_id`, `album_id`, `rating` |
| **`albums.csv`** | `album_id`, `artist`, `genre`, `year`, `avg_rating` |

If `aoty_scraped.dir` is unset, the recommender can fall back to **`ratings.csv`** / **`albums.csv`** in repo-root **`data/raw/`** instead.

## Wiring

1. Run your scraper(s) and emit the two CSVs above into your chosen directory.
2. Set `aoty_scraped.dir` in `recommender/configs/base.yaml` to that path (and optional `ratings_file` / `albums_file` overrides if you use non-default names).
3. Run `python -m recommender.pipeline` (see [recommender/README.md](../../recommender/README.md)).

Loader used by the pipeline: **`shared.aoty`** (`load_ratings_from_scraped`, `load_album_metadata_from_scraped`).
