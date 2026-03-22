"""
Album of the Year (albumoftheyear.org) scraped data - used by the
recommender subproject.

Scrapers and scraped data live elsewhere; this module defines how to load
that data into a standard format (ratings, album metadata).
"""

from shared.aoty.loader import (
    load_album_metadata_from_scraped,
    load_ratings_from_scraped,
)
from shared.aoty.mongo_loader import (
    MongoConfig,
    load_album_metadata_from_mongo,
    load_ratings_from_mongo,
)

__all__ = [
    "load_ratings_from_scraped",
    "load_album_metadata_from_scraped",
    "load_ratings_from_mongo",
    "load_album_metadata_from_mongo",
    "MongoConfig",
]
