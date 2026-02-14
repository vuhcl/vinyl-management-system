"""
Album of the Year (albumoftheyear.org) scraped data – used by the recommender subproject.
Scrapers and scraped data live elsewhere; this module defines how to load that data
into a standard format (ratings, album metadata).
"""
from aoty.loader import load_ratings_from_scraped, load_album_metadata_from_scraped

__all__ = ["load_ratings_from_scraped", "load_album_metadata_from_scraped"]
