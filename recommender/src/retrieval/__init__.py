"""
Album metadata helpers for reranker feature generation.
"""

from .candidates import (
    RetrievalMetadata,
    build_retrieval_metadata,
)

__all__ = [
    "RetrievalMetadata",
    "build_retrieval_metadata",
]
