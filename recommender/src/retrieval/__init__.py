"""
Two-stage retrieval: narrow candidate items before CF ranking.
"""

from .candidates import (
    CandidateRetrievalConfig,
    RetrievalMetadata,
    build_retrieval_metadata,
    candidate_item_indices_for_user,
    retrieval_config_from_dict,
)

__all__ = [
    "CandidateRetrievalConfig",
    "RetrievalMetadata",
    "build_retrieval_metadata",
    "candidate_item_indices_for_user",
    "retrieval_config_from_dict",
]
