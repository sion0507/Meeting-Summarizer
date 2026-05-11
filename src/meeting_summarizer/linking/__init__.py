"""Candidate linking and FAISS vector storage utilities."""

from meeting_summarizer.linking.faiss_store import (
    CandidateVectorRecord,
    FaissCandidateStore,
    FaissStoreError,
    FaissStoreMetadata,
    SearchResult,
)
from meeting_summarizer.linking.grouping import (
    CandidateGroup,
    CandidateGroupingError,
    SimilarCandidatePair,
    group_and_save_candidates,
    group_candidates,
    save_candidate_groups,
)

__all__ = [
    "CandidateGroup",
    "CandidateGroupingError",
    "SimilarCandidatePair",
    "group_and_save_candidates",
    "group_candidates",
    "save_candidate_groups",
    "CandidateVectorRecord",
    "FaissCandidateStore",
    "FaissStoreError",
    "FaissStoreMetadata",
    "SearchResult",
]
