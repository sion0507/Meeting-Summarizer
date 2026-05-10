"""Candidate linking and FAISS vector storage utilities."""

from meeting_summarizer.linking.faiss_store import (
    CandidateVectorRecord,
    FaissCandidateStore,
    FaissStoreError,
    FaissStoreMetadata,
    SearchResult,
)

__all__ = [
    "CandidateVectorRecord",
    "FaissCandidateStore",
    "FaissStoreError",
    "FaissStoreMetadata",
    "SearchResult",
]
