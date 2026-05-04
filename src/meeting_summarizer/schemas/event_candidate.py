"""Event candidate schema used by extraction stage."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class EventCandidate:
    """A candidate event extracted from a single segment."""

    candidate_id: str
    meeting_id: str
    segment_id: str
    source_file: str

    title: str
    summary: str

    occurred_at: str | None
    actors: list[str]

    problem: str | None
    discussion: str | None
    action: str | None
    result: str | None

    status: str
    evidence_text: str
    keywords: list[str]
    embedding_text: str

    def __post_init__(self) -> None:
        for name in ("candidate_id", "meeting_id", "segment_id", "source_file", "title", "summary", "status", "evidence_text", "embedding_text"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"EventCandidate.{name} must be a non-empty string.")
        if not isinstance(self.actors, list):
            raise ValueError("EventCandidate.actors must be a list[str].")
        if not isinstance(self.keywords, list):
            raise ValueError("EventCandidate.keywords must be a list[str].")
