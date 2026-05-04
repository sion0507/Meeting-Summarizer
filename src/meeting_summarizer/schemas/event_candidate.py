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
