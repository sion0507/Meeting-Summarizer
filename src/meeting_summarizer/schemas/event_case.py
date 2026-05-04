"""Final merged event case schemas."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class EvidenceSpan:
    """Traceable evidence snippet used by an event case."""

    evidence_id: str
    candidate_id: str
    meeting_id: str
    segment_id: str
    source_file: str
    text: str


@dataclass(slots=True)
class TimelineItem:
    """One timeline row for the final event case."""

    timeline_id: str
    date: str | None
    order: int
    stage: str
    description: str
    evidence_ids: list[str]


@dataclass(slots=True)
class EventCase:
    """Merged event record built from one or more candidates."""

    case_id: str
    title: str
    summary: str

    candidate_ids: list[str]
    related_meeting_ids: list[str]

    first_occurred_at: str | None
    actors: list[str]

    occurrence: str | None
    discussion: str | None
    actions: list[str]
    result: str | None

    status: str
    remaining_issues: list[str]
    evidence: list[EvidenceSpan]
    timeline: list[TimelineItem]
