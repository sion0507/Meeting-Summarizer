"""Final merged event case schemas."""

from __future__ import annotations

from dataclasses import dataclass

_ALLOWED_TIMELINE_STAGES = {
    "occurrence",
    "discussion",
    "action",
    "result",
    "status_update",
    "unknown",
}


@dataclass(slots=True)
class EvidenceSpan:
    """Traceable evidence snippet used by an event case."""

    evidence_id: str
    candidate_id: str
    meeting_id: str
    segment_id: str
    source_file: str
    text: str

    def __post_init__(self) -> None:
        for name in (
            "evidence_id",
            "candidate_id",
            "meeting_id",
            "segment_id",
            "source_file",
            "text",
        ):
            if not getattr(self, name).strip():
                raise ValueError(f"EvidenceSpan.{name} must not be empty.")


@dataclass(slots=True)
class TimelineItem:
    """One timeline row for the final event case."""

    timeline_id: str
    date: str | None
    order: int
    stage: str
    description: str
    evidence_ids: list[str]

    def __post_init__(self) -> None:
        if not self.timeline_id.strip():
            raise ValueError("TimelineItem.timeline_id must not be empty.")
        if self.order < 0:
            raise ValueError("TimelineItem.order must be >= 0.")
        if self.stage not in _ALLOWED_TIMELINE_STAGES:
            raise ValueError(
                f"TimelineItem.stage must be one of {_ALLOWED_TIMELINE_STAGES}."
            )
        if not self.description.strip():
            raise ValueError("TimelineItem.description must not be empty.")
        if not isinstance(self.evidence_ids, list):
            raise ValueError("TimelineItem.evidence_ids must be a list[str].")


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

    def __post_init__(self) -> None:
        for name in ("case_id", "title", "summary", "status"):
            if not getattr(self, name).strip():
                raise ValueError(f"EventCase.{name} must not be empty.")
        if not self.candidate_ids:
            raise ValueError("EventCase.candidate_ids must include at least one candidate.")
        if not isinstance(self.evidence, list):
            raise ValueError("EventCase.evidence must be a list[EvidenceSpan].")
        if not isinstance(self.timeline, list):
            raise ValueError("EventCase.timeline must be a list[TimelineItem].")
