"""LLM-backed merge of grouped event candidates into ``EventCase`` objects.

This stage owns only semantic synthesis for candidate groups.  Grouping has
already been decided by embeddings, so each group is sent to the LLM once (with
retry for malformed JSON) to produce one coherent event case.  Pipeline-owned
traceability fields are then normalized deterministically so evidence references
remain stable even when the model changes formatting.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from meeting_summarizer.agents.llm_client import (
    LLMClient,
    LLMClientError,
    validate_json_schema,
)
from meeting_summarizer.linking.grouping import CandidateGroup
from meeting_summarizer.schemas import EventCandidate, EventCase, EvidenceSpan
from meeting_summarizer.utils.ids import make_case_id, make_evidence_id
from meeting_summarizer.utils.logging import get_logger

DEFAULT_PROMPT_NAME = "event_case_merge"
DEFAULT_MAX_RESPONSE_ATTEMPTS = 2

LOGGER = get_logger(__name__)


class EventMergeError(RuntimeError):
    """Raised when a candidate group cannot be converted into an EventCase."""


class EventMerger:
    """Merge every :class:`CandidateGroup` into a validated :class:`EventCase`.

    The LLM is responsible for synthesizing title, summary, occurrence,
    discussion, actions, result, status, remaining issues, and initial timeline
    intent.  This class then enforces pipeline invariants that should not depend
    on model formatting choices:

    - one output case per input group;
    - deterministic ``case_id`` from grouped candidate IDs;
    - exact preservation of all group ``candidate_ids``;
    - evidence spans derived from the source candidates' ``evidence_text``;
    - related meeting IDs derived from the source candidates.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        *,
        prompt_name: str = DEFAULT_PROMPT_NAME,
        max_response_attempts: int = DEFAULT_MAX_RESPONSE_ATTEMPTS,
    ) -> None:
        if max_response_attempts < 1:
            raise ValueError("max_response_attempts must be >= 1.")
        self.llm_client = llm_client
        self.prompt_name = prompt_name
        self.max_response_attempts = max_response_attempts

    def merge_groups(
        self,
        groups: list[CandidateGroup],
        candidates: list[EventCandidate],
    ) -> list[EventCase]:
        """Merge all candidate groups, preserving group order."""

        if not groups:
            raise EventMergeError("No candidate groups were provided for event merging.")
        candidates_by_id = _index_candidates(candidates)
        return [self.merge_group(group, candidates_by_id) for group in groups]

    def merge_group(
        self,
        group: CandidateGroup,
        candidates_by_id: dict[str, EventCandidate],
    ) -> EventCase:
        """Merge one group into one schema-valid event case."""

        group_candidates = _resolve_group_candidates(group, candidates_by_id)
        case_id = make_case_id(group.candidate_ids)
        prompt_payload = _build_prompt_payload(group, group_candidates, case_id)

        last_error: Exception | None = None
        for attempt in range(1, self.max_response_attempts + 1):
            try:
                event_case = self.llm_client.generate_json(
                    self.prompt_name,
                    {"candidate_group_json": prompt_payload},
                    schema=EventCase,
                )
                normalized = normalize_event_case(
                    event_case,
                    group=group,
                    group_candidates=group_candidates,
                    case_id=case_id,
                )
                validate_event_case_consistency(normalized, group)
                return normalized
            except (LLMClientError, EventMergeError, ValueError) as exc:
                last_error = exc
                LOGGER.warning(
                    "Event case merge failed for group %s (attempt %s/%s): %s",
                    group.group_id,
                    attempt,
                    self.max_response_attempts,
                    exc,
                )

        raise EventMergeError(
            f"Event case merge failed for group {group.group_id} after "
            f"{self.max_response_attempts} attempt(s)."
        ) from last_error


def normalize_event_case(
    event_case: EventCase,
    *,
    group: CandidateGroup,
    group_candidates: list[EventCandidate],
    case_id: str | None = None,
) -> EventCase:
    """Return an EventCase with deterministic traceability fields.

    The model-generated semantic fields are preserved, but evidence is rebuilt
    from the source candidates so every candidate retains a direct, stable
    evidence reference.  Timeline IDs/evidence IDs are adjusted in
    ``TimelineBuilder`` after timeline refinement; this function leaves the
    validated model timeline attached for that next stage.
    """

    resolved_case_id = case_id or make_case_id(group.candidate_ids)
    evidence = build_evidence_spans(resolved_case_id, group_candidates)
    candidate_ids = list(group.candidate_ids)

    return EventCase(
        case_id=resolved_case_id,
        title=event_case.title,
        summary=event_case.summary,
        candidate_ids=candidate_ids,
        related_meeting_ids=_unique_ordered(
            candidate.meeting_id for candidate in group_candidates
        ),
        first_occurred_at=event_case.first_occurred_at,
        actors=_unique_ordered(event_case.actors),
        occurrence=event_case.occurrence,
        discussion=event_case.discussion,
        actions=list(event_case.actions),
        result=event_case.result,
        status=event_case.status,
        remaining_issues=list(event_case.remaining_issues),
        evidence=evidence,
        timeline=list(event_case.timeline),
    )


def build_evidence_spans(
    case_id: str,
    candidates: list[EventCandidate],
) -> list[EvidenceSpan]:
    """Build one deterministic evidence span per source candidate."""

    evidence: list[EvidenceSpan] = []
    for ordinal, candidate in enumerate(candidates, start=1):
        evidence.append(
            EvidenceSpan(
                evidence_id=make_evidence_id(case_id, candidate.candidate_id, ordinal),
                candidate_id=candidate.candidate_id,
                meeting_id=candidate.meeting_id,
                segment_id=candidate.segment_id,
                source_file=candidate.source_file,
                text=candidate.evidence_text,
            )
        )
    return evidence


def validate_event_case_consistency(event_case: EventCase, group: CandidateGroup) -> None:
    """Validate cross-field consistency not covered by dataclass schemas."""

    expected_candidate_ids = list(group.candidate_ids)
    if event_case.candidate_ids != expected_candidate_ids:
        raise EventMergeError(
            "EventCase.candidate_ids must exactly match the candidate group order."
        )

    evidence_candidate_ids = {span.candidate_id for span in event_case.evidence}
    missing_evidence = [
        candidate_id
        for candidate_id in expected_candidate_ids
        if candidate_id not in evidence_candidate_ids
    ]
    if missing_evidence:
        raise EventMergeError(
            f"EventCase.evidence is missing candidate references: {missing_evidence}"
        )



def _index_candidates(candidates: list[EventCandidate]) -> dict[str, EventCandidate]:
    if not candidates:
        raise EventMergeError("No event candidates were provided for event merging.")

    indexed: dict[str, EventCandidate] = {}
    for candidate in candidates:
        if candidate.candidate_id in indexed:
            raise EventMergeError(f"Duplicate candidate_id: {candidate.candidate_id}")
        indexed[candidate.candidate_id] = candidate
    return indexed


def _resolve_group_candidates(
    group: CandidateGroup,
    candidates_by_id: dict[str, EventCandidate],
) -> list[EventCandidate]:
    if not group.candidate_ids:
        raise EventMergeError(f"Candidate group {group.group_id} has no candidate IDs.")

    missing = [
        candidate_id
        for candidate_id in group.candidate_ids
        if candidate_id not in candidates_by_id
    ]
    if missing:
        raise EventMergeError(
            f"Candidate group {group.group_id} references missing candidates: {missing}"
        )
    return [candidates_by_id[candidate_id] for candidate_id in group.candidate_ids]


def _build_prompt_payload(
    group: CandidateGroup,
    candidates: list[EventCandidate],
    case_id: str,
) -> dict[str, Any]:
    return {
        "group": asdict(group),
        "case_id": case_id,
        "candidate_ids": list(group.candidate_ids),
        "candidates": [asdict(candidate) for candidate in candidates],
        "evidence_instruction": (
            "Create EventCase.evidence from candidate evidence_text and preserve "
            "candidate_id, meeting_id, segment_id, source_file references."
        ),
    }


def _unique_ordered(values: Any) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if not isinstance(value, str):
            value = str(value)
        normalized = value.strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result
