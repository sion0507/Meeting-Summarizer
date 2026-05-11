"""LLM-backed timeline refinement for validated ``EventCase`` objects.

The timeline stage receives structured event-case data only; it never reads raw
meeting text.  The LLM may reorganize the case timeline, while this module
normalizes IDs, sequential order, and evidence references so the final structure
stays consistent with the preserved evidence spans.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from typing import Any

from meeting_summarizer.agents.llm_client import (
    LLMClient,
    LLMClientError,
    validate_json_schema,
)
from meeting_summarizer.schemas import EventCase, TimelineItem
from meeting_summarizer.utils.ids import make_timeline_id
from meeting_summarizer.utils.logging import get_logger

DEFAULT_PROMPT_NAME = "timeline_builder"
DEFAULT_MAX_RESPONSE_ATTEMPTS = 2

LOGGER = get_logger(__name__)


class TimelineBuildError(RuntimeError):
    """Raised when a valid event-case timeline cannot be produced."""


class TimelineBuilder:
    """Refine event-case timelines using the configured LLM provider."""

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

    def build_for_cases(self, event_cases: list[EventCase]) -> list[EventCase]:
        """Return cases with refined, schema-valid timelines."""

        return [self.build_for_case(event_case) for event_case in event_cases]

    def build_for_case(self, event_case: EventCase) -> EventCase:
        """Refine one event case timeline without changing other fields."""

        last_error: Exception | None = None
        for attempt in range(1, self.max_response_attempts + 1):
            try:
                payload = self.llm_client.generate_json(
                    self.prompt_name,
                    {"event_case_json": asdict(event_case)},
                    schema=dict,
                )
                timeline = _extract_timeline(payload, event_case.case_id)
                normalized = normalize_timeline(event_case, timeline)
                return replace(event_case, timeline=normalized)
            except (LLMClientError, TimelineBuildError, ValueError) as exc:
                last_error = exc
                LOGGER.warning(
                    "Timeline build failed for case %s (attempt %s/%s): %s",
                    event_case.case_id,
                    attempt,
                    self.max_response_attempts,
                    exc,
                )

        raise TimelineBuildError(
            f"Timeline build failed for case {event_case.case_id} after "
            f"{self.max_response_attempts} attempt(s)."
        ) from last_error


def normalize_timeline(
    event_case: EventCase,
    timeline: list[TimelineItem],
) -> list[TimelineItem]:
    """Normalize timeline IDs/order and keep evidence references in-case only."""

    if not timeline:
        raise TimelineBuildError(
            f"Timeline for case {event_case.case_id} must contain at least one item."
        )

    allowed_evidence_ids = {span.evidence_id for span in event_case.evidence}
    if not allowed_evidence_ids:
        raise TimelineBuildError(
            f"Case {event_case.case_id} has no evidence spans for timeline references."
        )

    normalized: list[TimelineItem] = []
    for order, item in enumerate(timeline):
        evidence_ids = _filter_unique_evidence_ids(item.evidence_ids, allowed_evidence_ids)
        if not evidence_ids:
            evidence_ids = _fallback_evidence_ids(event_case, order)

        normalized.append(
            TimelineItem(
                timeline_id=make_timeline_id(event_case.case_id, order + 1),
                date=item.date,
                order=order,
                stage=item.stage,
                description=item.description,
                evidence_ids=evidence_ids,
            )
        )

    return normalized


def _extract_timeline(payload: Any, case_id: str) -> list[TimelineItem]:
    if not isinstance(payload, dict):
        raise TimelineBuildError(
            f"Timeline payload for case {case_id} must be a JSON object."
        )
    timeline_payload = payload.get("timeline")
    try:
        return validate_json_schema(timeline_payload, list[TimelineItem])
    except LLMClientError as exc:
        raise TimelineBuildError(
            f"Timeline payload for case {case_id} failed schema validation: {exc}"
        ) from exc


def _filter_unique_evidence_ids(
    evidence_ids: list[str],
    allowed_evidence_ids: set[str],
) -> list[str]:
    seen: set[str] = set()
    filtered: list[str] = []
    for evidence_id in evidence_ids:
        if evidence_id in allowed_evidence_ids and evidence_id not in seen:
            seen.add(evidence_id)
            filtered.append(evidence_id)
    return filtered


def _fallback_evidence_ids(event_case: EventCase, order: int) -> list[str]:
    evidence_count = len(event_case.evidence)
    evidence = event_case.evidence[min(order, evidence_count - 1)]
    return [evidence.evidence_id]
