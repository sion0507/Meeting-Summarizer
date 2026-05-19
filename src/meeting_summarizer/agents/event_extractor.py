"""LLM-backed extraction of :class:`EventCandidate` objects from segments.

This module owns the event-candidate extraction stage only.  It sends one
already-parsed/preprocessed segment to the configured LLM, validates every
candidate against the core schema, fills a missing ``embedding_text`` with a
stable code-based identity string, and skips invalid candidate objects after
logging the validation error.
"""

from __future__ import annotations

from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from meeting_summarizer.agents.llm_client import (
    LLMClient,
    LLMClientError,
    validate_json_schema,
)
from meeting_summarizer.schemas import EventCandidate, Segment
from meeting_summarizer.utils.ids import make_candidate_id
from meeting_summarizer.utils.logging import get_logger

DEFAULT_PROMPT_NAME = "event_extraction"
DEFAULT_MAX_RESPONSE_ATTEMPTS = 2

LOGGER = get_logger(__name__)


class EventExtractionError(RuntimeError):
    """Raised when a segment cannot produce a usable extraction response."""


class EventExtractor:
    """Extract validated event candidates from meeting segments.

    Retry/skip policy:
    - Invalid LLM responses at the response-envelope level (non-JSON, missing
      ``candidates``, or non-list ``candidates``) are retried up to
      ``max_response_attempts`` and then fail clearly with ``EventExtractionError``.
    - Invalid individual candidate objects are logged and skipped.  A bad
      candidate should not discard other valid candidates from the same segment.
    - ``embedding_text`` is the only candidate field repaired by this stage: if
      missing or blank, it is deterministically rebuilt from event identity
      fields before schema validation.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        *,
        prompt_name: str = DEFAULT_PROMPT_NAME,
        max_response_attempts: int = DEFAULT_MAX_RESPONSE_ATTEMPTS,
        max_workers: int = 1,
    ) -> None:
        if max_response_attempts < 1:
            raise ValueError("max_response_attempts must be >= 1.")
        if max_workers < 1:
            raise ValueError("max_workers must be >= 1.")
        self.llm_client = llm_client
        self.prompt_name = prompt_name
        self.max_response_attempts = max_response_attempts
        self.max_workers = max_workers

    def extract_from_segments(self, segments: list[Segment]) -> list[EventCandidate]:
        """Extract candidates from every segment and return a flat list."""

        if self.max_workers == 1 or len(segments) <= 1:
            candidates: list[EventCandidate] = []
            for segment in segments:
                candidates.extend(self.extract_from_segment(segment))
            return candidates

        segment_results: dict[int, list[EventCandidate]] = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_pairs = [
                (index, executor.submit(self.extract_from_segment, segment))
                for index, segment in enumerate(segments)
            ]
            for index, future in future_pairs:
                segment_results[index] = future.result()

        candidates: list[EventCandidate] = []
        for index in range(len(segments)):
            candidates.extend(segment_results[index])
        return candidates

    def extract_from_segment(self, segment: Segment) -> list[EventCandidate]:
        """Extract, repair ``embedding_text``, and validate candidates for one segment."""

        raw_candidates = self._request_candidate_payload(segment)
        validated: list[EventCandidate] = []

        for ordinal, raw_candidate in enumerate(raw_candidates, start=1):
            if not isinstance(raw_candidate, dict):
                LOGGER.warning(
                    "Skipping non-object event candidate from segment %s at ordinal %s: %r",
                    segment.segment_id,
                    ordinal,
                    raw_candidate,
                )
                continue

            candidate_data = self._prepare_candidate_data(raw_candidate, segment, ordinal)
            try:
                candidate = validate_json_schema(candidate_data, EventCandidate)
            except LLMClientError as exc:
                LOGGER.warning(
                    "Skipping invalid event candidate from segment %s at ordinal %s: %s",
                    segment.segment_id,
                    ordinal,
                    exc,
                )
                continue
            validated.append(candidate)

        return validated

    def _request_candidate_payload(self, segment: Segment) -> list[Any]:
        last_error: Exception | None = None
        for attempt in range(1, self.max_response_attempts + 1):
            try:
                payload = self.llm_client.generate_json(
                    self.prompt_name,
                    {"segment_json": asdict(segment)},
                    schema=dict,
                )
                return _extract_candidates_list(payload, segment.segment_id)
            except (LLMClientError, EventExtractionError) as exc:
                last_error = exc
                LOGGER.warning(
                    "Event candidate extraction response failed for segment %s "
                    "(attempt %s/%s): %s",
                    segment.segment_id,
                    attempt,
                    self.max_response_attempts,
                    exc,
                )

        raise EventExtractionError(
            "Event candidate extraction failed for segment "
            f"{segment.segment_id} after {self.max_response_attempts} attempt(s)."
        ) from last_error

    def _prepare_candidate_data(
        self,
        raw_candidate: dict[str, Any],
        segment: Segment,
        ordinal: int,
    ) -> dict[str, Any]:
        candidate_data = dict(raw_candidate)

        # Pipeline-owned identifiers and source metadata must be deterministic;
        # they should not depend on model formatting choices.
        candidate_data["candidate_id"] = make_candidate_id(
            segment.meeting_id,
            segment.segment_id,
            ordinal,
        )
        candidate_data["meeting_id"] = segment.meeting_id
        candidate_data["segment_id"] = segment.segment_id
        candidate_data["source_file"] = segment.source_file

        if not _is_non_empty_string(candidate_data.get("embedding_text")):
            candidate_data["embedding_text"] = build_embedding_text(candidate_data)

        return candidate_data


def build_embedding_text(candidate_data: dict[str, Any]) -> str:
    """Build deterministic candidate identity text for embeddings.

    The order intentionally mirrors the schema notes: title, summary, problem,
    action, result, actors, and keywords.  Empty values are omitted so repeated
    runs over the same LLM candidate object produce the same string.
    """

    parts: list[str] = []
    for field_name in ("title", "summary", "problem", "action", "result"):
        value = candidate_data.get(field_name)
        if _is_non_empty_string(value):
            parts.append(str(value).strip())

    for field_name in ("actors", "keywords"):
        value = candidate_data.get(field_name)
        if isinstance(value, list):
            normalized_items = [str(item).strip() for item in value if str(item).strip()]
            if normalized_items:
                parts.append(", ".join(normalized_items))

    return " | ".join(parts)


def _extract_candidates_list(payload: Any, segment_id: str) -> list[Any]:
    if not isinstance(payload, dict):
        raise EventExtractionError(
            f"Extraction payload for segment {segment_id} must be a JSON object."
        )
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        raise EventExtractionError(
            f"Extraction payload for segment {segment_id} must contain a candidates list."
        )
    return candidates


def _is_non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())
