"""LLM-backed final Markdown report generation from ``EventCase`` data.

The final report stage must not re-read raw meeting text. This writer accepts
only validated ``EventCase`` objects, sends their structured JSON shape to the
configured LLM prompt, validates the required event-centered Markdown sections,
and saves the report artifact.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from meeting_summarizer.agents.llm_client import LLMClient, LLMClientError
from meeting_summarizer.reporting.markdown_report import (
    MarkdownReportError,
    save_markdown_report,
    validate_event_report_markdown,
)
from meeting_summarizer.schemas import EventCase
from meeting_summarizer.utils.logging import get_logger

DEFAULT_PROMPT_NAME = "final_report"
DEFAULT_MAX_RESPONSE_ATTEMPTS = 2

LOGGER = get_logger(__name__)


class ReportWriteError(RuntimeError):
    """Raised when final Markdown report generation cannot complete."""


class ReportWriter:
    """Generate and save a final Markdown report from structured event cases."""

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

    def write_report(
        self, event_cases: list[EventCase], output_path: str | Path
    ) -> Path:
        """Generate Markdown from ``event_cases`` and save it to ``output_path``."""

        markdown = self.generate_report(event_cases)
        try:
            return save_markdown_report(output_path, markdown)
        except MarkdownReportError as exc:
            raise ReportWriteError(str(exc)) from exc

    def generate_report(self, event_cases: list[EventCase]) -> str:
        """Return validated Markdown generated only from ``EventCase`` objects."""

        if not event_cases:
            raise ReportWriteError(
                "No event cases were provided for Markdown report generation."
            )

        event_cases_json = [_event_case_to_prompt_dict(case) for case in event_cases]
        last_error: Exception | None = None
        for attempt in range(1, self.max_response_attempts + 1):
            try:
                markdown = self.llm_client.generate_text(
                    self.prompt_name,
                    {"event_cases_json": event_cases_json},
                )
                validate_event_report_markdown(markdown)
                return markdown.rstrip() + "\n"
            except (LLMClientError, MarkdownReportError) as exc:
                last_error = exc
                LOGGER.warning(
                    "Markdown report generation failed (attempt %s/%s): %s",
                    attempt,
                    self.max_response_attempts,
                    exc,
                )

        raise ReportWriteError("Failed to generate valid Markdown report.") from last_error


def _event_case_to_prompt_dict(event_case: EventCase) -> dict[str, Any]:
    return asdict(event_case)
