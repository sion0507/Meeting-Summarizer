"""Markdown report persistence and format checks.

This module is intentionally limited to Markdown handling. It does not parse
source meeting files and does not synthesize event content from raw text. The
report body must already have been generated from structured ``EventCase``
objects by the report-writing stage.
"""

from __future__ import annotations

from pathlib import Path

REPORT_ENCODING = "utf-8"
REQUIRED_EVENT_SECTIONS = (
    "최초 발생",
    "관련 회의록",
    "사건 내용",
    "처리 과정",
    "담당자",
    "최종 결과",
    "현재 상태",
    "남은 이슈",
    "근거",
)


class MarkdownReportError(RuntimeError):
    """Raised when a Markdown report cannot be validated or saved."""


def validate_event_report_markdown(markdown: str) -> None:
    """Validate the MVP event-centered Markdown report shape.

    The check is deliberately structural rather than content-generating: every
    event section introduced by ``##`` must contain the agreed Korean bullet
    labels, and the report must contain at least one event. Missing values are
    allowed inside the bullets because the prompt instructs the LLM to write
    ``확인 필요`` for unknown facts.
    """

    if not markdown.strip():
        raise MarkdownReportError("Markdown report must not be empty.")

    event_blocks = _extract_event_blocks(markdown)
    if not event_blocks:
        raise MarkdownReportError(
            "Markdown report must include at least one event heading starting with '## '."
        )

    for index, block in enumerate(event_blocks, start=1):
        first_line = block.splitlines()[0].strip()
        if first_line == "##" or first_line == "## []":
            raise MarkdownReportError(f"Event heading {index} must include a title.")
        missing = [
            section
            for section in REQUIRED_EVENT_SECTIONS
            if f"- {section}:" not in block
        ]
        if missing:
            raise MarkdownReportError(
                f"Event report block {index} is missing required section(s): {missing}."
            )


def save_markdown_report(path: str | Path, markdown: str) -> Path:
    """Validate and atomically save a Markdown report to ``path``."""

    validate_event_report_markdown(markdown)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = markdown.rstrip() + "\n"
    tmp_path = target.with_name(f".{target.name}.tmp")
    try:
        tmp_path.write_text(text, encoding=REPORT_ENCODING)
        tmp_path.replace(target)
    except OSError as exc:
        raise MarkdownReportError(f"Failed to save Markdown report: {target}") from exc
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return target


def _extract_event_blocks(markdown: str) -> list[str]:
    lines = markdown.splitlines()
    heading_indexes = [
        index for index, line in enumerate(lines) if line.startswith("## ")
    ]
    blocks: list[str] = []
    for offset, start in enumerate(heading_indexes):
        end = heading_indexes[offset + 1] if offset + 1 < len(heading_indexes) else len(lines)
        block = "\n".join(lines[start:end]).strip()
        if block:
            blocks.append(block)
    return blocks
