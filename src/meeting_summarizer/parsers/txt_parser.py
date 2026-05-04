"""TXT parser implementation."""

from __future__ import annotations

from pathlib import Path

from meeting_summarizer.parsers.base import BaseParser


class TxtParser(BaseParser):
    """Parse plain text files into one page-like chunk."""

    @property
    def file_type(self) -> str:
        return ".txt"

    def _extract_pages(self, file_path: Path) -> list[str]:
        text = file_path.read_text(encoding="utf-8")
        # TXT has no page concept in MVP. Normalize as one page-like unit.
        return [text]
