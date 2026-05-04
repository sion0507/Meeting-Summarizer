"""DOCX parser implementation."""

from __future__ import annotations

from pathlib import Path

from meeting_summarizer.parsers.base import BaseParser, ParseError


class DocxParser(BaseParser):
    """Parse DOCX files into one page-like chunk (paragraph-joined text)."""

    @property
    def file_type(self) -> str:
        return ".docx"

    def _extract_pages(self, file_path: Path) -> list[str]:
        try:
            from docx import Document  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ParseError(
                "python-docx is required for DOCX parsing. Install dependency 'python-docx'."
            ) from exc

        document = Document(str(file_path))
        lines = [p.text.strip() for p in document.paragraphs if p.text and p.text.strip()]
        return ["\n".join(lines)]
