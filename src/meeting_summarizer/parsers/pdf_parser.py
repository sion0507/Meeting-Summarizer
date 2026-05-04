"""PDF parser implementation."""

from __future__ import annotations

from pathlib import Path

from meeting_summarizer.parsers.base import BaseParser, ParseError


class PdfParser(BaseParser):
    """Parse PDF into per-page text list."""

    @property
    def file_type(self) -> str:
        return ".pdf"

    def _extract_pages(self, file_path: Path) -> list[str]:
        try:
            from pypdf import PdfReader  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ParseError("pypdf is required for PDF parsing. Install dependency 'pypdf'.") from exc

        reader = PdfReader(str(file_path))
        pages: list[str] = []
        for page in reader.pages:
            text = page.extract_text() or ""
            pages.append(text)
        return pages
