"""Parser interfaces and file-type dispatching for meeting documents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from meeting_summarizer.schemas.document import ParsedDocument
from meeting_summarizer.utils.ids import make_meeting_id


class UnsupportedFileTypeError(ValueError):
    """Raised when the input extension is not supported by MVP."""


class ParseError(RuntimeError):
    """Raised when parser cannot extract text from a supported input."""


class BaseParser(ABC):
    """Abstract parser that converts one source file into ParsedDocument."""

    @property
    @abstractmethod
    def file_type(self) -> str:
        """Supported extension including dot, e.g. '.txt'."""

    def parse(self, file_path: str | Path) -> ParsedDocument:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")

        ext = path.suffix.lower()
        if ext != self.file_type:
            raise UnsupportedFileTypeError(
                f"{self.__class__.__name__} supports only {self.file_type}, got {ext or '[no extension]'}"
            )

        pages = self._extract_pages(path)
        if not pages:
            raise ParseError(f"No text extracted from file: {path}")

        normalized_pages = [p.strip() for p in pages if p and p.strip()]
        if not normalized_pages:
            raise ParseError(f"Extracted pages are empty after normalization: {path}")

        return ParsedDocument(
            meeting_id=make_meeting_id(path.name),
            source_file=path.name,
            file_type=ext,
            total_pages=len(normalized_pages),
            pages=normalized_pages,
        )

    @abstractmethod
    def _extract_pages(self, file_path: Path) -> list[str]:
        """Return page-like chunks from a source document."""


def get_parser_for_file(file_path: str | Path) -> BaseParser:
    """Create parser instance based on file extension."""

    path = Path(file_path)
    ext = path.suffix.lower()

    if ext == ".txt":
        from meeting_summarizer.parsers.txt_parser import TxtParser

        return TxtParser()
    if ext == ".docx":
        from meeting_summarizer.parsers.docx_parser import DocxParser

        return DocxParser()
    if ext == ".pdf":
        from meeting_summarizer.parsers.pdf_parser import PdfParser

        return PdfParser()

    raise UnsupportedFileTypeError(
        "Unsupported file type for MVP: "
        f"{ext or '[no extension]'} (supported: .txt, .docx, .pdf)"
    )
