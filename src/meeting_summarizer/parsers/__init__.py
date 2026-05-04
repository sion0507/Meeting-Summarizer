"""Parser package."""

from meeting_summarizer.parsers.base import (
    BaseParser,
    ParseError,
    UnsupportedFileTypeError,
    get_parser_for_file,
)
from meeting_summarizer.parsers.docx_parser import DocxParser
from meeting_summarizer.parsers.pdf_parser import PdfParser
from meeting_summarizer.parsers.txt_parser import TxtParser

__all__ = [
    "BaseParser",
    "ParseError",
    "UnsupportedFileTypeError",
    "get_parser_for_file",
    "TxtParser",
    "DocxParser",
    "PdfParser",
]
