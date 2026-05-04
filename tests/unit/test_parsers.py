from pathlib import Path

import pytest

from meeting_summarizer.parsers import DocxParser, PdfParser, TxtParser, get_parser_for_file
from meeting_summarizer.parsers.base import UnsupportedFileTypeError


def test_get_parser_for_supported_extensions() -> None:
    assert isinstance(get_parser_for_file("a.txt"), TxtParser)
    assert isinstance(get_parser_for_file("a.docx"), DocxParser)
    assert isinstance(get_parser_for_file("a.pdf"), PdfParser)


def test_get_parser_for_unsupported_extension() -> None:
    with pytest.raises(UnsupportedFileTypeError):
        get_parser_for_file("a.md")


def test_txt_parser_returns_parsed_document(tmp_path: Path) -> None:
    txt_path = tmp_path / "meeting.txt"
    txt_path.write_text("line1\nline2", encoding="utf-8")

    parsed = TxtParser().parse(txt_path)

    assert parsed.source_file == "meeting.txt"
    assert parsed.file_type == ".txt"
    assert parsed.total_pages == 1
    assert parsed.pages == ["line1\nline2"]


def test_docx_parser_returns_parsed_document(tmp_path: Path) -> None:
    docx = pytest.importorskip("docx")

    docx_path = tmp_path / "meeting.docx"
    document = docx.Document()
    document.add_paragraph("first")
    document.add_paragraph("second")
    document.save(docx_path)

    parsed = DocxParser().parse(docx_path)

    assert parsed.file_type == ".docx"
    assert parsed.total_pages == 1
    assert "first" in parsed.pages[0]
    assert "second" in parsed.pages[0]


def test_pdf_parser_returns_parsed_document(tmp_path: Path) -> None:
    pypdf = pytest.importorskip("pypdf")

    pdf_path = tmp_path / "meeting.pdf"
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=300, height=300)
    with pdf_path.open("wb") as fp:
        writer.write(fp)

    parsed = PdfParser().parse(pdf_path)

    assert parsed.file_type == ".pdf"
    assert parsed.total_pages == 1
    assert len(parsed.pages) == 1
