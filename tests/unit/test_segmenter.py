from meeting_summarizer.preprocessing.segmenter import Segmenter
from meeting_summarizer.schemas import ParsedDocument


def test_segmenter_builds_one_page_segments_by_default_for_paginated_docs() -> None:
    document = ParsedDocument(
        meeting_id="m1",
        source_file="meeting.pdf",
        file_type=".pdf",
        total_pages=5,
        pages=["p1", "p2", "p3", "p4", "p5"],
    )

    segments = Segmenter().build_document_segments(document)

    assert [s.segment_id for s in segments] == [
        "segment_001",
        "segment_002",
        "segment_003",
        "segment_004",
        "segment_005",
    ]
    assert [(s.start_page, s.end_page) for s in segments] == [
        (1, 1),
        (2, 2),
        (3, 3),
        (4, 4),
        (5, 5),
    ]
    assert [s.text for s in segments] == ["p1", "p2", "p3", "p4", "p5"]


def test_segmenter_txt_fallback_uses_paragraph_pseudo_pages_with_configured_size() -> None:
    txt = "\n\n".join([f"paragraph {i}" for i in range(1, 8)])
    document = ParsedDocument(
        meeting_id="m2",
        source_file="meeting.txt",
        file_type=".txt",
        total_pages=1,
        pages=[txt],
    )

    segmenter = Segmenter(segment_size_pages=2, txt_paragraphs_per_pseudo_page=3)
    segments = segmenter.build_document_segments(document)

    assert len(segments) == 2
    assert segments[0].start_page == 1
    assert segments[0].end_page == 2
    assert "paragraph 1" in segments[0].text
    assert "paragraph 6" in segments[0].text
    assert segments[1].start_page == 3
    assert segments[1].end_page == 3
    assert "paragraph 7" in segments[1].text


def test_segmenter_is_deterministic_for_same_input() -> None:
    document = ParsedDocument(
        meeting_id="m3",
        source_file="meeting.docx",
        file_type=".docx",
        total_pages=3,
        pages=["a", "b", "c"],
    )
    segmenter = Segmenter(segment_size_pages=2)

    first = segmenter.build_document_segments(document)
    second = segmenter.build_document_segments(document)

    assert first == second
