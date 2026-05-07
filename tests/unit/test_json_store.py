from pathlib import Path

import pytest

from meeting_summarizer.schemas import ParsedDocument, Segment
from meeting_summarizer.storage.json_store import JsonStore, JsonStoreError, load_json, save_json


def test_save_and_load_json_uses_utf8_indented_debug_format(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "artifact.json"

    save_json(path, {"title": "회의", "items": [1, 2]})

    assert path.read_text(encoding="utf-8") == '{\n  "title": "회의",\n  "items": [\n    1,\n    2\n  ]\n}\n'
    assert load_json(path) == {"title": "회의", "items": [1, 2]}


def test_json_store_creates_standard_debug_artifact_structure(tmp_path: Path) -> None:
    store = JsonStore(tmp_path)

    store.ensure_structure()

    for directory in ("raw", "parsed", "segments", "candidates", "cases", "reports", "vector_store"):
        assert (tmp_path / directory).is_dir()
    assert store.artifact_path("parsed_documents") == tmp_path / "parsed" / "parsed_documents.json"
    assert store.artifact_path("segments") == tmp_path / "segments" / "segments.json"


def test_json_store_round_trips_parsed_documents_and_segments(tmp_path: Path) -> None:
    store = JsonStore(tmp_path)
    parsed_documents = [
        ParsedDocument(
            meeting_id="meeting_1",
            source_file="회의.txt",
            file_type=".txt",
            total_pages=1,
            pages=["본문"],
        )
    ]
    segments = [
        Segment(
            segment_id="segment_001",
            meeting_id="meeting_1",
            source_file="회의.txt",
            start_page=1,
            end_page=1,
            text="본문",
        )
    ]

    parsed_path = store.save_parsed_documents(parsed_documents)
    segment_path = store.save_segments(segments)

    assert parsed_path == tmp_path / "parsed" / "parsed_documents.json"
    assert segment_path == tmp_path / "segments" / "segments.json"
    assert store.load_parsed_documents() == parsed_documents
    assert store.load_segments() == segments


def test_typed_load_reports_invalid_artifact_shape(tmp_path: Path) -> None:
    store = JsonStore(tmp_path)
    save_json(store.artifact_path("parsed_documents"), {"not": "a list"})

    with pytest.raises(JsonStoreError, match="Expected a list"):
        store.load_parsed_documents()
