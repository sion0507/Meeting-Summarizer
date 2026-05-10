from __future__ import annotations

import logging
from pathlib import Path

import pytest

from meeting_summarizer.agents.event_extractor import (
    EventExtractionError,
    EventExtractor,
    build_embedding_text,
)
from meeting_summarizer.agents.llm_client import LLMClient, PromptLoader
from meeting_summarizer.schemas import Segment


class QueueProvider:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.prompts: list[str] = []

    def generate(self, prompt: str, *, response_format: str = "json") -> str:
        self.prompts.append(prompt)
        assert response_format == "json"
        if not self.responses:
            raise AssertionError("No queued LLM response left.")
        return self.responses.pop(0)


def _client(tmp_path: Path, responses: list[str]) -> LLMClient:
    (tmp_path / "event_extraction.md").write_text("segment=$segment_json", encoding="utf-8")
    return LLMClient(QueueProvider(responses), PromptLoader(tmp_path))


def _segment(segment_id: str = "segment_001") -> Segment:
    return Segment(
        segment_id=segment_id,
        meeting_id="meeting_alpha",
        source_file="alpha.txt",
        start_page=1,
        end_page=2,
        text="운영팀은 장애 로그를 확인하기로 했다.",
    )


def test_event_extractor_creates_candidates_for_each_segment(tmp_path: Path) -> None:
    first_response = """
    {
      "candidates": [
        {
          "candidate_id": "model_id_should_be_replaced",
          "meeting_id": "wrong_meeting",
          "segment_id": "wrong_segment",
          "source_file": "wrong.txt",
          "title": "장애 대응",
          "summary": "서비스 장애 대응 방안이 논의됨",
          "occurred_at": null,
          "actors": ["운영팀"],
          "problem": "서비스 장애",
          "discussion": "원인 분석 필요",
          "action": "로그 확인",
          "result": null,
          "status": "진행 중",
          "evidence_text": "운영팀은 장애 로그를 확인하기로 했다.",
          "keywords": ["장애", "로그"]
        }
      ]
    }
    """
    second_response = "{\"candidates\": []}"
    extractor = EventExtractor(_client(tmp_path, [first_response, second_response]))

    candidates = extractor.extract_from_segments([_segment("segment_001"), _segment("segment_002")])

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.candidate_id == "candidate_meeting_alpha_segment_001_001"
    assert candidate.meeting_id == "meeting_alpha"
    assert candidate.segment_id == "segment_001"
    assert candidate.source_file == "alpha.txt"
    assert candidate.embedding_text == (
        "장애 대응 | 서비스 장애 대응 방안이 논의됨 | 서비스 장애 | 로그 확인 | 운영팀 | 장애, 로그"
    )


def test_build_embedding_text_is_deterministic() -> None:
    candidate_data = {
        "title": "장애 대응",
        "summary": "장애 로그 확인",
        "problem": "장애",
        "action": "로그 확인",
        "result": "재발 방지 논의",
        "actors": ["운영팀", "개발팀"],
        "keywords": ["장애", "로그"],
    }

    assert build_embedding_text(candidate_data) == build_embedding_text(dict(candidate_data))
    assert build_embedding_text(candidate_data) == (
        "장애 대응 | 장애 로그 확인 | 장애 | 로그 확인 | 재발 방지 논의 | 운영팀, 개발팀 | 장애, 로그"
    )


def test_event_extractor_logs_and_skips_invalid_candidate(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    response = """
    {
      "candidates": [
        {
          "candidate_id": "candidate_1",
          "meeting_id": "meeting_alpha",
          "segment_id": "segment_001",
          "source_file": "alpha.txt",
          "title": "",
          "summary": "요약",
          "occurred_at": null,
          "actors": [],
          "problem": null,
          "discussion": null,
          "action": null,
          "result": null,
          "status": "확인 필요",
          "evidence_text": "근거",
          "keywords": [],
          "embedding_text": "요약"
        }
      ]
    }
    """
    extractor = EventExtractor(_client(tmp_path, [response]))

    with caplog.at_level(logging.WARNING):
        candidates = extractor.extract_from_segment(_segment())

    assert candidates == []
    assert "Skipping invalid event candidate" in caplog.text


def test_event_extractor_retries_bad_response_then_raises(tmp_path: Path) -> None:
    extractor = EventExtractor(_client(tmp_path, ["not-json", "{\"items\": []}"]))

    with pytest.raises(EventExtractionError, match="failed for segment segment_001 after 2 attempt"):
        extractor.extract_from_segment(_segment())
