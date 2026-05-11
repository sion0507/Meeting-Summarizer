from __future__ import annotations

from pathlib import Path

import pytest

from meeting_summarizer.agents.event_merger import EventMergeError, EventMerger
from meeting_summarizer.agents.llm_client import LLMClient, PromptLoader
from meeting_summarizer.linking.grouping import CandidateGroup
from meeting_summarizer.schemas import EventCandidate


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
    (tmp_path / "event_case_merge.md").write_text(
        "group=$candidate_group_json", encoding="utf-8"
    )
    return LLMClient(QueueProvider(responses), PromptLoader(tmp_path))


def _candidate(candidate_id: str, meeting_id: str = "meeting_alpha") -> EventCandidate:
    return EventCandidate(
        candidate_id=candidate_id,
        meeting_id=meeting_id,
        segment_id=f"segment_{candidate_id}",
        source_file=f"{meeting_id}.txt",
        title="장애 대응",
        summary="서비스 장애 대응 방안이 논의됨",
        occurred_at=None,
        actors=["운영팀"],
        problem="서비스 장애",
        discussion="원인 분석 필요",
        action="로그 확인",
        result=None,
        status="진행 중",
        evidence_text=f"{candidate_id} 근거 문장",
        keywords=["장애", "로그"],
        embedding_text="장애 대응 서비스 장애 로그 확인",
    )


def _group() -> CandidateGroup:
    return CandidateGroup(
        group_id="group_alpha",
        candidate_ids=["candidate_1", "candidate_2"],
        is_singleton=False,
        threshold=0.8,
        links=[],
    )


def test_event_merger_returns_one_normalized_case_per_group(tmp_path: Path) -> None:
    response = """
    {
      "case_id": "model_case_id",
      "title": "서비스 장애 대응",
      "summary": "서비스 장애 원인 확인과 로그 점검이 논의됨",
      "candidate_ids": ["candidate_2"],
      "related_meeting_ids": ["wrong"],
      "first_occurred_at": null,
      "actors": ["운영팀", "운영팀"],
      "occurrence": "서비스 장애가 확인됨",
      "discussion": "운영팀이 원인 분석 필요성을 논의함",
      "actions": ["로그 확인"],
      "result": null,
      "status": "진행 중",
      "remaining_issues": ["원인 확정"],
      "evidence": [
        {
          "evidence_id": "model_evidence",
          "candidate_id": "candidate_2",
          "meeting_id": "wrong",
          "segment_id": "wrong",
          "source_file": "wrong.txt",
          "text": "model text"
        }
      ],
      "timeline": [
        {
          "timeline_id": "model_timeline",
          "date": null,
          "order": 0,
          "stage": "action",
          "description": "로그 확인 결정",
          "evidence_ids": ["model_evidence"]
        }
      ]
    }
    """
    merger = EventMerger(_client(tmp_path, [response]))
    candidates = [_candidate("candidate_1"), _candidate("candidate_2", "meeting_beta")]

    cases = merger.merge_groups([_group()], candidates)

    assert len(cases) == 1
    event_case = cases[0]
    assert event_case.case_id.startswith("case_")
    assert event_case.candidate_ids == ["candidate_1", "candidate_2"]
    assert event_case.related_meeting_ids == ["meeting_alpha", "meeting_beta"]
    assert event_case.actors == ["운영팀"]
    assert [span.candidate_id for span in event_case.evidence] == [
        "candidate_1",
        "candidate_2",
    ]
    assert event_case.evidence[0].text == "candidate_1 근거 문장"
    assert "candidate_1" in merger.llm_client.provider.prompts[0]


def test_event_merger_rejects_group_with_missing_candidate(tmp_path: Path) -> None:
    merger = EventMerger(_client(tmp_path, []))

    with pytest.raises(EventMergeError, match="missing candidates"):
        merger.merge_groups([_group()], [_candidate("candidate_1")])


def test_event_merger_retries_invalid_llm_case_then_raises(tmp_path: Path) -> None:
    merger = EventMerger(_client(tmp_path, ["not-json", '{"case_id": "case_1"}']))

    with pytest.raises(EventMergeError, match="failed for group group_alpha after 2 attempt"):
        merger.merge_groups([_group()], [_candidate("candidate_1"), _candidate("candidate_2")])
