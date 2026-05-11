from __future__ import annotations

from pathlib import Path

import pytest

from meeting_summarizer.agents.llm_client import LLMClient, PromptLoader
from meeting_summarizer.agents.timeline_builder import TimelineBuildError, TimelineBuilder
from meeting_summarizer.schemas import EventCase, EvidenceSpan, TimelineItem


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
    (tmp_path / "timeline_builder.md").write_text(
        "case=$event_case_json", encoding="utf-8"
    )
    return LLMClient(QueueProvider(responses), PromptLoader(tmp_path))


def _case() -> EventCase:
    return EventCase(
        case_id="case_alpha",
        title="서비스 장애 대응",
        summary="장애 대응 경과",
        candidate_ids=["candidate_1", "candidate_2"],
        related_meeting_ids=["meeting_alpha"],
        first_occurred_at=None,
        actors=["운영팀"],
        occurrence="서비스 장애 발생",
        discussion="원인 분석 논의",
        actions=["로그 확인"],
        result=None,
        status="진행 중",
        remaining_issues=["원인 확정"],
        evidence=[
            EvidenceSpan(
                evidence_id="evidence_1",
                candidate_id="candidate_1",
                meeting_id="meeting_alpha",
                segment_id="segment_1",
                source_file="alpha.txt",
                text="장애가 보고되었다.",
            ),
            EvidenceSpan(
                evidence_id="evidence_2",
                candidate_id="candidate_2",
                meeting_id="meeting_alpha",
                segment_id="segment_2",
                source_file="alpha.txt",
                text="로그 확인을 결정했다.",
            ),
        ],
        timeline=[
            TimelineItem(
                timeline_id="old_timeline",
                date=None,
                order=0,
                stage="unknown",
                description="초기 항목",
                evidence_ids=["evidence_1"],
            )
        ],
    )


def test_timeline_builder_normalizes_order_ids_and_evidence_refs(tmp_path: Path) -> None:
    response = """
    {
      "timeline": [
        {
          "timeline_id": "model_10",
          "date": null,
          "order": 99,
          "stage": "occurrence",
          "description": "장애가 보고됨",
          "evidence_ids": ["unknown", "evidence_1", "evidence_1"]
        },
        {
          "timeline_id": "model_20",
          "date": null,
          "order": 1,
          "stage": "action",
          "description": "로그 확인을 결정함",
          "evidence_ids": ["unknown"]
        }
      ]
    }
    """
    builder = TimelineBuilder(_client(tmp_path, [response]))

    updated = builder.build_for_case(_case())

    assert [item.order for item in updated.timeline] == [0, 1]
    assert [item.timeline_id for item in updated.timeline] == [
        "timeline_case_alpha_001",
        "timeline_case_alpha_002",
    ]
    assert updated.timeline[0].stage == "occurrence"
    assert updated.timeline[0].evidence_ids == ["evidence_1"]
    assert updated.timeline[1].evidence_ids == ["evidence_2"]
    assert "장애 대응 경과" in builder.llm_client.provider.prompts[0]


def test_timeline_builder_rejects_invalid_stage_after_retries(tmp_path: Path) -> None:
    bad_response = """
    {
      "timeline": [
        {
          "timeline_id": "model_10",
          "date": null,
          "order": 0,
          "stage": "invalid_stage",
          "description": "잘못된 단계",
          "evidence_ids": ["evidence_1"]
        }
      ]
    }
    """
    builder = TimelineBuilder(_client(tmp_path, [bad_response, bad_response]))

    with pytest.raises(TimelineBuildError, match="failed for case case_alpha after 2 attempt"):
        builder.build_for_case(_case())
