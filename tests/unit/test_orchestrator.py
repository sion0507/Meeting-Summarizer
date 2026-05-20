from __future__ import annotations

from pathlib import Path

import pytest

from meeting_summarizer.agents.llm_client import LLMClient, PromptLoader
from meeting_summarizer.config import AppConfig
from meeting_summarizer.linking.grouping import CandidateGroup
from meeting_summarizer.orchestrator import MeetingEventPipeline, PipelineError
from meeting_summarizer.schemas import (
    EventCandidate,
    EventCase,
    EvidenceSpan,
    TimelineItem,
)
from meeting_summarizer.storage.json_store import JsonStore, load_json


class QueueProvider:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.prompts: list[str] = []
        self.response_formats: list[str] = []

    def generate(self, prompt: str, *, response_format: str = "json") -> str:
        self.prompts.append(prompt)
        self.response_formats.append(response_format)
        if not self.responses:
            raise AssertionError("No queued LLM response left.")
        return self.responses.pop(0)


def _config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        llm_provider="test",
        local_model_path="",
        embedding_model="test-embedder",
        embedding_model_path=tmp_path / "models" / "embedding",
        embedding_device="cpu",
        similarity_threshold=0.8,
        segment_size_pages=2,
        extraction_max_workers=1,
        data_dir=tmp_path,
        input_dir=tmp_path / "raw",
        parsed_dir=tmp_path / "parsed",
        segments_dir=tmp_path / "segments",
        candidates_dir=tmp_path / "candidates",
        cases_dir=tmp_path / "cases",
        reports_dir=tmp_path / "reports",
        vector_store_dir=tmp_path / "vector_store",
        parsed_documents_path=tmp_path / "parsed" / "parsed_documents.json",
        segments_path=tmp_path / "segments" / "segments.json",
        event_candidates_path=tmp_path / "candidates" / "event_candidates.json",
        event_cases_path=tmp_path / "cases" / "event_cases.json",
        report_path=tmp_path / "reports" / "report.md",
        faiss_index_path=tmp_path / "vector_store" / "candidates.faiss",
        candidate_metadata_path=tmp_path / "vector_store" / "candidate_metadata.json",
    )


def _client(tmp_path: Path, responses: list[str]) -> LLMClient:
    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir()
    (prompt_dir / "event_case_merge.md").write_text(
        "merge=$candidate_group_json",
        encoding="utf-8",
    )
    (prompt_dir / "timeline_builder.md").write_text(
        "timeline=$event_case_json",
        encoding="utf-8",
    )
    (prompt_dir / "final_report.md").write_text(
        "report=$event_cases_json",
        encoding="utf-8",
    )
    return LLMClient(QueueProvider(responses), PromptLoader(prompt_dir))


def _candidate(candidate_id: str) -> EventCandidate:
    return EventCandidate(
        candidate_id=candidate_id,
        meeting_id="meeting_alpha",
        segment_id=f"segment_{candidate_id}",
        source_file="alpha.txt",
        title="장애 대응",
        summary="장애 대응이 논의됨",
        occurred_at=None,
        actors=["운영팀"],
        problem="장애",
        discussion="원인 분석",
        action="로그 확인",
        result=None,
        status="진행 중",
        evidence_text=f"{candidate_id} 근거",
        keywords=["장애"],
        embedding_text="장애 대응 로그 확인",
    )


def _group() -> CandidateGroup:
    return CandidateGroup(
        group_id="group_alpha",
        candidate_ids=["candidate_1", "candidate_2"],
        is_singleton=False,
        threshold=0.8,
        links=[],
    )


def test_pipeline_merges_timelines_and_saves_event_cases_from_artifacts(tmp_path: Path) -> None:
    merge_response = """
    {
      "case_id": "model_case",
      "title": "서비스 장애 대응",
      "summary": "장애 원인 분석과 로그 확인이 논의됨",
      "candidate_ids": ["candidate_1", "candidate_2"],
      "related_meeting_ids": ["meeting_alpha"],
      "first_occurred_at": null,
      "actors": ["운영팀"],
      "occurrence": "서비스 장애가 보고됨",
      "discussion": "원인 분석 필요성이 논의됨",
      "actions": ["로그 확인"],
      "result": null,
      "status": "진행 중",
      "remaining_issues": ["원인 확정"],
      "evidence": [
        {
          "evidence_id": "model_evidence",
          "candidate_id": "candidate_1",
          "meeting_id": "meeting_alpha",
          "segment_id": "segment_candidate_1",
          "source_file": "alpha.txt",
          "text": "model text"
        }
      ],
      "timeline": [
        {
          "timeline_id": "model_timeline",
          "date": null,
          "order": 9,
          "stage": "discussion",
          "description": "논의됨",
          "evidence_ids": ["model_evidence"]
        }
      ]
    }
    """
    timeline_response = """
    {
      "timeline": [
        {
          "timeline_id": "model_timeline_2",
          "date": null,
          "order": 4,
          "stage": "action",
          "description": "로그 확인을 진행하기로 함",
          "evidence_ids": ["unknown"]
        }
      ]
    }
    """
    store = JsonStore(tmp_path)
    store.save_event_candidates(
        [_candidate("candidate_1"), _candidate("candidate_2")]
    )
    store.save_candidate_groups([_group()])

    pipeline = MeetingEventPipeline(
        _config(tmp_path),
        llm_client=_client(tmp_path, [merge_response, timeline_response]),
        store=store,
    )

    result = pipeline.merge_event_cases_from_artifacts()

    assert result.event_case_count == 1
    assert result.event_cases_path == tmp_path / "cases" / "event_cases.json"
    payload = load_json(result.event_cases_path)
    assert payload[0]["candidate_ids"] == ["candidate_1", "candidate_2"]
    assert [span["candidate_id"] for span in payload[0]["evidence"]] == [
        "candidate_1",
        "candidate_2",
    ]
    assert payload[0]["timeline"][0]["timeline_id"].startswith("timeline_case_")
    assert payload[0]["timeline"][0]["order"] == 0
    assert payload[0]["timeline"][0]["stage"] == "action"
    assert payload[0]["timeline"][0]["evidence_ids"] == [
        payload[0]["evidence"][0]["evidence_id"]
    ]
    assert store.load_event_cases()[0].title == "서비스 장애 대응"


def test_pipeline_requires_saved_candidates_and_groups(tmp_path: Path) -> None:
    store = JsonStore(tmp_path)
    pipeline = MeetingEventPipeline(
        _config(tmp_path),
        llm_client=_client(tmp_path, []),
        store=store,
    )

    with pytest.raises(Exception, match="event_candidates.json"):
        pipeline.merge_event_cases_from_artifacts()

    store.save_event_candidates([])
    store.save_candidate_groups([])

    with pytest.raises(PipelineError, match="No event candidates"):
        pipeline.merge_event_cases_from_artifacts()


def _event_case() -> EventCase:
    return EventCase(
        case_id="case_alpha",
        title="서비스 장애 대응",
        summary="장애 대응 경과",
        candidate_ids=["candidate_1"],
        related_meeting_ids=["meeting_alpha"],
        first_occurred_at=None,
        actors=["운영팀"],
        occurrence="서비스 장애가 보고됨",
        discussion="원인 분석을 논의함",
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
            )
        ],
        timeline=[
            TimelineItem(
                timeline_id="timeline_1",
                date=None,
                order=0,
                stage="occurrence",
                description="장애 보고",
                evidence_ids=["evidence_1"],
            )
        ],
    )


def _report_markdown() -> str:
    return """## 서비스 장애 대응

- 최초 발생: 확인 필요
- 관련 회의록: meeting_alpha
- 사건 내용: 서비스 장애가 보고됨
- 처리 과정: 원인 분석을 논의하고 로그 확인을 결정함
- 담당자: 운영팀
- 최종 결과: 확인 필요
- 현재 상태: 진행 중
- 남은 이슈: 원인 확정
- 근거: evidence_1 / alpha.txt / 장애가 보고되었다.
"""


def test_pipeline_writes_markdown_report_from_event_case_artifact(tmp_path: Path) -> None:
    store = JsonStore(tmp_path)
    store.save_event_cases([_event_case()])
    client = _client(tmp_path, [_report_markdown()])
    pipeline = MeetingEventPipeline(
        _config(tmp_path),
        llm_client=client,
        store=store,
    )

    result = pipeline.write_markdown_report_from_artifacts()

    assert result.event_case_count == 1
    assert result.report_path == tmp_path / "reports" / "report.md"
    assert (
        result.report_path.read_text(encoding="utf-8")
        == _report_markdown().rstrip() + "\n"
    )
    assert client.provider.response_formats == ["text"]
    assert '"case_id": "case_alpha"' in client.provider.prompts[0]
