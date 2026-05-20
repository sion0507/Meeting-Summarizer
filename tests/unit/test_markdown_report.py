from __future__ import annotations

from pathlib import Path

import pytest

from meeting_summarizer.agents.llm_client import LLMClient, PromptLoader
from meeting_summarizer.agents.report_writer import ReportWriteError, ReportWriter
from meeting_summarizer.reporting.markdown_report import (
    MarkdownReportError,
    save_markdown_report,
    validate_event_report_markdown,
)
from meeting_summarizer.schemas import EventCase, EvidenceSpan, TimelineItem


class QueueTextProvider:
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


def _client(tmp_path: Path, responses: list[str]) -> LLMClient:
    (tmp_path / "final_report.md").write_text(
        "report=$event_cases_json",
        encoding="utf-8",
    )
    return LLMClient(QueueTextProvider(responses), PromptLoader(tmp_path))


def _case() -> EventCase:
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


def _valid_markdown() -> str:
    return """# 회의 이벤트 리포트

## 서비스 장애 대응

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


def test_validate_and_save_markdown_report_requires_event_sections(tmp_path: Path) -> None:
    path = tmp_path / "reports" / "report.md"

    saved_path = save_markdown_report(path, _valid_markdown())

    assert saved_path == path
    assert path.read_text(encoding="utf-8").startswith("# 회의 이벤트 리포트")

    with pytest.raises(MarkdownReportError, match="missing required section"):
        validate_event_report_markdown("## 서비스 장애 대응\n\n- 최초 발생: 확인 필요")


def test_report_writer_uses_event_cases_only_and_saves_report(tmp_path: Path) -> None:
    client = _client(tmp_path, [_valid_markdown()])
    writer = ReportWriter(client)

    report_path = writer.write_report([_case()], tmp_path / "reports" / "report.md")

    provider = client.provider
    assert report_path == tmp_path / "reports" / "report.md"
    assert report_path.read_text(encoding="utf-8") == _valid_markdown().rstrip() + "\n"
    assert provider.response_formats == ["text"]
    assert '"case_id": "case_alpha"' in provider.prompts[0]
    assert "장애가 보고되었다." in provider.prompts[0]


def test_report_writer_retries_invalid_markdown(tmp_path: Path) -> None:
    client = _client(tmp_path, ["not a report", _valid_markdown()])
    writer = ReportWriter(client)

    markdown = writer.generate_report([_case()])

    assert markdown == _valid_markdown().rstrip() + "\n"
    assert len(client.provider.prompts) == 2


def test_report_writer_rejects_empty_event_cases(tmp_path: Path) -> None:
    writer = ReportWriter(_client(tmp_path, []))

    with pytest.raises(ReportWriteError, match="No event cases"):
        writer.generate_report([])


def test_report_writer_batches_event_cases_and_merges_markdown(tmp_path: Path) -> None:
    client = _client(tmp_path, [_valid_markdown(), _valid_markdown()])
    writer = ReportWriter(client, event_case_batch_size=1)

    markdown = writer.generate_report([_case(), _case()])

    assert len(client.provider.prompts) == 2
    assert markdown.count("# 회의 이벤트 리포트") == 1
    assert markdown.count("## 서비스 장애 대응") == 2
