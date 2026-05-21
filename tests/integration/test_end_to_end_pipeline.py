from __future__ import annotations

from pathlib import Path
from typing import Sequence

from meeting_summarizer.agents.llm_client import LLMClient, PromptLoader
from meeting_summarizer.config import AppConfig
from meeting_summarizer.embeddings.embedder import BaseEmbedder
from meeting_summarizer.orchestrator import MeetingEventPipeline
from meeting_summarizer.schemas import EventCandidate
from meeting_summarizer.storage.json_store import JsonStore, load_json


class QueueProvider:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.response_formats: list[str] = []

    def generate(self, prompt: str, *, response_format: str = "json") -> str:
        self.response_formats.append(response_format)
        if not self.responses:
            raise AssertionError("No queued LLM response left.")
        return self.responses.pop(0)


class FixedEmbedder(BaseEmbedder):
    @property
    def dimension(self) -> int:
        return 3

    def embed_texts(
        self, texts: Sequence[str], *, batch_size: int | None = None
    ) -> list[list[float]]:
        return [[1.0, 0.0, 0.0] for _ in texts]


class FakeFaissStore:
    def __init__(self, vector_store_dir: Path) -> None:
        self.index_path = vector_store_dir / "candidates.faiss"
        self.metadata_path = vector_store_dir / "candidate_metadata.json"
        self.seen_vectors: dict[str, list[float]] = {}

    def build_and_save(
        self,
        candidate_vectors: dict[str, list[float]],
        *,
        candidate_metadata: dict[str, dict[str, str]] | None = None,
    ) -> None:
        self.seen_vectors = dict(candidate_vectors)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.index_path.write_bytes(b"fake-faiss-index")
        self.metadata_path.write_text(
            '{"count": %d}\n' % len(candidate_vectors),
            encoding="utf-8",
        )


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
        report_event_case_batch_size=2,
        report_max_workers=1,
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
    return LLMClient(
        QueueProvider(responses),
        PromptLoader(Path(__file__).resolve().parents[2] / "prompts"),
    )


def test_end_to_end_pipeline_with_sample_txt_and_test_doubles(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / "meeting.txt").write_text(
        "서비스 장애가 보고되었다.\n\n운영팀은 로그를 확인하기로 했다.\n",
        encoding="utf-8",
    )

    extraction_response = """
    {
      "candidates": [
        {
          "title": "서비스 장애 대응",
          "summary": "서비스 장애 보고 후 로그 확인 조치가 논의됨",
          "occurred_at": null,
          "actors": ["운영팀"],
          "problem": "서비스 장애 보고",
          "discussion": "로그 확인 필요성 논의",
          "action": "로그 확인",
          "result": null,
          "status": "진행 중",
          "evidence_text": "서비스 장애가 보고되었다.",
          "keywords": ["서비스 장애", "로그"],
          "embedding_text": "서비스 장애 대응 로그 확인"
        }
      ]
    }
    """
    merge_response = """
    {
      "case_id": "model_case",
      "title": "서비스 장애 대응",
      "summary": "서비스 장애 보고 후 로그 확인이 결정됨",
      "candidate_ids": ["model_candidate"],
      "related_meeting_ids": ["model_meeting"],
      "first_occurred_at": null,
      "actors": ["운영팀"],
      "occurrence": "서비스 장애가 보고됨",
      "discussion": "로그 확인 필요성이 논의됨",
      "actions": ["로그 확인"],
      "result": null,
      "status": "진행 중",
      "remaining_issues": ["원인 확정"],
      "evidence": [],
      "timeline": [
        {
          "timeline_id": "model_timeline",
          "date": null,
          "order": 0,
          "stage": "occurrence",
          "description": "서비스 장애 보고",
          "evidence_ids": []
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
          "order": 0,
          "stage": "action",
          "description": "운영팀이 로그를 확인하기로 함",
          "evidence_ids": []
        }
      ]
    }
    """
    report_response = """## 서비스 장애 대응

- 최초 발생: 확인 필요
- 관련 회의록: meeting.txt
- 사건 내용: 서비스 장애가 보고됨
- 처리 과정: 운영팀이 로그를 확인하기로 함
- 담당자: 운영팀
- 최종 결과: 확인 필요
- 현재 상태: 진행 중
- 남은 이슈: 원인 확정
- 근거: 서비스 장애가 보고되었다.
"""

    config = _config(tmp_path)
    faiss_store = FakeFaissStore(config.vector_store_dir)
    client = _client(
        tmp_path,
        [extraction_response, merge_response, timeline_response, report_response],
    )
    pipeline = MeetingEventPipeline(
        config,
        llm_client=client,
        store=JsonStore(tmp_path),
        embedder=FixedEmbedder(),
        faiss_store=faiss_store,
    )

    result = pipeline.run()

    assert result.parsed_document_count == 1
    assert result.segment_count == 1
    assert result.event_candidate_count == 1
    assert result.candidate_vector_count == 1
    assert result.candidate_group_count == 1
    assert result.event_case_count == 1
    assert result.report_path.read_text(encoding="utf-8").startswith(
        "## 서비스 장애 대응"
    )
    assert result.faiss_index_path.read_bytes() == b"fake-faiss-index"
    assert client.provider.response_formats == ["json", "json", "json", "text"]

    candidates = JsonStore(tmp_path).load_event_candidates()
    assert isinstance(candidates[0], EventCandidate)
    assert list(faiss_store.seen_vectors) == [candidates[0].candidate_id]

    groups = load_json(result.candidate_groups_path)
    assert groups[0]["is_singleton"] is True
    cases = load_json(result.event_cases_path)
    assert cases[0]["title"] == "서비스 장애 대응"
    assert cases[0]["evidence"][0]["text"] == "서비스 장애가 보고되었다."
