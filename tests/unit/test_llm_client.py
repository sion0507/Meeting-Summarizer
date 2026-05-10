from __future__ import annotations

from pathlib import Path

import pytest

from meeting_summarizer.agents.llm_client import (
    LLMClient,
    LLMJSONParseError,
    LLMProviderError,
    LLMResponseValidationError,
    LocalLLMProvider,
    PromptLoader,
    PromptLoadError,
    parse_json_response,
    validate_json_schema,
)
from meeting_summarizer.schemas import EventCandidate, EventCase


class StaticProvider:
    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []

    def generate(self, prompt: str, *, response_format: str = "json") -> str:
        self.prompts.append(prompt)
        assert response_format == "json"
        return self.response


class FakeLlama:
    def __init__(self, **kwargs: object) -> None:
        self.init_kwargs = kwargs
        self.chat_kwargs: dict[str, object] | None = None

    def create_chat_completion(self, **kwargs: object) -> dict[str, object]:
        self.chat_kwargs = kwargs
        return {"choices": [{"message": {"content": "{\"ok\": true}"}}]}


def test_prompt_loader_loads_and_binds_variables(tmp_path: Path) -> None:
    (tmp_path / "sample.md").write_text("회의: $meeting\n$payload", encoding="utf-8")
    loader = PromptLoader(tmp_path)

    rendered = loader.render("sample", {"meeting": "주간", "payload": {"a": 1}})

    assert "회의: 주간" in rendered
    assert '"a": 1' in rendered


def test_prompt_loader_reports_missing_variable(tmp_path: Path) -> None:
    (tmp_path / "sample.md").write_text("$required", encoding="utf-8")
    loader = PromptLoader(tmp_path)

    with pytest.raises(PromptLoadError, match="Missing variable"):
        loader.render("sample", {})


def test_llm_client_parses_and_validates_event_candidate_list(tmp_path: Path) -> None:
    (tmp_path / "event_extraction.md").write_text("segment=$segment_json", encoding="utf-8")
    response = """
    {
      "candidates": [
        {
          "candidate_id": "candidate_1",
          "meeting_id": "meeting_1",
          "segment_id": "segment_1",
          "source_file": "meeting.txt",
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
          "keywords": ["장애", "로그"],
          "embedding_text": "장애 대응 서비스 장애 로그 확인 운영팀 장애 로그"
        }
      ]
    }
    """
    provider = StaticProvider(response)
    client = LLMClient(provider, PromptLoader(tmp_path))

    payload = client.generate_json(
        "event_extraction",
        {"segment_json": {"segment_id": "segment_1"}},
        schema=dict,
    )
    candidates = client.generate_json(
        "event_extraction",
        {"segment_json": {"segment_id": "segment_1"}},
        schema=dict,
    )["candidates"]
    validated = validate_json_schema(candidates, list[EventCandidate])

    assert payload["candidates"][0]["candidate_id"] == "candidate_1"
    assert validated[0].title == "장애 대응"
    assert "segment_1" in provider.prompts[-1]


def test_llm_client_validates_nested_event_case_schema(tmp_path: Path) -> None:
    (tmp_path / "event_case_merge.md").write_text("$candidate_group_json", encoding="utf-8")
    response = """
    {
      "case_id": "case_1",
      "title": "장애 대응",
      "summary": "장애 대응 경과",
      "candidate_ids": ["candidate_1"],
      "related_meeting_ids": ["meeting_1"],
      "first_occurred_at": null,
      "actors": ["운영팀"],
      "occurrence": "장애 발생",
      "discussion": "원인 분석",
      "actions": ["로그 확인"],
      "result": null,
      "status": "진행 중",
      "remaining_issues": ["원인 확정"],
      "evidence": [
        {
          "evidence_id": "evidence_1",
          "candidate_id": "candidate_1",
          "meeting_id": "meeting_1",
          "segment_id": "segment_1",
          "source_file": "meeting.txt",
          "text": "운영팀은 장애 로그를 확인하기로 했다."
        }
      ],
      "timeline": [
        {
          "timeline_id": "timeline_1",
          "date": null,
          "order": 0,
          "stage": "action",
          "description": "로그 확인 결정",
          "evidence_ids": ["evidence_1"]
        }
      ]
    }
    """
    client = LLMClient(StaticProvider(response), PromptLoader(tmp_path))

    event_case = client.generate_json(
        "event_case_merge",
        {"candidate_group_json": [{"candidate_id": "candidate_1"}]},
        schema=EventCase,
    )

    assert event_case.case_id == "case_1"
    assert event_case.evidence[0].evidence_id == "evidence_1"
    assert event_case.timeline[0].stage == "action"


def test_invalid_json_response_raises_parse_error() -> None:
    with pytest.raises(LLMJSONParseError):
        parse_json_response("not json")


def test_schema_validation_error_is_clear(tmp_path: Path) -> None:
    (tmp_path / "event_case_merge.md").write_text("$candidate_group_json", encoding="utf-8")
    client = LLMClient(StaticProvider('{"case_id": "case_1"}'), PromptLoader(tmp_path))

    with pytest.raises(LLMResponseValidationError, match="missing fields"):
        client.generate_json("event_case_merge", {"candidate_group_json": []}, schema=EventCase)


def test_local_provider_requires_model_path() -> None:
    with pytest.raises(LLMProviderError, match="LOCAL_MODEL_PATH"):
        LocalLLMProvider(model_path="")


def test_local_provider_loads_gguf_directly_without_http(tmp_path: Path) -> None:
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"fake gguf for constructor validation")

    provider = LocalLLMProvider(
        model_path=model_path,
        n_ctx=1024,
        n_gpu_layers=10,
        max_tokens=128,
        llama_factory=FakeLlama,
    )
    response = provider.generate("Return JSON", response_format="json")

    assert response == '{"ok": true}'
    assert provider._llm.init_kwargs == {
        "model_path": str(model_path),
        "n_ctx": 1024,
        "n_gpu_layers": 10,
        "verbose": False,
    }
    assert provider._llm.chat_kwargs is not None
    assert provider._llm.chat_kwargs["response_format"] == {"type": "json_object"}


def test_local_provider_reports_missing_model_file(tmp_path: Path) -> None:
    with pytest.raises(LLMProviderError, match="GGUF model file not found"):
        LocalLLMProvider(model_path=tmp_path / "missing.gguf", llama_factory=FakeLlama)
