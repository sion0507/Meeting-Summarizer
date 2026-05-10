import pytest

from meeting_summarizer.embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    KureV1Embedder,
    HashingEmbedder,
    KiwiMorphTokenizer,
    RegexTokenizer,
    build_candidate_embedding_text,
    create_embedder,
)
from meeting_summarizer.schemas.event_candidate import EventCandidate


def _candidate(candidate_id: str, embedding_text: str) -> EventCandidate:
    return EventCandidate(
        candidate_id=candidate_id,
        meeting_id="meeting_001",
        segment_id="segment_001",
        source_file="meeting.txt",
        title="장애 대응",
        summary="로그인 장애를 처리했다.",
        occurred_at=None,
        actors=["플랫폼팀"],
        problem="로그인 장애",
        discussion="원인 분석",
        action="패치 배포",
        result="복구",
        status="closed",
        evidence_text="로그인 장애 후 패치를 배포했다.",
        keywords=["로그인", "장애"],
        embedding_text=embedding_text,
    )


def test_hashing_embedder_embeds_texts_in_batches_deterministically() -> None:
    embedder = HashingEmbedder(vector_size=16, tokenizer=RegexTokenizer())

    first = embedder.embed_texts(["alpha beta", "gamma delta"], batch_size=1)
    second = embedder.embed_texts(["alpha beta", "gamma delta"], batch_size=2)

    assert first == second
    assert len(first) == 2
    assert all(len(vector) == 16 for vector in first)


def test_embed_candidates_returns_candidate_id_mapping() -> None:
    candidates = [
        _candidate("candidate_001", "로그인 장애"),
        _candidate("candidate_002", "결제 장애"),
    ]

    vectors = HashingEmbedder(
        vector_size=8,
        tokenizer=RegexTokenizer(),
    ).embed_candidates(candidates, batch_size=2)

    assert list(vectors) == ["candidate_001", "candidate_002"]
    assert all(len(vector) == 8 for vector in vectors.values())


def test_candidate_embedding_text_prefers_embedding_identity_fields() -> None:
    candidate = _candidate("candidate_001", "핵심 임베딩 텍스트")

    text = build_candidate_embedding_text(candidate)

    assert "핵심 임베딩 텍스트" in text
    assert "장애 대응" in text


def test_create_embedder_defaults_to_local_kure_v1_without_loading_model(tmp_path) -> None:
    embedder = create_embedder(
        DEFAULT_EMBEDDING_MODEL,
        model_path=tmp_path / "KURE-v1",
        tokenizer=RegexTokenizer(),
    )

    assert isinstance(embedder, KureV1Embedder)
    assert embedder.model_path == tmp_path / "KURE-v1"


def test_kiwi_tokenizer_splits_korean_text_into_morpheme_tokens() -> None:
    pytest.importorskip("kiwipiepy")

    tokens = KiwiMorphTokenizer().tokenize("로그인 장애를 처리했습니다.")

    assert "로그인" in tokens
    assert "장애" in tokens
