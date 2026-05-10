"""Application configuration and fixed artifact paths for MVP."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Configuration contract that should remain stable across MVP stages."""

    llm_provider: str
    local_model_path: str
    embedding_model: str
    embedding_model_path: Path
    similarity_threshold: float
    segment_size_pages: int

    data_dir: Path
    input_dir: Path
    parsed_dir: Path
    segments_dir: Path
    candidates_dir: Path
    cases_dir: Path
    reports_dir: Path
    vector_store_dir: Path

    parsed_documents_path: Path
    segments_path: Path
    event_candidates_path: Path
    event_cases_path: Path
    report_path: Path
    faiss_index_path: Path
    candidate_metadata_path: Path


def _require_float_env(key: str, default: str) -> float:
    raw = os.getenv(key, default)
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{key} must be a float, got: {raw!r}") from exc


def _require_int_env(key: str, default: str) -> int:
    raw = os.getenv(key, default)
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{key} must be an int, got: {raw!r}") from exc


def load_config() -> AppConfig:
    """Load config from environment with explicit validation errors."""

    data_dir = Path(os.getenv("DATA_DIR", "data"))
    input_dir = Path(os.getenv("INPUT_DIR", str(data_dir / "raw")))

    parsed_dir = data_dir / "parsed"
    segments_dir = data_dir / "segments"
    candidates_dir = data_dir / "candidates"
    cases_dir = data_dir / "cases"
    reports_dir = data_dir / "reports"
    vector_store_dir = data_dir / "vector_store"

    similarity_threshold = _require_float_env("SIMILARITY_THRESHOLD", "0.8")
    if not 0.0 <= similarity_threshold <= 1.0:
        raise ValueError(
            f"SIMILARITY_THRESHOLD must be between 0 and 1, got {similarity_threshold}."
        )

    segment_size_pages = _require_int_env("SEGMENT_SIZE_PAGES", "2")
    if segment_size_pages < 1:
        raise ValueError(
            f"SEGMENT_SIZE_PAGES must be >= 1, got {segment_size_pages}."
        )

    llm_provider = os.getenv("LLM_PROVIDER", "local").strip()
    if not llm_provider:
        raise ValueError("LLM_PROVIDER must not be empty.")

    local_model_path = os.getenv("LOCAL_MODEL_PATH", "").strip()
    if llm_provider == "local" and not local_model_path:
        raise ValueError(
            "LOCAL_MODEL_PATH is required when LLM_PROVIDER=local."
        )

    return AppConfig(
        llm_provider=llm_provider,
        local_model_path=local_model_path,
        embedding_model=os.getenv("EMBEDDING_MODEL", "nlpai-lab/KURE-v1").strip(),
        embedding_model_path=Path(
            os.getenv("EMBEDDING_MODEL_PATH", "./models/KURE-v1")
        ),
        similarity_threshold=similarity_threshold,
        segment_size_pages=segment_size_pages,
        data_dir=data_dir,
        input_dir=input_dir,
        parsed_dir=parsed_dir,
        segments_dir=segments_dir,
        candidates_dir=candidates_dir,
        cases_dir=cases_dir,
        reports_dir=reports_dir,
        vector_store_dir=vector_store_dir,
        parsed_documents_path=parsed_dir / "parsed_documents.json",
        segments_path=segments_dir / "segments.json",
        event_candidates_path=candidates_dir / "event_candidates.json",
        event_cases_path=cases_dir / "event_cases.json",
        report_path=reports_dir / "report.md",
        faiss_index_path=vector_store_dir / "candidates.faiss",
        candidate_metadata_path=vector_store_dir / "candidate_metadata.json",
    )


def ensure_output_directories(config: AppConfig) -> None:
    """Create all required output directories for pipeline artifacts."""

    for directory in (
        config.data_dir,
        config.input_dir,
        config.parsed_dir,
        config.segments_dir,
        config.candidates_dir,
        config.cases_dir,
        config.reports_dir,
        config.vector_store_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)
