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


def load_config() -> AppConfig:
    """Load config from environment with MVP-safe defaults."""

    data_dir = Path(os.getenv("DATA_DIR", "data"))
    input_dir = Path(os.getenv("INPUT_DIR", str(data_dir / "raw")))

    parsed_dir = data_dir / "parsed"
    segments_dir = data_dir / "segments"
    candidates_dir = data_dir / "candidates"
    cases_dir = data_dir / "cases"
    reports_dir = data_dir / "reports"
    vector_store_dir = data_dir / "vector_store"

    return AppConfig(
        llm_provider=os.getenv("LLM_PROVIDER", "local"),
        local_model_path=os.getenv("LOCAL_MODEL_PATH", ""),
        embedding_model=os.getenv("EMBEDDING_MODEL", ""),
        similarity_threshold=float(os.getenv("SIMILARITY_THRESHOLD", "0.8")),
        segment_size_pages=int(os.getenv("SEGMENT_SIZE_PAGES", "2")),
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
