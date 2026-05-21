"""Pipeline orchestration for the local meeting event analysis MVP.

The orchestrator fixes the MVP stage order and connects dedicated modules while
keeping parsing, LLM prompting, embedding, vector storage, grouping, merging,
timeline building, and reporting in their own packages.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from meeting_summarizer.agents.event_extractor import EventExtractor
from meeting_summarizer.agents.event_merger import EventMerger
from meeting_summarizer.agents.llm_client import LLMClient, build_llm_client
from meeting_summarizer.agents.report_writer import ReportWriter
from meeting_summarizer.agents.timeline_builder import TimelineBuilder
from meeting_summarizer.config import AppConfig, ensure_output_directories
from meeting_summarizer.embeddings.embedder import BaseEmbedder, create_embedder
from meeting_summarizer.linking.faiss_store import FaissCandidateStore
from meeting_summarizer.linking.grouping import CandidateGroup, group_candidates
from meeting_summarizer.parsers import UnsupportedFileTypeError, get_parser_for_file
from meeting_summarizer.preprocessing.segmenter import Segmenter
from meeting_summarizer.schemas import EventCandidate, EventCase, ParsedDocument, Segment
from meeting_summarizer.storage.json_store import JsonStore
from meeting_summarizer.utils.logging import get_logger

LOGGER = get_logger(__name__) #__name__ is current module name for representing source of log

T = TypeVar("T") # T is used for _run_stage() to get any type of var and return tsame type

SUPPORTED_INPUT_EXTENSIONS = {".txt", ".docx", ".pdf"}


class PipelineError(RuntimeError):
    """Raised when an orchestrated pipeline stage cannot complete."""


class PipelineStageError(PipelineError):
    """Raised with stage context when an end-to-end stage fails."""

    def __init__(self, stage: str, reason: str) -> None:
        self.stage = stage
        self.reason = reason
        super().__init__(f"Pipeline stage failed [{stage}]: {reason}")



@dataclass(frozen=True, slots=True)
class CandidateVectorResult:
    """Embedding and vector-store outputs for event candidates."""

    vectors: dict[str, list[float]]
    faiss_index_path: Path
    candidate_metadata_path: Path
    count: int


@dataclass(frozen=True, slots=True)
class CandidateGroupingResult:
    """Candidate grouping artifact output."""

    groups_path: Path
    group_count: int
    groups: list[CandidateGroup]


@dataclass(slots=True)
class EndToEndPipelineResult:
    """All canonical artifacts produced by a full pipeline run."""

    parsed_documents_path: Path
    parsed_document_count: int
    segments_path: Path
    segment_count: int
    event_candidates_path: Path
    event_candidate_count: int
    faiss_index_path: Path
    candidate_metadata_path: Path
    candidate_vector_count: int
    candidate_groups_path: Path
    candidate_group_count: int
    event_cases_path: Path
    event_case_count: int
    report_path: Path


@dataclass(slots=True)
class PipelineRunResult:
    """Paths and counts produced by the event-case merge pipeline stage."""

    event_cases_path: Path
    event_case_count: int


@dataclass(slots=True)
class ReportRunResult:
    """Path and count produced by the Markdown report pipeline stage."""

    report_path: Path
    event_case_count: int


class MeetingEventPipeline:
    """Coordinate the fixed local MVP pipeline.

    Full stage order::

        input files -> parsing/preprocessing -> 1-page segments -> LLM candidate
        extraction -> candidate embeddings -> FAISS vector storage -> similarity
        grouping -> LLM case merge -> LLM timeline organization -> JSON output ->
        Markdown report output.

    The final report stage consumes only structured ``EventCase`` data and never
    re-reads raw meeting text.
    """

    def __init__(
        self,
        config: AppConfig,
        *,
        llm_client: LLMClient | None = None,
        store: JsonStore | None = None,
        segmenter: Segmenter | None = None,
        embedder: BaseEmbedder | None = None,
        faiss_store: Any | None = None,
    ) -> None:
        self.config = config
        self.llm_client = llm_client or build_llm_client(config)
        self.store = store or JsonStore(config.data_dir)
        self.segmenter = segmenter or Segmenter(config.segment_size_pages)
        self.embedder = embedder
        self.faiss_store = faiss_store
        self.event_extractor = EventExtractor(
            self.llm_client, max_workers=self.config.extraction_max_workers
        )
        self.event_merger = EventMerger(self.llm_client)
        self.timeline_builder = TimelineBuilder(self.llm_client)
        self.report_writer = ReportWriter(
            self.llm_client,
            event_case_batch_size=self.config.report_event_case_batch_size,
            max_workers=self.config.report_max_workers,
        )

    def run(self) -> EndToEndPipelineResult:
        """Run the complete end-to-end MVP pipeline with stage-level errors."""

        ensure_output_directories(self.config)
        self.store.ensure_structure()

        parsed_documents = self._run_stage("parsing / preprocessing", self.parse_inputs) # not put () means just pass function to _run_stage and actually woks in there by action;)
        segments = self._run_stage(
            "1-page segment generation",
            lambda: self.build_segments(parsed_documents), # by usinf lambda it can pass function with parameters to _run_stages
        )
        candidates = self._run_stage(
            "event candidate extraction with LLM",
            lambda: self.extract_event_candidates(segments),
        )
        vector_result = self._run_stage(
            "embedding generation and FAISS vector storage",
            lambda: self.embed_and_store_candidates(candidates),
        )
        grouping_result = self._run_stage(
            "similarity-based candidate grouping",
            lambda: self.group_and_save_candidates(vector_result.vectors),
        )
        merge_result = self._run_stage(
            "LLM-based event case merge and timeline organization",
            lambda: self.merge_event_cases(grouping_result.groups, candidates),
        )
        report_result = self._run_stage(
            "Markdown report output",
            self.write_markdown_report_from_artifacts,
        )

        return EndToEndPipelineResult(
            parsed_documents_path=self.config.parsed_documents_path,
            parsed_document_count=len(parsed_documents),
            segments_path=self.config.segments_path,
            segment_count=len(segments),
            event_candidates_path=self.config.event_candidates_path,
            event_candidate_count=len(candidates),
            faiss_index_path=vector_result.faiss_index_path,
            candidate_metadata_path=vector_result.candidate_metadata_path,
            candidate_vector_count=vector_result.count,
            candidate_groups_path=grouping_result.groups_path,
            candidate_group_count=grouping_result.group_count,
            event_cases_path=merge_result.event_cases_path,
            event_case_count=merge_result.event_case_count,
            report_path=report_result.report_path,
        )

    def parse_inputs(self) -> list[ParsedDocument]:
        """Parse all supported files from ``config.input_dir`` and save JSON."""

        input_files = _discover_input_files(self.config.input_dir)
        documents: list[ParsedDocument] = []
        for input_file in input_files:
            parser = get_parser_for_file(input_file)
            documents.append(parser.parse(input_file))

        if not documents:
            raise PipelineError(
                f"No parsed documents were produced from {self.config.input_dir}."
            )
        self.store.save_parsed_documents(documents)
        LOGGER.info("Parsed %s document(s).", len(documents))
        return documents

    def build_segments(self, documents: list[ParsedDocument]) -> list[Segment]:
        """Build fixed-size segments from parsed documents and save JSON."""

        if not documents:
            raise PipelineError("No parsed documents were provided for segmentation.")
        segments = self.segmenter.build_segments(documents)
        if not segments:
            raise PipelineError("No segments were generated from parsed documents.")
        self.store.save_segments(segments)
        LOGGER.info("Generated %s segment(s).", len(segments))
        return segments

    def extract_event_candidates(self, segments: list[Segment]) -> list[EventCandidate]:
        """Extract candidates from segments using the configured LLM and save JSON."""

        if not segments:
            raise PipelineError("No segments were provided for event candidate extraction.")
        candidates = self.event_extractor.extract_from_segments(segments)
        if not candidates:
            raise PipelineError("No event candidates were found by the extraction stage.")
        self.store.save_event_candidates(candidates)
        LOGGER.info("Extracted %s event candidate(s).", len(candidates))
        return candidates

    def embed_and_store_candidates(
        self,
        candidates: list[EventCandidate],
    ) -> CandidateVectorResult:
        """Generate candidate embeddings, build FAISS index, and save metadata."""

        if not candidates:
            raise PipelineError("No event candidates were provided for embedding.")
        embedder = self.embedder or create_embedder(
            self.config.embedding_model,
            model_path=self.config.embedding_model_path,
            device=self.config.embedding_device,
        )
        vectors = embedder.embed_candidates(candidates)
        vector_store = self.faiss_store or FaissCandidateStore(
            self.config.vector_store_dir
        )
        metadata_payload = {
            candidate.candidate_id: {
                "meeting_id": candidate.meeting_id,
                "segment_id": candidate.segment_id,
                "source_file": candidate.source_file,
                "title": candidate.title,
            }
            for candidate in candidates
        }
        vector_store.build_and_save(vectors, candidate_metadata=metadata_payload)
        LOGGER.info("Stored %s candidate vector(s) in FAISS.", len(vectors))
        return CandidateVectorResult(
            vectors=vectors,
            faiss_index_path=Path(vector_store.index_path),
            candidate_metadata_path=Path(vector_store.metadata_path),
            count=len(vectors),
        )

    def group_and_save_candidates(
        self,
        candidate_vectors: dict[str, list[float]],
    ) -> CandidateGroupingResult:
        """Group candidates by strict configured similarity threshold and save JSON."""

        if not candidate_vectors:
            raise PipelineError("No candidate vectors were provided for grouping.")
        groups = group_candidates(
            candidate_vectors,
            similarity_threshold=self.config.similarity_threshold,
        )
        if not groups:
            raise PipelineError("No candidate groups were produced by grouping.")
        groups_path = self.store.save_candidate_groups(groups)
        LOGGER.info("Saved %s candidate group(s).", len(groups))
        return CandidateGroupingResult(
            groups_path=groups_path,
            group_count=len(groups),
            groups=groups,
        )

    def merge_event_cases(
        self,
        groups: list[CandidateGroup],
        candidates: list[EventCandidate],
    ) -> PipelineRunResult:
        """Merge groups, build timelines, save final event cases as JSON."""

        if not candidates:
            raise PipelineError("No event candidates were provided for event-case merge.")
        if not groups:
            raise PipelineError("No candidate groups were provided for event-case merge.")
        merged_cases = self.event_merger.merge_groups(groups, candidates)
        event_cases = self.timeline_builder.build_for_cases(merged_cases)
        if not event_cases:
            raise PipelineError("No event cases were produced by event-case merge.")
        event_cases_path = self.store.save_event_cases(event_cases)
        LOGGER.info(
            "Saved %s event case(s) to %s.", len(event_cases), event_cases_path
        )
        return PipelineRunResult(
            event_cases_path=event_cases_path,
            event_case_count=len(event_cases),
        )

    def merge_event_cases_from_artifacts(self) -> PipelineRunResult:
        """Load candidates/groups, merge cases, build timelines, and save JSON.

        This method intentionally does not re-read raw meeting text.  It starts
        from structured artifacts generated by earlier stages and writes the
        canonical final case artifact at ``data/cases/event_cases.json``.
        """

        ensure_output_directories(self.config)
        self.store.ensure_structure()

        candidates = self.store.load_event_candidates()
        groups = self.store.load_candidate_groups()
        return self.merge_event_cases(groups, candidates)

    def save_event_cases(self, event_cases: list[EventCase]) -> PipelineRunResult:
        """Persist already-built event cases through the canonical JSON store."""

        if not event_cases:
            raise PipelineError("No event cases were provided for saving.")
        ensure_output_directories(self.config)
        self.store.ensure_structure()
        event_cases_path = self.store.save_event_cases(event_cases)
        return PipelineRunResult(
            event_cases_path=event_cases_path,
            event_case_count=len(event_cases),
        )

    def write_markdown_report_from_artifacts(self) -> ReportRunResult:
        """Load final event cases and write ``data/reports/report.md``.

        This stage intentionally starts from ``event_cases.json`` only. It does
        not load parsed documents, segments, or raw meeting files, preserving the
        MVP rule that final reports are generated from structured event data.
        """

        ensure_output_directories(self.config)
        self.store.ensure_structure()

        event_cases = self.store.load_event_cases()
        if not event_cases:
            raise PipelineError("No event cases found for Markdown report generation.")

        report_path = self.report_writer.write_report(
            event_cases,
            self.config.report_path,
        )
        LOGGER.info(
            "Saved Markdown report for %s event case(s) to %s.",
            len(event_cases),
            report_path,
        )
        return ReportRunResult(
            report_path=report_path,
            event_case_count=len(event_cases),
        )

    def write_markdown_report(self, event_cases: list[EventCase]) -> ReportRunResult:
        """Generate and save a Markdown report from already-built event cases.
           Used by tests or callers that already hold EventCase objects in memory.
        """

        if not event_cases:
            raise PipelineError(
                "No event cases were provided for Markdown report generation."
            )
        ensure_output_directories(self.config)
        self.store.ensure_structure()
        report_path = self.report_writer.write_report(
            event_cases, self.config.report_path
        )
        return ReportRunResult(
            report_path=report_path,
            event_case_count=len(event_cases),
        )

    def _run_stage(self, stage: str, action: Callable[[], T]) -> T:
        LOGGER.info("Starting pipeline stage: %s", stage)
        stage_started = time.perf_counter()
        usage_before = self.llm_client.usage_totals_snapshot()
        try:
            result = action()
        except PipelineStageError:
            raise
        except Exception as exc:
            raise PipelineStageError(stage, str(exc)) from exc
        stage_elapsed_sec = time.perf_counter() - stage_started
        usage_after = self.llm_client.usage_totals_snapshot()
        completion_tokens = usage_after["completion_tokens"] - usage_before["completion_tokens"]
        total_tokens = usage_after["total_tokens"] - usage_before["total_tokens"]
        if completion_tokens > 0 and stage_elapsed_sec > 0:
            aggregate_tok_per_sec = round(completion_tokens / stage_elapsed_sec, 3)
            total_token_per_sec = round(total_tokens / stage_elapsed_sec, 3)
            LOGGER.info(
                "Stage throughput: stage=%s stage_elapsed_sec=%.3f completion_tokens=%s total_tokens=%s aggregate_tok_per_sec=%s total_token_per_sec=%s",
                stage,
                stage_elapsed_sec,
                completion_tokens,
                total_tokens,
                aggregate_tok_per_sec,
                total_token_per_sec,
            )
        LOGGER.info("Completed pipeline stage: %s", stage)
        return result


def _discover_input_files(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise PipelineError(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise PipelineError(f"Input path is not a directory: {input_dir}")

    files = sorted(path for path in input_dir.iterdir() if path.is_file())
    if not files:
        raise PipelineError(f"Input directory is empty: {input_dir}")

    unsupported = [
        path for path in files if path.suffix.lower() not in SUPPORTED_INPUT_EXTENSIONS
    ]
    if unsupported:
        unsupported_list = ", ".join(path.name for path in unsupported)
        raise UnsupportedFileTypeError(
            "Unsupported input file type(s): "
            f"{unsupported_list} (supported: .txt, .docx, .pdf)"
        )

    return files
