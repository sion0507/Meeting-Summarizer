#!/usr/bin/env python
"""Command-line entrypoint for the local MVP meeting-event pipeline."""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from meeting_summarizer.config import load_config  # noqa: E402
from meeting_summarizer.orchestrator import (  # noqa: E402
    MeetingEventPipeline,
    PipelineStageError,
)
from meeting_summarizer.utils.logging import configure_logging  # noqa: E402


def _add_common_path_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Override DATA_DIR for this run.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="Override INPUT_DIR for this run (defaults to DATA_DIR/raw).",
    )

# terminal command parser 
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local meeting event analysis pipeline stages.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print a Python traceback when a pipeline stage fails.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True) # for making sub-command

    run_parser = subparsers.add_parser(
        "run",
        help="Run the full end-to-end MVP pipeline from input files to report.md.",
    )
    _add_common_path_args(run_parser)

    parse_segment_parser = subparsers.add_parser(
        "parse-and-segment",
        help=(
            "Parse supported raw files and build segments, then write "
            "data/parsed/parsed_documents.json and data/segments/segments.json."
        ),
    )
    _add_common_path_args(parse_segment_parser)

    extract_parser = subparsers.add_parser(
        "extract-candidates",
        help=(
            "Load data/segments/segments.json and extract event candidates to "
            "data/candidates/event_candidates.json."
        ),
    )
    _add_common_path_args(extract_parser)

    embed_parser = subparsers.add_parser(
        "embed-candidates",
        help=(
            "Load data/candidates/event_candidates.json and write "
            "data/vector_store/candidate_vectors.json, candidates.faiss, "
            "and candidate_metadata.json."
        ),
    )
    _add_common_path_args(embed_parser)

    group_parser = subparsers.add_parser(
        "group-candidates",
        help=(
            "Load data/vector_store/candidate_vectors.json and write "
            "data/vector_store/candidate_groups.json."
        ),
    )
    _add_common_path_args(group_parser)

    merge_parser = subparsers.add_parser(
        "merge-cases",
        help=(
            "Load data/candidates/event_candidates.json and "
            "data/vector_store/candidate_groups.json, then write "
            "data/cases/event_cases.json."
        ),
    )
    _add_common_path_args(merge_parser)

    report_parser = subparsers.add_parser(
        "generate-report",
        help=(
            "Load data/cases/event_cases.json and write "
            "data/reports/report.md from structured EventCase data."
        ),
    )
    _add_common_path_args(report_parser)
    return parser.parse_args(argv) # this parse_args is diffrent with above parse_args. This is method of parser(object of Class ArgumentParser) summarizing command and returnig argparse.namsesapce object
    


def _apply_path_overrides(args: argparse.Namespace) -> None:
    if args.data_dir is not None:
        os.environ["DATA_DIR"] = str(args.data_dir)
    if args.input_dir is not None:
        os.environ["INPUT_DIR"] = str(args.input_dir)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _apply_path_overrides(args)

    try:
        config = load_config() # load configures
        configure_logging(log_file=config.pipeline_log_path)
        os.environ.setdefault("LLM_METRICS_LOG_PATH", str(config.llm_metrics_log_path))
        pipeline = MeetingEventPipeline(config) # get object of MeetingEventPipeline Class

        if args.command == "run":
            result = pipeline.run()
            print("Pipeline completed successfully.")
            print(
                "- Parsed documents: "
                f"{result.parsed_document_count} -> {result.parsed_documents_path}"
            )
            print(f"- Segments: {result.segment_count} -> {result.segments_path}")
            print(
                "- Event candidates: "
                f"{result.event_candidate_count} -> {result.event_candidates_path}"
            )
            print(
                "- Candidate vectors: "
                f"{result.candidate_vector_count} -> {config.vector_store_dir / 'candidate_vectors.json'}"
            )
            print(f"- FAISS index: {result.faiss_index_path}")
            print(f"- Candidate metadata: {result.candidate_metadata_path}")
            print(
                "- Candidate groups: "
                f"{result.candidate_group_count} -> {result.candidate_groups_path}"
            )
            print(
                f"- Event cases: {result.event_case_count} -> "
                f"{result.event_cases_path}"
            )
            print(f"- Markdown report: {result.report_path}")
            return 0
        if args.command == "parse-and-segment":
            result = pipeline.parse_and_segment_inputs()
            print(
                f"Parsed documents: {result.parsed_document_count} -> {result.parsed_documents_path}"
            )
            print(f"Segments: {result.segment_count} -> {result.segments_path}")
            return 0

        if args.command == "extract-candidates":
            candidates = pipeline.extract_event_candidates_from_artifacts()
            print(
                f"Event candidates: {len(candidates)} -> {config.event_candidates_path}"
            )
            return 0

        if args.command == "embed-candidates":
            result = pipeline.embed_candidates_from_artifacts()
            print(
                f"Candidate vectors: {result.count} -> {config.vector_store_dir / 'candidate_vectors.json'}"
            )
            print(f"FAISS index: {result.faiss_index_path}")
            print(f"Candidate metadata: {result.candidate_metadata_path}")
            return 0

        if args.command == "group-candidates":
            result = pipeline.group_candidates_from_artifacts()
            print(
                f"Candidate groups: {result.group_count} -> {result.groups_path}"
            )
            return 0

        if args.command == "merge-cases":
            result = pipeline.merge_event_cases_from_artifacts()
            print(
                f"Saved {result.event_case_count} event case(s) to "
                f"{result.event_cases_path}"
            )
            return 0

        if args.command == "generate-report":
            result = pipeline.write_markdown_report_from_artifacts()
            print(
                f"Saved Markdown report for {result.event_case_count} event case(s) to "
                f"{result.report_path}"
            )
            return 0

        raise AssertionError(f"Unhandled command: {args.command}")
    except PipelineStageError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print(f"Stage: {exc.stage}", file=sys.stderr)
        print(f"Reason: {exc.reason}", file=sys.stderr)
        if args.debug:
            traceback.print_exc()
        return 1
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        if args.debug:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
