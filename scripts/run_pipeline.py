#!/usr/bin/env python
"""Command-line entrypoint for runnable MVP pipeline stages."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from meeting_summarizer.config import load_config  # noqa: E402
from meeting_summarizer.orchestrator import MeetingEventPipeline  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local meeting event analysis pipeline stages.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    merge_parser = subparsers.add_parser(
        "merge-cases",
        help=(
            "Load data/candidates/event_candidates.json and "
            "data/vector_store/candidate_groups.json, then write "
            "data/cases/event_cases.json."
        ),
    )
    merge_parser.add_argument(
        "--data-dir",
        type=Path,
        help="Override DATA_DIR for this run.",
    )

    report_parser = subparsers.add_parser(
        "generate-report",
        help=(
            "Load data/cases/event_cases.json and write "
            "data/reports/report.md from structured EventCase data."
        ),
    )
    report_parser.add_argument(
        "--data-dir",
        type=Path,
        help="Override DATA_DIR for this run.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.data_dir is not None:
        os.environ["DATA_DIR"] = str(args.data_dir)

    config = load_config()
    pipeline = MeetingEventPipeline(config)

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


if __name__ == "__main__":
    raise SystemExit(main())
