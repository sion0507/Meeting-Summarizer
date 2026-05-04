"""Stable ID generation helpers for pipeline entities."""

from __future__ import annotations

import hashlib
from pathlib import Path


def _slug(text: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in text).strip("-")


def _short_hash(*parts: str, size: int = 10) -> str:
    digest = hashlib.sha1("::".join(parts).encode("utf-8")).hexdigest()
    return digest[:size]


def make_meeting_id(source_file: str) -> str:
    stem = _slug(Path(source_file).stem)
    return f"meeting_{stem}_{_short_hash(source_file)}"


def make_segment_id(meeting_id: str, start_page: int, end_page: int) -> str:
    return f"segment_{meeting_id}_p{start_page:03d}-{end_page:03d}"


def make_candidate_id(meeting_id: str, segment_id: str, ordinal: int) -> str:
    return f"candidate_{meeting_id}_{segment_id}_{ordinal:03d}"


def make_case_id(group_candidate_ids: list[str]) -> str:
    normalized = sorted(group_candidate_ids)
    return f"case_{_short_hash(*normalized, size=12)}"


def make_evidence_id(case_id: str, candidate_id: str, ordinal: int) -> str:
    return f"evidence_{case_id}_{_short_hash(candidate_id)}_{ordinal:03d}"


def make_timeline_id(case_id: str, ordinal: int) -> str:
    return f"timeline_{case_id}_{ordinal:03d}"
