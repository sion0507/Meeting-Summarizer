"""Document/segment schemas used before LLM stages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class ParsedDocument:
    """Parsed text/metadata for one source file."""

    meeting_id: str
    source_file: str
    file_type: str
    total_pages: int
    pages: list[str]

    def __post_init__(self) -> None:
        if not self.meeting_id.strip():
            raise ValueError("ParsedDocument.meeting_id must not be empty.")
        if not self.source_file.strip():
            raise ValueError("ParsedDocument.source_file must not be empty.")
        suffix = Path(self.source_file).suffix.lower()
        if suffix and suffix != self.file_type.lower():
            raise ValueError(
                "ParsedDocument.file_type must match source_file extension "
                f"({suffix} != {self.file_type})."
            )
        if self.total_pages < 1:
            raise ValueError("ParsedDocument.total_pages must be >= 1.")
        if len(self.pages) != self.total_pages:
            raise ValueError(
                "ParsedDocument.pages length must equal total_pages "
                f"({len(self.pages)} != {self.total_pages})."
            )


@dataclass(slots=True)
class Segment:
    """1-page unit (or TXT fallback unit) used for candidate extraction."""

    segment_id: str
    meeting_id: str
    source_file: str
    start_page: int
    end_page: int
    text: str

    def __post_init__(self) -> None:
        if not self.segment_id.strip():
            raise ValueError("Segment.segment_id must not be empty.")
        if not self.meeting_id.strip():
            raise ValueError("Segment.meeting_id must not be empty.")
        if self.start_page < 1:
            raise ValueError("Segment.start_page must be >= 1.")
        if self.end_page < self.start_page:
            raise ValueError("Segment.end_page must be >= start_page.")
        if not self.text.strip():
            raise ValueError("Segment.text must not be empty.")
