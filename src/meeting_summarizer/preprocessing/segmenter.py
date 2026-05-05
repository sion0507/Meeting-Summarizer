"""Deterministic segment builder for parsed meeting documents.

Policy
------
- Default: create 2-page segments from parsed pages.
- TXT fallback: because TXT has no native page concept, build pseudo-pages from
  paragraph blocks (fixed number of paragraphs per pseudo-page), then apply the
  same page-based segmentation logic.

This design guarantees reproducible segment boundaries for identical inputs.
"""

from __future__ import annotations

from dataclasses import dataclass

from meeting_summarizer.schemas import ParsedDocument, Segment


@dataclass(slots=True)
class Segmenter:
    """Convert parsed documents into fixed-size segments."""

    segment_size_pages: int = 2
    txt_paragraphs_per_pseudo_page: int = 12

    def __post_init__(self) -> None:
        if self.segment_size_pages < 1:
            raise ValueError("segment_size_pages must be >= 1.")
        if self.txt_paragraphs_per_pseudo_page < 1:
            raise ValueError("txt_paragraphs_per_pseudo_page must be >= 1.")

    def build_segments(self, documents: list[ParsedDocument]) -> list[Segment]:
        """Build segments for multiple parsed documents."""
        segments: list[Segment] = []
        for document in documents:
            segments.extend(self.build_document_segments(document))
        return segments

    def build_document_segments(self, document: ParsedDocument) -> list[Segment]:
        """Build deterministic segments for a single parsed document."""
        page_units = self._page_units(document)

        segments: list[Segment] = []
        for start_idx in range(0, len(page_units), self.segment_size_pages):
            chunk = page_units[start_idx : start_idx + self.segment_size_pages]
            start_page = start_idx + 1
            end_page = start_idx + len(chunk)
            segment_id = f"segment_{(len(segments) + 1):03d}"
            text = "\n\n".join(chunk).strip()

            segments.append(
                Segment(
                    segment_id=segment_id,
                    meeting_id=document.meeting_id,
                    source_file=document.source_file,
                    start_page=start_page,
                    end_page=end_page,
                    text=text,
                )
            )

        return segments

    def _page_units(self, document: ParsedDocument) -> list[str]:
        if document.file_type.lower() == ".txt":
            pages = self._txt_pseudo_pages(document)
        else:
            pages = [page.strip() for page in document.pages if page.strip()]

        if not pages:
            raise ValueError(f"Document has no segmentable text: {document.source_file}")
        return pages

    def _txt_pseudo_pages(self, document: ParsedDocument) -> list[str]:
        """TXT fallback policy: fixed paragraph-count pseudo-pages.

        1) Split by blank-line-separated paragraph blocks.
        2) Group blocks by ``txt_paragraphs_per_pseudo_page``.
        3) Join each group into one pseudo-page.

        If no paragraph blocks are detected, fallback to raw text as one unit.
        """
        raw_text = "\n".join(document.pages).strip()
        if not raw_text:
            return []

        paragraphs = [block.strip() for block in raw_text.split("\n\n") if block.strip()]
        if not paragraphs:
            return [raw_text]

        pseudo_pages: list[str] = []
        chunk_size = self.txt_paragraphs_per_pseudo_page
        for start_idx in range(0, len(paragraphs), chunk_size):
            pseudo_pages.append("\n\n".join(paragraphs[start_idx : start_idx + chunk_size]))

        return pseudo_pages
