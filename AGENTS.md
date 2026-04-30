# AGENTS.md

## Purpose of This Document

This document defines the implementation rules for Codex or any AI coding agent working on this repository.

The goal is not to describe the project to end users.  
The goal is to make sure the implementation follows the agreed architecture, data flow, schema, and development constraints.

When modifying the project, follow this document first.

---

## Project Goal

Build a local meeting-minutes event analysis system.

The system receives multiple meeting transcript/minutes files in `PDF`, `DOCX`, and `TXT` formats, extracts event candidates from segmented text, groups similar candidates, merges each group into a final event case, and outputs event-based reports in both `JSON` and `Markdown`.

The final report should organize each event by:

- occurrence
- discussion
- actions
- result
- current status
- remaining issues
- evidence

The first MVP does not include a web service or a Q&A interface.

---

## MVP Scope

### In Scope

The first implementation must support:

1. Loading multiple meeting files.
2. Parsing `PDF`, `DOCX`, and `TXT`.
3. Splitting parsed meeting text into segments.
4. Extracting event candidates from each segment with an LLM.
5. Creating embeddings for event candidates.
6. Storing vectors with FAISS.
7. Grouping event candidates whose similarity is greater than `0.8`.
8. Sending each candidate group to an LLM to produce one final event case.
9. Saving intermediate and final results as JSON.
10. Generating a user-facing Markdown report.

### Out of Scope for MVP

Do not implement these in the first MVP unless explicitly requested:

- Web UI
- API server
- PDF/HTML report generation
- HWP/HWPX parsing
- Meeting-agenda-based segmentation
- Full meeting-minutes Q&A
- Fine-tuning
- Database-backed persistence such as PostgreSQL or SQLite

The implementation may be designed so these features can be added later.

---

## Core Pipeline

The pipeline must follow this order:

```text
Input files
→ parsing / preprocessing
→ 2-page segment generation
→ event candidate extraction with LLM
→ embedding generation for candidates
→ FAISS vector storage
→ similarity-based candidate grouping
→ LLM-based event case merge
→ timeline organization
→ JSON output
→ Markdown report output
```

Parsing and preprocessing must be code-based.  
Do not use an LLM for parsing raw files.

---

## LLM Usage Rules

Use an LLM only in the following stages:

| Stage | LLM Usage |
|---|---|
| Parsing / preprocessing | Do not use LLM |
| Event candidate extraction | Use LLM |
| Event grouping search | Do not use LLM; use embeddings |
| Event case merge | Use LLM |
| Timeline organization | Use LLM |
| Final Markdown report generation | Use LLM |

Important rule:

The final report generation step must not re-read the full original meeting text.  
It must use structured event data such as event cases, evidence spans, and timelines.

---

## Local Model Configuration

The target runtime is local.

Default target model:

```text
LLM: Qwen2.5 14B Q4 quantized model
GPU: RTX 5080 16GB
```

The implementation must keep the LLM provider configurable.  
Do not hard-code the model path directly into business logic.

Use a config file or environment variables for model-related settings.

Example:

```env
LLM_PROVIDER=local
LOCAL_MODEL_PATH=./models/qwen2.5-14b-q4.gguf
EMBEDDING_MODEL=
VECTOR_STORE=faiss
```

The embedding model is not fixed yet.  
Design the embedding module so the model can be replaced later.

---

## Input File Policy

The MVP supports:

- `.txt`
- `.docx`
- `.pdf`

Do not implement `.hwp` or `.hwpx` in the MVP.

If unsupported files are provided, return a clear error.

---

## Parsing and Segmentation Policy

### Parsing

Parsing must extract text and useful metadata from each source file.

The exact parsed-result storage structure is not fixed in advance.  
The implementation agent may design the structure that is most convenient for later modules.

However, the chosen structure must be documented in code, schema files, or project documentation.

### Segmentation

The default segmentation rule is:

```text
Split meeting text into 2-page segments.
```

Expected examples:

```text
segment_001: pages 1-2
segment_002: pages 3-4
segment_003: pages 5-6
```

For `TXT` files, there is no native page concept.  
The implementation agent must choose a reasonable fallback method and document it clearly.

Possible fallback examples:

- split by character length
- split by token estimate
- split by paragraph count

Do not implement agenda-based segmentation in the MVP.  
It may be added later as an advanced feature.

---

## Required Data Schemas

Use two main schema layers:

1. `EventCandidate`
2. `EventCase`

These schemas are core to the project.  
Do not casually rename fields.

Python dataclasses or Pydantic models are recommended.

---

## EventCandidate Schema

`EventCandidate` represents one event candidate extracted from one segment.

```python
class EventCandidate:
    candidate_id: str
    meeting_id: str
    segment_id: str
    source_file: str

    title: str
    summary: str

    occurred_at: str | None
    actors: list[str]

    problem: str | None
    discussion: str | None
    action: str | None
    result: str | None

    status: str
    evidence_text: str
    keywords: list[str]
    embedding_text: str
```

### Field Notes

- `candidate_id`: unique ID for the extracted candidate.
- `meeting_id`: ID of the source meeting document.
- `segment_id`: ID of the segment where the candidate was found.
- `source_file`: original filename.
- `title`: short event title.
- `summary`: concise description of the candidate event.
- `occurred_at`: date or time expression if available; otherwise `None`.
- `actors`: people, teams, or organizations involved.
- `problem`: issue, incident, or triggering situation.
- `discussion`: discussion related to the event.
- `action`: action or follow-up mentioned in the segment.
- `result`: result or outcome if mentioned.
- `status`: current status inferred from the segment.
- `evidence_text`: direct supporting sentence or passage from the segment.
- `keywords`: keywords for search and grouping.
- `embedding_text`: text used to generate the candidate embedding.

`embedding_text` should be built from fields that represent event identity, such as title, summary, problem, action, result, actors, and keywords.

---

## EventCase Schema

`EventCase` represents one final merged event.

```python
class EventCase:
    case_id: str
    title: str
    summary: str

    candidate_ids: list[str]
    related_meeting_ids: list[str]

    first_occurred_at: str | None
    actors: list[str]

    occurrence: str | None
    discussion: str | None
    actions: list[str]
    result: str | None

    status: str
    remaining_issues: list[str]
    evidence: list[EvidenceSpan]
    timeline: list[TimelineItem]
```

The implementation must define `EvidenceSpan` and `TimelineItem` in a way that supports this schema.

Recommended minimum shape:

```python
class EvidenceSpan:
    evidence_id: str
    candidate_id: str
    meeting_id: str
    segment_id: str
    source_file: str
    text: str
```

```python
class TimelineItem:
    timeline_id: str
    date: str | None
    order: int
    stage: str
    description: str
    evidence_ids: list[str]
```

Allowed `stage` values should be simple and report-friendly:

```text
occurrence
discussion
action
result
status_update
unknown
```

---

## Event Grouping and Merge Rules

Event grouping is embedding-based.

### Similarity Rule

Use candidate title and/or `embedding_text` to generate embeddings.

Group candidates as the same event pool if their embedding similarity is:

```text
similarity > 0.8
```

Candidates that do not exceed the threshold with any other candidate should remain as single-candidate groups.

### Merge Rule

For each candidate group:

1. Collect the candidate JSON objects.
2. Send them to the event merge LLM prompt.
3. Require the LLM to output one valid `EventCase`.
4. Validate the output against the schema.
5. Save the merged case as JSON.

The LLM is responsible for synthesizing the final case from the grouped candidates.

Do not merge candidates into an `EventCase` only by string concatenation.

---

## Date Handling Policy

Do not implement separate code-based date normalization in the MVP.

All date interpretation should be handled by the LLM during event extraction, event merge, or timeline generation.

If the date is unclear, uncertain, or only relative, prefer conservative output:

```python
date = None
```

For early timeline ordering, it is acceptable to use the first meeting appearance order as a fallback.

Do not hallucinate exact dates that are not supported by the meeting data.

---

## Storage Policy

The system must work without a web service or external database.

Use:

- JSON files for structured intermediate and final data.
- FAISS for vector storage.

The implementation may choose exact directory names, but the structure must be documented.

Recommended structure:

```text
data/
  raw/
  parsed/
  segments/
  candidates/
  cases/
  reports/
  vector_store/
```

The implementation must save enough intermediate data to debug the pipeline without re-running every step.

---

## Output Policy

The system must produce two main outputs:

1. Internal structured JSON
2. User-facing Markdown report

### JSON Output

JSON output should include:

- parsed document metadata
- segments
- event candidates
- candidate groups
- final event cases
- timeline items
- evidence spans

### Markdown Report Output

The Markdown report should follow this event-centered format:

```markdown
## [Event Title]

- 최초 발생: ...
- 관련 회의록: ...
- 사건 내용: ...
- 처리 과정: ...
- 담당자: ...
- 최종 결과: ...
- 현재 상태: ...
- 남은 이슈: ...
- 근거: ...
```

Use the final `EventCase` data as the source of truth.

Do not generate the final report directly from raw meeting text.

---

## Prompt Management

Prompts must be stored as separate files.

Do not embed long prompts directly inside Python logic.

Recommended structure:

```text
prompts/
  event_extraction.md
  event_case_merge.md
  timeline_builder.md
  final_report.md
```

Each prompt should clearly specify:

- input format
- output schema
- rules for missing information
- rules against hallucination
- requirement to preserve evidence references

---

## Suggested Repository Structure

Use the existing folder structure from the repository as the starting point.

If the implementation agent changes the folder structure, it must also update this `AGENTS.md` file.

Recommended structure:

```text
meeting-summarizer/
  README.md
  AGENTS.md
  pyproject.toml
  .env.example

  src/
    meeting_summarizer/
      __init__.py
      config.py
      orchestrator.py

      parsers/
        __init__.py
        base.py
        pdf_parser.py
        docx_parser.py
        txt_parser.py

      preprocessing/
        __init__.py
        segmenter.py

      schemas/
        __init__.py
        document.py
        event_candidate.py
        event_case.py

      agents/
        __init__.py
        event_extractor.py
        event_merger.py
        timeline_builder.py
        report_writer.py

      embeddings/
        __init__.py
        embedder.py

      linking/
        __init__.py
        faiss_store.py
        grouping.py

      storage/
        __init__.py
        json_store.py

      reporting/
        __init__.py
        markdown_report.py

      utils/
        __init__.py
        ids.py
        logging.py

  prompts/
    event_extraction.md
    event_case_merge.md
    timeline_builder.md
    final_report.md

  tests/
    unit/
    integration/

  data/
    raw/
    parsed/
    segments/
    candidates/
    cases/
    reports/
    vector_store/

  scripts/
    run_pipeline.py
```

---

## Module Responsibilities

### `parsers/`

Responsible for extracting text and metadata from supported files.

Must not call an LLM.

### `preprocessing/`

Responsible for segmenting parsed text into 2-page chunks or documented TXT fallback chunks.

Must not call an LLM.

### `schemas/`

Responsible for defining structured data models.

Keep schemas stable and explicit.

### `agents/`

Responsible for LLM-based work:

- event candidate extraction
- final event case merge
- timeline organization
- final report drafting support

### `embeddings/`

Responsible for embedding model loading and embedding generation.

The embedding model must be replaceable.

### `linking/`

Responsible for FAISS vector storage, similarity search, and grouping.

Do not put LLM calls directly in similarity search code.

### `storage/`

Responsible for reading and writing JSON artifacts.

### `reporting/`

Responsible for generating Markdown reports from final `EventCase` data.

### `orchestrator.py`

Responsible for connecting all modules into the full pipeline.

It should not contain heavy parsing, LLM prompt logic, or vector search internals directly.

---

## Testing Policy

The implementation agent decides the concrete test design.

However, after implementation or major changes, it must report:

1. Test goal
2. Test process
3. Test result

Recommended tests:

- parser tests for TXT, DOCX, PDF
- segmenter tests for 2-page chunking
- schema validation tests
- event candidate extraction output validation
- grouping threshold tests
- event merge output validation
- Markdown report generation tests
- end-to-end pipeline test with sample files

If tests cannot be run, explain why and what was checked instead.

---

## Q&A Feature Policy

Do not implement Q&A in the first MVP.

However, keep the data structure suitable for future Q&A by preserving:

- event cases
- evidence spans
- candidate IDs
- meeting IDs
- segment IDs
- vector store entries

Future Q&A should use structured event cases and timelines first, not raw full meeting text by default.

---

## Coding Rules

Follow these rules:

1. Keep modules small and separated by responsibility.
2. Do not mix parsing, LLM calls, vector search, and report writing in one file.
3. Use type hints.
4. Prefer dataclasses or Pydantic models for schemas.
5. Validate LLM JSON outputs before saving them.
6. Save intermediate artifacts for debugging.
7. Keep prompts outside code.
8. Make model paths and thresholds configurable.
9. Do not hard-code user-specific absolute paths.
10. Update documentation when changing architecture, schema, or folder structure.

---

## Configuration Rules

Important values should be configurable:

- input directory
- output directory
- LLM provider
- local model path
- embedding model name
- FAISS index path
- similarity threshold
- segment size

Default similarity threshold:

```text
0.8
```

Default segment size:

```text
2 pages
```

---

## Error Handling Rules

The system should fail clearly.

Examples:

- unsupported file type
- failed parsing
- missing model path
- invalid LLM JSON output
- FAISS index load failure
- empty input directory
- no event candidates found

Error messages should be understandable to a developer running the pipeline locally.

---

## Documentation Update Rules

Update this file when changing:

- pipeline order
- schema fields
- folder structure
- LLM usage rules
- storage format
- similarity threshold policy
- supported input formats
- output format

Update `README.md` when changing:

- project description
- setup instructions
- run commands
- user-facing features
- output examples

---

## Non-Negotiable Constraints

Do not violate these constraints:

1. Do not use LLM for parsing or preprocessing.
2. Do not generate final reports directly from full raw meeting text.
3. Do not skip JSON schema validation for LLM outputs.
4. Do not hard-code local machine paths.
5. Do not implement web service features in MVP.
6. Do not implement Q&A in MVP.
7. Do not add HWP/HWPX support in MVP unless explicitly requested.
8. Do not silently change core schema field names.
9. Do not bury prompts inside Python business logic.
10. Do not merge event candidates without preserving evidence references.

---

## Expected Final Output

A successful run should produce artifacts similar to:

```text
data/
  parsed/
    parsed_documents.json
  segments/
    segments.json
  candidates/
    event_candidates.json
  cases/
    event_cases.json
  reports/
    report.md
  vector_store/
    candidates.faiss
    candidate_metadata.json
```

The exact structure may differ if documented, but the implementation must preserve the same logical outputs.

---

## Current Design Summary

The project is a local, file-based, multi-stage event analysis pipeline.

It does not aim to summarize meetings one by one.  
It aims to identify events scattered across multiple meeting records, connect similar event mentions, and produce final event cases with evidence and timeline structure.

The central unit of analysis is not a meeting.  
The central unit of analysis is an event case.
