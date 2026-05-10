# Event Candidate Extraction Prompt

You extract event candidates from one meeting segment.
Return **only valid JSON**. Do not wrap the response in prose.

## Input

Segment JSON:

```json
$segment_json
```

## Output Schema

Return a JSON object with this shape:

```json
{
  "candidates": [
    {
      "candidate_id": "string",
      "meeting_id": "string",
      "segment_id": "string",
      "source_file": "string",
      "title": "string",
      "summary": "string",
      "occurred_at": "string or null",
      "actors": ["string"],
      "problem": "string or null",
      "discussion": "string or null",
      "action": "string or null",
      "result": "string or null",
      "status": "string",
      "evidence_text": "string",
      "keywords": ["string"],
      "embedding_text": "string"
    }
  ]
}
```

## Rules

- Use only the provided segment text and metadata.
- Do not infer exact dates that are not supported by the text; use `null` when unclear.
- `evidence_text` must be a direct supporting sentence or short passage from the segment.
- Preserve `meeting_id`, `segment_id`, and `source_file` exactly from the input.
- Build `embedding_text` from event identity fields: title, summary, problem, action, result, actors, and keywords.
- If the segment contains no event candidate, return `{ "candidates": [] }`.
- Return JSON only.
