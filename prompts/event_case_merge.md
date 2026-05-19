# Event Case Merge Prompt

You merge a group of similar `EventCandidate` objects into one final `EventCase`.
Return **only valid JSON**. Do not wrap the response in prose.

## Output Schema

Return one JSON object matching this schema:

```json
{
  "case_id": "string",
  "title": "string",
  "summary": "string",
  "candidate_ids": ["string"],
  "related_meeting_ids": ["string"],
  "first_occurred_at": "string or null",
  "actors": ["string"],
  "occurrence": "string or null",
  "discussion": "string or null",
  "actions": ["string"],
  "result": "string or null",
  "status": "string",
  "remaining_issues": ["string"],
  "evidence": [
    {
      "evidence_id": "string",
      "candidate_id": "string",
      "meeting_id": "string",
      "segment_id": "string",
      "source_file": "string",
      "text": "string"
    }
  ],
  "timeline": [
    {
      "timeline_id": "string",
      "date": "string or null",
      "order": 0,
      "stage": "occurrence | discussion | action | result | status_update | unknown",
      "description": "string",
      "evidence_ids": ["string"]
    }
  ]
}
```

## Rules

- Use only the provided candidate JSON objects.
- Preserve every candidate ID in `candidate_ids`.
- Preserve traceability by creating evidence items from candidate evidence fields.
- Do not merge by simple concatenation; synthesize a coherent final event case.
- Do not invent dates, actors, actions, or results that are not supported by candidate evidence.
- Use `null` for uncertain date fields.
- Timeline `order` must start at 0 and increase by 1.
- Timeline `stage` must be one of: `occurrence`, `discussion`, `action`, `result`, `status_update`, `unknown`.
- Return JSON only.

## Input

Candidate group JSON:

```json
$candidate_group_json
```
