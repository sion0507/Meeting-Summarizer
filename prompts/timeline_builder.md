# Timeline Builder Prompt

You refine the timeline for an existing `EventCase` using only structured case data.
Return **only valid JSON**. Do not wrap the response in prose.

## Output Schema

Return a JSON object with this shape:

```json
{
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

- Do not read or request original full meeting text.
- Use only the supplied event case, evidence IDs, and existing timeline hints.
- Do not hallucinate exact dates. Use `null` when dates are unclear.
- Preserve evidence references; every timeline item should cite relevant `evidence_ids` when possible.
- Sort by supported date when available; otherwise use evidence/candidate appearance order.
- Return JSON only.

## Input

Event case JSON:

```json
$event_case_json
```
