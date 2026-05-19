# Final Markdown Report Prompt

You write a Korean user-facing Markdown report from structured `EventCase` data.
Return **only Markdown text**. Do not use raw full meeting text.

## Required Event Format

For each event case, use this structure:

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

## Rules

- Use final `EventCase` data as the source of truth.
- Do not re-read, request, or rely on full original meeting text.
- Preserve evidence references by mentioning evidence IDs and source files where useful.
- If information is missing, write `확인 필요` rather than inventing details.
- Keep the report event-centered, not meeting-centered.

## Input

Event cases JSON:

```json
$event_cases_json
```
