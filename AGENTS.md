# Project Agent Instructions

## Session startup

1. Inspect `git status` and the five most recent commits.
2. Read the root `README.md` and only the code, tests, and reports relevant to the request.
3. For research design, labeling, evaluation, or methodology work, read `docs/PROJECT_DIRECTION.md` and the highest-numbered `docs/history/decisions-*.md` volume.
4. Search all decision volumes for terms relevant to the task before making a new research decision. Read every volume only for thesis writing, repository-wide design review, audit, or when targeted search is insufficient.
5. Routine implementation work does not require loading the decision history.

## Decision history

- Treat `docs/PROJECT_DIRECTION.md` as the stable project charter. Change it only when the research objective or scope changes.
- Record a new research decision at the end of the highest-numbered `docs/history/decisions-*.md` file.
- Use a timestamp heading with no decision number or category:

```markdown
## YYYY-MM-DD HH:MM

- 결정: ...
- 이유: ...
- 논문 메모: ...
```

- When the latest volume is already 300 lines or longer, create the next `decisions-NN.md` volume before appending.
- Do not create an index, current-decision file, category hierarchy, or replacement-status table.
- Do not rewrite old entries. Use full-text search when an earlier rationale is needed.

## Documentation hygiene

- Keep current commands in the root `README.md`; code and configuration are the execution source of truth.
- Keep temporary plans and working notes out of permanent project documentation. Remove them after the task is complete.
