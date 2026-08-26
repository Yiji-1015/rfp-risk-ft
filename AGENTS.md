# Project Agent Instructions

## Session startup

1. Inspect `git status` and the five most recent commits. **Never infer project state from file
   modification times.** The working tree is often parked on a feature branch while newer work sits
   on `origin/main`; the files on disk can be days behind. Check the branch, and check whether the
   branch is behind its remote, before concluding that anything is missing or unrecorded.
   If a shell is unavailable, read `.git/HEAD`, `.git/refs/heads/*`, `.git/refs/remotes/origin/*`,
   and `.git/logs/HEAD` directly.
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
- Before appending to a decision volume, confirm the local copy matches the remote. Appending to a
  stale copy silently drops the entries added since.

## Documentation hygiene

- Keep current commands in the root `README.md`; code and configuration are the execution source of truth.
- Keep temporary plans and working notes out of permanent project documentation. Remove them after the task is complete.

## Before reporting a problem

This project's records are dense, and past sessions have repeatedly "found" problems that were
neither new nor problems. Run these four checks before writing anything into `docs/issues/` or
calling something a defect, a leak, or a confound.

1. **Is it already fixed?** Search `docs/history/decisions-*.md` and the relevant code for the
   thing you are about to report. A decision recorded hours earlier may already have implemented it.
2. **Is it intentional?** Much of what looks like contamination is the method. Few-shot anchors
   participate in label generation by design (결정 9·13·14·18); only same-document injection is
   blocked (결정 10). A gap between the design's own future plan (e.g. §8.4) and today's data is a
   known tradeoff, not a defect.
3. **What does the null look like?** Never report a fold-to-fold spread without computing the spread
   the design produces mechanically. A 16–44% range across folds meant nothing once the structural
   expectation turned out to be 25–43%.
4. **Would the answer change anything?** The requirement dataset, the anchor pool, and the label
   dataset are all frozen. If a finding cannot change data, code, or a claim in the thesis, it is a
   record, not a task. Say so instead of listing it as open work.

When an earlier report fails one of these checks, retract it the way 결정 34 did: leave the original
entry, append a correction that states what was observed and what was misnamed, and move the
surviving measurement into the decision record.

## What belongs in `docs/issues/`

Defects in the data as it stands, with numbers and a reproduction snippet. Not: intended design
properties, open research questions (those go in `PROJECT_DIRECTION.md`), or procedure
(`PIPELINE.md`). Retired issue numbers are not reused.
