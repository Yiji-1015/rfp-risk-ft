# Claude Labeling Runner Implementation Plan

> **For Codex:** Execute this plan task-by-task. Keep paid API execution opt-in; tests must use fakes and make no network calls.

**Goal:** Replace the active Gemini labeling path with a small, provider-specific Claude runner whose parameters, schema, costs, and outputs are reproducible.

**Architecture:** Keep one canonical Pydantic label schema, one thin Anthropic client adapter, and one CLI runner. The CLI defaults to dry-run, supports a synchronous pilot, and prepares/submits a Message Batch only when `--execute` is explicit. Existing Gemini experiments move under `scripts/labeling/legacy/`; their report artifacts remain immutable under `reports/archive/`.

**Tech Stack:** Python 3.9+, `anthropic`, Pydantic, pytest, JSONL.

## Frozen API parameters

| Parameter | Pilot default | Full run | Reason |
|---|---|---|---|
| `model` | `claude-sonnet-5` | `claude-sonnet-5` | Primary quality path |
| comparison model | `claude-haiku-4-5-20251001` | none | 40-item speed/cost baseline only |
| `output_config.effort` | `medium` on Sonnet 5 | `medium` | Balanced quality/cost; keep constant for cache reuse |
| `max_tokens` | `4096` | `4096` | Includes adaptive-thinking tokens and structured answer |
| `temperature`, `top_p`, `top_k` | omitted | omitted | Sonnet 5 does not accept non-default sampling controls |
| `thinking` | omitted | omitted | Sonnet 5 adaptive thinking is the model default |
| `timeout` | 120 seconds | SDK submission/status calls: 120 seconds | Short structured requests; avoids SDK 10-minute default |
| `max_retries` | 2 | 2 | Anthropic SDK default for transient errors |
| prompt cache | ephemeral, 5 minutes | ephemeral, 5 minutes | Reused system prompt/schema; verify hits in usage logs |
| streaming | off | not applicable to Message Batches | Structured response is short |
| output format | Pydantic structured output | same | One schema and validation path |

Every response must record model, request ID, stop reason, token usage (including cache creation/read), latency, schema version, prompt version, and parameter snapshot. Treat refusal, `max_tokens`, invalid parsed output, or missing content as item failure; never silently coerce it.

### Task 1: Canonicalize the label schema

**Files:**
- Create: `scripts/labeling/label_schema.py`
- Modify: `scripts/labeling/validate_label_schema.py`
- Modify: `tests/test_label_schema.py`

1. Add failing tests for valid output, forbidden extra fields, confidence vocabulary, evidence list shape, and semantic constraints.
2. Run `pytest tests/test_label_schema.py -q` and confirm failure.
3. Implement a single strict Pydantic schema and expose `SCHEMA_VERSION` plus JSONL validation helpers.
4. Make `validate_label_schema.py` import that schema instead of maintaining another definition.
5. Run the focused tests and confirm pass.

### Task 2: Implement the Claude client adapter

**Files:**
- Create: `scripts/labeling/claude_client.py`
- Modify: `scripts/labeling/llm_token_tracker.py`
- Modify: `tests/test_llm_token_tracker.py`
- Create: `tests/test_claude_client.py`

1. Add fake-client tests covering the frozen request parameters, Sonnet effort inclusion, Haiku effort omission, parsed output, refusal, truncation, timeout/error propagation, and no import-time API-key requirement.
2. Add token tests for `cache_creation_input_tokens` and `cache_read_input_tokens`.
3. Run focused tests and confirm failure.
4. Add a settings dataclass, lazy Anthropic client construction, `messages.parse(..., output_format=LabelResult)`, 5-minute automatic prompt caching, and normalized response metadata.
5. Update token extraction without changing existing provider behavior.
6. Run focused tests and confirm pass.

### Task 3: Add a safe experiment runner

**Files:**
- Create: `scripts/labeling/run_claude_labeling.py`
- Create: `tests/test_run_claude_labeling.py`
- Create when reviewed by a human: `data/anchors/anchor_pool_v1.jsonl`

1. Add CLI tests for dry-run default, `--execute`, `--mode pilot|batch`, model allow-listing, budget guard, resume behavior, and missing anchor-pool failure.
2. Run the CLI tests and confirm failure.
3. Implement:
   - dry-run manifest generation with no client construction;
   - 40-item stratified pilot comparing Sonnet 5 and Haiku 4.5;
   - synchronous execution with per-item atomic JSONL checkpoints;
   - batch request generation/submission/status/result download for the selected Sonnet path;
   - explicit USD budget estimate/guard (`10` pilot, `50` full defaults);
   - output directories keyed by UTC run ID so prior results are never overwritten;
   - few-shot mode that refuses to run without a separately human-reviewed anchor pool.
4. Never promote existing LLM-generated pilot labels into the anchor pool automatically.
5. Run focused tests and confirm pass.

### Task 4: Retire the active Gemini path without losing history

**Files:**
- Move: `scripts/labeling/run_labeling_pilot.py` -> `scripts/labeling/legacy/run_labeling_pilot_gemini.py`
- Move: `scripts/labeling/run_pilot_all_3methods_paid.py` -> `scripts/labeling/legacy/run_pilot_all_3methods_paid_gemini.py`
- Move: `scripts/labeling/run_pilot_experiment_3docs.py` -> `scripts/labeling/legacy/run_pilot_experiment_3docs_gemini.py`
- Move: `scripts/labeling/run_pilot_experiment_3docs_stratified.py` -> `scripts/labeling/legacy/run_pilot_experiment_3docs_stratified_gemini.py`
- Create: `scripts/labeling/legacy/README.md`
- Modify: imports/tests affected by moves

1. Use Git-aware moves so history remains traceable.
2. Mark legacy scripts read-only/reference-only in the README; do not repair or execute them.
3. Ensure no active README command points at Gemini labeling.
4. Run import/collection checks.

### Task 5: Update dependencies, environment, notebooks, and docs

**Files:**
- Modify: `requirements.txt`
- Modify: `.env.example`
- Modify: `README.md`
- Modify: `docs/CLAUDE_API_MIGRATION_DECISIONS.md`
- Modify: `notebooks/02_labeling_experiment.ipynb`
- Modify: `tests/test_notebooks.py`

1. Add `anthropic` as the active labeling dependency. Keep `google-genai` only while the standalone Gemini diagnostic utility remains.
2. Document `ANTHROPIC_API_KEY`; never persist a real key.
3. Add the exact frozen-parameter table and commands for dry-run, pilot, batch submission, status, and resume.
4. Keep the notebook at six cells or fewer. It may validate data, build a dry-run manifest, and inspect saved results; it must not perform a paid call by default.
5. Run notebook structure/output checks.

### Task 6: Full verification and commit

1. Run `pytest -q` with the known-good Python runtime.
2. Run `python -m compileall scripts tests`.
3. Run `git diff --check`.
4. Run the Claude CLI in dry-run mode and verify no network/API-key dependency.
5. Inspect `git status --short`; confirm no report artifact was overwritten and no secret/cache file is tracked.
6. Commit the implementation as one coherent Claude migration commit.

## Evaluation gate before a paid full run

On the same 40 human-reviewed validation items, compare exact schema validity, reviewer agreement, critical-label recall, latency, and estimated cost. Select Haiku only if it stays within 2 percentage points of Sonnet on reviewer agreement and has no regression in critical-label recall; otherwise keep Sonnet 5. Full execution remains blocked until the user supplies `ANTHROPIC_API_KEY`, a human-reviewed anchor pool for few-shot experiments, and explicit approval of the estimated run cost.
