# Workspace Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize the research workspace, preserve experiment history, and add three short notebooks that expose the maintained workflow.

**Architecture:** Keep reusable research code as importable modules under role-based `scripts` subpackages. Keep current and historical artifacts in separate report directories, while notebooks remain thin callers of Python code rather than duplicate implementations.

**Tech Stack:** Python 3, pytest, Jupyter notebook JSON, pandas, scikit-learn, Google GenAI SDK

## Global Constraints

- Preserve every existing report, CSV, JSON, and JSONL research artifact.
- Preserve current edits to `scripts/run_pilot_all_3methods_paid.py`, `scripts/llm_token_tracker.py`, and `tests/test_llm_token_tracker.py`.
- Do not make paid LLM calls.
- Do not delete source RFP documents.
- Keep notebooks to one short introduction and at most five small code cells each.
- Use Git-aware moves so file history remains traceable.

---

### Task 1: Role-based Python package layout

**Files:**
- Create: `scripts/__init__.py`
- Create: `scripts/data/__init__.py`
- Create: `scripts/labeling/__init__.py`
- Create: `scripts/utilities/__init__.py`
- Move: `scripts/build_dataset.py` -> `scripts/data/build_dataset.py`
- Move: `scripts/preprocess_text.py` -> `scripts/data/preprocess_text.py`
- Move: `scripts/eda_requirements.py` -> `scripts/data/eda_requirements.py`
- Move: `scripts/sample_pilot.py` -> `scripts/data/sample_pilot.py`
- Move: `scripts/anchor_retriever.py` -> `scripts/labeling/anchor_retriever.py`
- Move: `scripts/validate_label_schema.py` -> `scripts/labeling/validate_label_schema.py`
- Move: `scripts/llm_token_tracker.py` -> `scripts/labeling/llm_token_tracker.py`
- Move: `scripts/run_labeling_pilot.py` -> `scripts/labeling/run_labeling_pilot.py`
- Move: `scripts/run_pilot_all_3methods_paid.py` -> `scripts/labeling/run_pilot_all_3methods_paid.py`
- Move: `scripts/run_pilot_experiment_3docs.py` -> `scripts/labeling/run_pilot_experiment_3docs.py`
- Move: `scripts/run_pilot_experiment_3docs_stratified.py` -> `scripts/labeling/run_pilot_experiment_3docs_stratified.py`
- Move: `scripts/update_notebook.py` -> `scripts/utilities/update_notebook.py`
- Modify: `tests/test_build_dataset.py`
- Modify: `tests/test_preprocess_text.py`
- Modify: `tests/test_eda.py`
- Modify: `tests/test_sample_pilot.py`
- Modify: `tests/test_label_schema.py`
- Modify: `tests/test_llm_token_tracker.py`

**Interfaces:**
- Consumes: existing public functions and classes without signature changes.
- Produces: imports under `scripts.data.*` and `scripts.labeling.*`; module CLI execution via `python -m scripts.<group>.<module>`.

- [ ] **Step 1: Update tests to import from the target package paths**

Example required mapping:

```python
from scripts.data.build_dataset import extract_document, validate
from scripts.labeling.llm_token_tracker import TokenTracker
from scripts.labeling.validate_label_schema import validate_label_output
```

- [ ] **Step 2: Run tests and confirm imports fail before moves**

Run: `python -m pytest tests -q`
Expected: collection errors for missing `scripts.data` and `scripts.labeling` packages.

- [ ] **Step 3: Create package markers and move modules with `git mv` where tracked**

For the untracked token tracker, move the file without overwriting it, then update local imports:

```python
from scripts.labeling.anchor_retriever import PureTfidfAnchorRetriever
from scripts.labeling.llm_token_tracker import TokenTracker, load_pricing_from_env
from scripts.labeling.validate_label_schema import validate_label_output
```

All modules must calculate repository root with `Path(__file__).resolve().parents[2]` after moving two levels below the root.

- [ ] **Step 4: Run package and CLI checks**

Run: `python -m pytest tests -q`
Expected: all tests pass.

Run: `python -m compileall -q scripts tests`
Expected: exit code 0.

- [ ] **Step 5: Commit the package layout**

```bash
git add scripts tests
git commit -m "refactor: organize research scripts by workflow"
```

### Task 2: Consolidated API diagnostics

**Files:**
- Create: `scripts/utilities/check_gemini.py`
- Create: `tests/test_check_gemini.py`
- Delete: `scripts/check_paid_quota.py`
- Delete: `scripts/check_prepay_status.py`
- Delete: `scripts/check_quotas.py`
- Delete: `scripts/test_call.py`
- Delete: `scripts/test_model.py`

**Interfaces:**
- Consumes: `GOOGLE_API_KEY` or `GEMINI_API_KEY` from `.env`.
- Produces: `build_parser() -> argparse.ArgumentParser`, `main(argv: list[str] | None = None) -> int`, and commands `models` and `smoke`.

- [ ] **Step 1: Write parser tests without making network calls**

```python
from scripts.utilities.check_gemini import build_parser

def test_models_command_parses():
    args = build_parser().parse_args(["models"])
    assert args.command == "models"

def test_smoke_command_requires_explicit_choice():
    args = build_parser().parse_args(["smoke"])
    assert args.command == "smoke"
```

- [ ] **Step 2: Run the new test and confirm it fails**

Run: `python -m pytest tests/test_check_gemini.py -q`
Expected: import failure because `check_gemini.py` does not exist.

- [ ] **Step 3: Implement one diagnostic CLI**

The module loads `.env`, creates `genai.Client(api_key=...)`, lists models for `models`, and sends one short prompt only for the explicit `smoke` command. It must return 2 with a clear message when no key exists and must not attempt billing or quota inference.

- [ ] **Step 4: Delete the five redundant scripts and run tests**

Run: `python -m pytest tests/test_check_gemini.py -q`
Expected: 2 tests pass without network access.

- [ ] **Step 5: Commit the utility consolidation**

```bash
git add scripts tests/test_check_gemini.py
git commit -m "refactor: consolidate Gemini diagnostics"
```

### Task 3: Current and archived research reports

**Files:**
- Create: `reports/current/README.md`
- Create: `reports/archive/README.md`
- Move current: `reports/eda_v0.2.0.*`, `reports/extraction_audit_v0.2.0.*`, `reports/extraction_freeze_v0.2.0.md`, `reports/extraction_readiness_v0.2.0.md`, `reports/labeling_pilot_results_v0.1.0.*`
- Move archive: `reports/extraction_audit_v0.1.0.*`, `reports/experiment_3docs_*.jsonl`, `reports/pilot_3docs_comparison_v0.1.0.md`
- Modify: scripts and docs containing old `reports/...` paths.

**Interfaces:**
- Consumes: existing tracked artifacts unchanged.
- Produces: stable current output location `reports/current/`; immutable historical location `reports/archive/`.

- [ ] **Step 1: Inventory every tracked report before moving**

Run: `git ls-files reports | Sort-Object`
Expected: 16 existing artifacts plus no lost files.

- [ ] **Step 2: Move artifacts with `git mv` and add index files**

`reports/current/README.md` identifies authoritative versions. `reports/archive/README.md` states that archived files are retained for reproducibility and should not be overwritten.

- [ ] **Step 3: Update hard-coded report paths**

Run: `rg -n 'reports/(eda_|extraction_|experiment_|labeling_|pilot_)' . -g '*.py' -g '*.md'`
Expected after edits: references point to `reports/current/` or `reports/archive/`; design and plan history may retain descriptive old-path text.

- [ ] **Step 4: Verify artifact count and run tests**

Run: `git ls-files reports | Measure-Object`
Expected: no existing artifact removed, plus two README files.

Run: `python -m pytest tests -q`
Expected: all tests pass.

- [ ] **Step 5: Commit report organization**

```bash
git add reports scripts docs README.md
git commit -m "chore: separate current and archived reports"
```

### Task 4: Minimal workflow notebooks

**Files:**
- Create: `notebooks/00_project_overview.ipynb`
- Create: `notebooks/01_dataset_pipeline.ipynb`
- Create: `notebooks/02_labeling_experiment.ipynb`
- Delete: `notebooks/eda_v0.2.0.ipynb` only after its generated EDA content is represented by the new dataset notebook and remains available in Git history.
- Create: `tests/test_notebooks.py`

**Interfaces:**
- Consumes: `scripts.data` and `scripts.labeling` modules from Tasks 1-2.
- Produces: three valid notebook JSON documents, each with at most six total cells and no paid call during normal execution.

- [ ] **Step 1: Write structural notebook tests**

```python
import json
from pathlib import Path

NOTEBOOKS = sorted(Path("notebooks").glob("*.ipynb"))

def test_expected_notebooks_exist():
    assert [p.name for p in NOTEBOOKS] == [
        "00_project_overview.ipynb",
        "01_dataset_pipeline.ipynb",
        "02_labeling_experiment.ipynb",
    ]

def test_notebooks_are_small_and_have_no_saved_outputs():
    for path in NOTEBOOKS:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        assert len(notebook["cells"]) <= 6
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                assert cell.get("outputs", []) == []
```

- [ ] **Step 2: Run structural tests and confirm they fail**

Run: `python -m pytest tests/test_notebooks.py -q`
Expected: notebook name assertion fails.

- [ ] **Step 3: Create the three notebooks**

Every notebook begins by locating the repository root and adding it to `sys.path`. The overview notebook inventories folders. The dataset notebook loads `data/processed/requirements_v0.1.0.jsonl` when available and runs `analyze_dataset`. The labeling notebook validates a fixed example, performs TF-IDF retrieval only when the sample pool exists, and demonstrates token-cost accounting with a fake response object.

- [ ] **Step 4: Remove saved outputs and run notebook tests**

Run: `python -m pytest tests/test_notebooks.py -q`
Expected: all notebook tests pass.

Run: `python -c "import json, pathlib; [json.loads(p.read_text(encoding='utf-8')) for p in pathlib.Path('notebooks').glob('*.ipynb')]"`
Expected: exit code 0.

- [ ] **Step 5: Commit notebooks**

```bash
git add notebooks tests/test_notebooks.py
git commit -m "docs: add minimal research workflow notebooks"
```

### Task 5: Workspace documentation and final verification

**Files:**
- Modify: `README.md`
- Modify: `docs/README.md`
- Modify: `data/README.md`
- Modify: `.gitignore`
- Modify: `requirements.txt`

**Interfaces:**
- Consumes: final paths and commands from Tasks 1-4.
- Produces: a UTF-8 root README with project map, setup, canonical commands, notebook guide, artifact policy, and current status.

- [ ] **Step 1: Replace the corrupted root README with concise UTF-8 Korean documentation**

Document these commands exactly:

```powershell
python -m pip install -r requirements.txt
python -m scripts.data.build_dataset
python -m scripts.data.eda_requirements
python -m pytest tests -q
python -m scripts.utilities.check_gemini models
```

- [ ] **Step 2: Update supporting documentation and dependency truth**

Add `python-dotenv` and `google-genai` to `requirements.txt` because maintained modules import them. Document `data/processed/` as generated, `reports/current/` as authoritative, and `reports/archive/` as immutable history.

- [ ] **Step 3: Tighten ignored generated files**

Keep `.env`, caches, virtual environments, notebook checkpoints, model binaries, `data/raw/*`, and `data/processed/*` ignored while retaining `.gitkeep` exceptions where present.

- [ ] **Step 4: Run complete verification**

Run: `python -m pytest tests -q`
Expected: all tests pass.

Run: `python -m compileall -q scripts tests`
Expected: exit code 0.

Run: `git diff --check`
Expected: exit code 0.

Run: `git status --short --ignored`
Expected: `.env`, caches, and `data/processed/` appear only with `!!`; no secret or generated data is staged.

- [ ] **Step 5: Commit final documentation**

```bash
git add README.md docs/README.md data/README.md .gitignore requirements.txt docs/superpowers/plans/2026-08-14-workspace-cleanup.md
git commit -m "docs: document the organized research workflow"
```
