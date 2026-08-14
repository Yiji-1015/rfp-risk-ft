# Workspace cleanup design

## Goal

Make the research workspace easy to understand and run without losing experiment history. Keep notebooks short and use them as guided entry points into the maintained Python code.

## Structure

```text
rfp-risk-ft/
|-- data/                 # generated datasets, samples, and review material
|-- docs/                 # project direction, decisions, and usage notes
|-- notebooks/            # short, numbered workflow notebooks
|-- reports/
|   |-- current/          # reports that describe the current dataset or experiment
|   `-- archive/          # superseded reports and historical experiment outputs
|-- scripts/
|   |-- data/             # extraction, preprocessing, sampling, and EDA
|   |-- labeling/         # schema validation, retrieval, and LLM experiments
|   `-- utilities/        # API diagnostics and maintenance helpers
|-- tests/
`-- RFP_data/             # source documents and converted Markdown
```

## File handling

- Move files with Git-aware renames whenever possible so history remains traceable.
- Preserve all existing report, CSV, and JSONL artifacts. Superseded outputs move to `reports/archive/` instead of being deleted.
- Consolidate redundant API diagnostic scripts into one utility. Deleted originals remain recoverable from Git history.
- Preserve the current token tracking work and its tests.
- Keep raw RFP files and their Markdown conversions together under `RFP_data/`.
- Keep generated `data/processed/`, secrets, caches, and notebook checkpoints ignored.

## Notebooks

Create three minimal notebooks:

1. `00_project_overview.ipynb`: directory map, current data/report inventory, and next commands.
2. `01_dataset_pipeline.ipynb`: load or build the requirements dataset, show a small sample, and run lightweight checks.
3. `02_labeling_experiment.ipynb`: demonstrate schema validation, anchor retrieval, and token tracking without making a paid API call by default.

Each notebook should contain one short introduction and no more than five small code cells. Reusable logic stays in Python modules.

## Compatibility

- Add package markers to moved script folders.
- Update internal imports and tests to the new paths.
- Keep command-line entry points directly runnable from the repository root.
- Document canonical commands in the root README.

## Verification

- Compile all maintained Python files.
- Run the complete test suite.
- Parse every notebook as JSON and execute only non-network smoke cells where practical.
- Confirm Git status contains no secrets, caches, or generated processed data.
- Confirm historical reports still exist under either `reports/current/` or `reports/archive/`.

## Non-goals

- No model redesign or new experiment methodology.
- No paid LLM calls.
- No conversion into a full `src/` package during this cleanup.
- No deletion of source RFP documents or research evidence.
