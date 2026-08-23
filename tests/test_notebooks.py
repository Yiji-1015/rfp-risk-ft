import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"
EXPECTED_NOTEBOOKS = {
    "00_project_overview.ipynb",
    "01_dataset_pipeline.ipynb",
    "02_labeling_experiment.ipynb",
    "03_requirements_eda.ipynb",
    "04_anchor_pool_analysis.ipynb",
    "05_run_comparison.ipynb",
    "06_label_eda.ipynb",
    "07_baseline_comparison.ipynb",
}


def test_notebook_set_is_minimal_and_exact() -> None:
    actual = {path.name for path in NOTEBOOK_DIR.glob("*.ipynb")}
    assert actual == EXPECTED_NOTEBOOKS


def test_notebooks_are_valid_small_and_clear_of_saved_outputs() -> None:
    for path in NOTEBOOK_DIR.glob("*.ipynb"):
        notebook = json.loads(path.read_text(encoding="utf-8"))

        assert notebook["nbformat"] == 4
        assert 1 <= len(notebook["cells"]) <= 6
        assert notebook["cells"][0]["cell_type"] == "markdown"

        for cell in notebook["cells"]:
            assert cell["cell_type"] in {"markdown", "code"}
            if cell["cell_type"] == "code":
                assert cell.get("outputs") == []
                assert cell.get("execution_count") is None
