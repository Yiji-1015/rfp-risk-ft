import os
from types import SimpleNamespace

import pytest

from scripts.evaluation.text_masking_ablation import compare, run_arm
from scripts.labeling.label_dataset import TEXT_MASK_ENV


def _fold(macro_f1: float, review_recall: float = 0.5) -> SimpleNamespace:
    return SimpleNamespace(
        macro_f1=macro_f1,
        accuracy=macro_f1,
        review_recall=review_recall,
        review_precision=review_recall,
        per_class_f1={"계약·질의검토": review_recall},
    )


def test_fold_wins_counts_folds_not_the_average() -> None:
    # 평균은 올랐지만 이긴 fold는 하나뿐인 경우. 평균만 보면 효과로 오해한다.
    baseline = [_fold(0.60), _fold(0.60), _fold(0.60)]
    variant = [_fold(0.90), _fold(0.55), _fold(0.55)]

    result = compare(baseline, variant)

    assert result["macro_f1"]["difference"] == pytest.approx(0.0667, abs=1e-4)
    assert result["fold_wins"] == 1
    assert result["fold_count"] == 3


def test_comparison_reports_every_metric_against_the_baseline() -> None:
    result = compare([_fold(0.60, 0.40)], [_fold(0.62, 0.50)])

    assert result["review_recall"]["baseline"] == 0.40
    assert result["review_recall"]["variant"] == 0.50
    assert result["review_f1"]["difference"] == pytest.approx(0.10)


def test_run_arm_restores_the_environment_even_when_the_run_fails(monkeypatch) -> None:
    monkeypatch.setattr(
        "scripts.evaluation.text_masking_ablation.make_lodo_folds",
        lambda rows: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    monkeypatch.delenv(TEXT_MASK_ENV, raising=False)

    with pytest.raises(RuntimeError):
        run_arm([], object(), "josa", {})

    assert TEXT_MASK_ENV not in os.environ


def test_run_arm_puts_back_a_previously_set_mask(monkeypatch) -> None:
    monkeypatch.setattr(
        "scripts.evaluation.text_masking_ablation.make_lodo_folds", lambda rows: []
    )
    monkeypatch.setenv(TEXT_MASK_ENV, "ending")

    run_arm([], object(), "josa", {})

    assert os.environ[TEXT_MASK_ENV] == "ending"
