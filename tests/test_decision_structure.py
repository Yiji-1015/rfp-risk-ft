import numpy as np
import pytest
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier

from scripts.evaluation.baselines import WORD_CHAR_BALANCED
from scripts.evaluation.decision_structure import (
    ACCEPT,
    BOUNDARY,
    StructureResult,
    boundary_ceiling,
    result_rows,
    run_all,
    run_cascade,
    run_flat,
)
from scripts.evaluation.folds import make_lodo_folds
from scripts.labeling.label_dataset import load_label_dataset


@pytest.fixture(scope="module")
def rows():
    loaded, _ = load_label_dataset()
    return loaded


def _fake_fold_result(macro, recall):
    class Stub:
        macro_f1 = macro
        review_recall = recall

    return Stub()


class TestStructureResult:
    def test_averages_over_folds(self):
        result = StructureResult("x", (_fake_fold_result(0.6, 0.5), _fake_fold_result(0.8, 0.7)), 10, 4)
        assert result.macro_f1 == pytest.approx(0.7)
        assert result.review_recall == pytest.approx(0.6)

    def test_boundary_share_is_a_fraction_of_errors(self):
        assert StructureResult("x", (), 10, 4).boundary_share == pytest.approx(0.4)

    def test_no_errors_means_no_share_rather_than_division_by_zero(self):
        assert StructureResult("x", (), 0, 0).boundary_share == 0.0


class TestBoundaryConstants:
    def test_boundary_is_the_two_minority_labels(self):
        assert BOUNDARY == {"견적반영", "계약·질의검토"}
        assert ACCEPT not in BOUNDARY


class TestRunVariants:
    """실제 데이터로 돌린다. 값이 아니라 **계약**을 검사한다 — 점수는 기록에 남긴다."""

    def test_folds_partition_the_evaluated_set(self, rows):
        # 평가 대상은 전체 1,024건이 아니다. 동결 앵커는 LODO에서 빠진다.
        expected = sum(len(fold.split(rows)[2]) for fold in make_lodo_folds(rows))
        result = run_flat(rows, WORD_CHAR_BALANCED)
        assert len(result.folds) == 10
        assert sum(f.test_size for f in result.folds) == expected
        assert expected < len(rows)

    def test_boundary_errors_never_exceed_total_errors(self, rows):
        for result in (run_flat(rows, WORD_CHAR_BALANCED), run_cascade(rows, WORD_CHAR_BALANCED)):
            assert 0 <= result.boundary_errors <= result.errors

    def test_cascade_only_ever_emits_the_three_labels(self, rows):
        result = run_cascade(rows, WORD_CHAR_BALANCED)
        # 캐스케이드는 중간 라벨 `검토필요`를 만들지만 밖으로 내보내면 안 된다.
        assert result.errors > 0
        assert result.boundary_errors > 0

    def test_wrappers_change_the_prediction_but_keep_the_fold_layout(self, rows):
        base = run_flat(rows, WORD_CHAR_BALANCED)
        for wrapper in (OneVsRestClassifier, OneVsOneClassifier):
            other = run_flat(rows, WORD_CHAR_BALANCED, wrapper=wrapper, name=wrapper.__name__)
            assert len(other.folds) == len(base.folds)
            assert [f.test_document for f in other.folds] == [f.test_document for f in base.folds]

    def test_run_all_returns_every_variant(self, rows):
        results = run_all(rows, WORD_CHAR_BALANCED)
        assert len(results) == 4
        assert len({r.name for r in results}) == 4


class TestBoundaryCeiling:
    def test_beats_the_majority_baseline_but_is_not_perfect(self, rows):
        ceiling = boundary_ceiling(rows, WORD_CHAR_BALANCED)
        assert ceiling["n"] > 0
        assert ceiling["majority_baseline"] < ceiling["pooled_accuracy"] < 1.0
        assert ceiling["folds"] > 0

    def test_raises_when_no_boundary_rows_exist(self):
        with pytest.raises(ValueError):
            boundary_ceiling([{"primary_action": ACCEPT}], WORD_CHAR_BALANCED)


class TestResultRows:
    def test_rows_carry_both_the_score_and_the_boundary_count(self):
        (row,) = result_rows([StructureResult("x", (_fake_fold_result(0.6, 0.5),), 10, 4)])
        assert row["구성"] == "x"
        assert row["오답"] == 10
        assert row["경계 혼동"] == 4
        assert row["경계 비중"] == pytest.approx(0.4)
