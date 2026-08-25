import pytest

from scripts.evaluation.baselines import (
    CHAR_BALANCED,
    CHAR_UNWEIGHTED,
    DUMMY,
    LABELS,
    Comparison,
    ModelSpec,
    run_lodo,
    summarize,
)
from scripts.labeling.label_dataset import load_label_dataset


# LODO 한 번이 학습 10회다. 같은 설정을 테스트마다 다시 돌리면 스위트가 1분을 넘고,
# 느린 테스트는 결국 안 돌리게 된다. 설정별로 한 번만 돌려 재사용한다.
@pytest.fixture(scope="module")
def rows():
    data, _ = load_label_dataset()
    return data


@pytest.fixture(scope="module")
def char_results(rows):
    return run_lodo(rows, CHAR_BALANCED)


@pytest.fixture(scope="module")
def dummy_results(rows):
    return run_lodo(rows, DUMMY)


@pytest.fixture(scope="module")
def unweighted_results(rows):
    return run_lodo(rows, CHAR_UNWEIGHTED)


@pytest.fixture(scope="module")
def nine_document_results(rows):
    return run_lodo(rows, CHAR_BALANCED, use_nine_documents=True)


def test_dummy_never_predicts_the_review_label(dummy_results):
    """
    최빈 클래스만 찍는 모델은 `계약·질의검토`를 한 번도 예측하지 않으므로 recall이 0이다.
    §10.2가 이 라벨의 recall을 따로 요구하는 이유가 여기 있다. 정확도만 보면 Dummy도
    49%를 내지만, 실무에서 정작 검토해야 할 조항은 **하나도 못 찾는다.**
    """
    summary = summarize(dummy_results)

    assert summary["review_recall"]["fold_mean"] == 0.0
    assert summary["accuracy"]["fold_mean"] > 0.4


def test_macro_f1_punishes_the_dummy_far_harder_than_accuracy(dummy_results):
    """
    같은 예측을 지표에 따라 다르게 읽는다는 것을 고정한다. Dummy는 정확도로는
    0.49지만 macro F1로는 0.21이다. 세 클래스 중 둘의 F1이 0이기 때문이다.
    accuracy를 주 지표로 쓰면 안 되는 이유를 숫자로 남긴다(§10.2).
    """
    summary = summarize(dummy_results)

    assert summary["accuracy"]["fold_mean"] > 2 * summary["macro_f1"]["fold_mean"]


def test_char_tfidf_clears_the_dummy_by_a_wide_margin(char_results):
    """
    기준선을 넘지 못하면 모델이 텍스트에서 아무것도 배우지 못한 것이다. 여기서는
    macro F1이 0.21에서 0.60으로 오르므로 표현이 실제로 신호를 담고 있다.
    """
    summary = summarize(char_results)

    assert summary["macro_f1"]["fold_mean"] > 0.55
    assert summary["lift_over_dummy"]["fold_mean"] > 0.1


def test_class_weight_is_a_real_effect_but_the_ninth_document_is_noise(
    char_results, unweighted_results, nine_document_results
):
    """
    이 프로젝트에서 "효과"와 "잡음"을 가르는 기준을 고정한다.

    문서가 10개뿐이라 fold 간 분산이 크다. 평균 차이가 편차 폭에 비해 작으면 그것은
    효과가 아니다. class_weight는 평균 +0.078로 효과이고, 검증 문서 한 개를 학습에
    더 넣는 것은 평균 +0.002로 잡음이다. **크기 차이가 40배다.**
    """
    weight_deltas = [
        b.macro_f1 - a.macro_f1 for a, b in zip(unweighted_results, char_results)
    ]
    weight_mean = sum(weight_deltas) / len(weight_deltas)
    assert weight_mean > 0.05
    assert sum(1 for d in weight_deltas if d > 0) >= 8

    deltas = [
        b.macro_f1 - a.macro_f1 for a, b in zip(char_results, nine_document_results)
    ]
    mean_delta = sum(deltas) / len(deltas)

    assert abs(mean_delta) < 0.01
    # 방향조차 일정하지 않다. 일정했다면 작아도 효과일 수 있다.
    assert 0 < sum(1 for d in deltas if d > 0) < len(deltas)


def test_holding_out_the_validation_document_costs_almost_nothing(
    char_results, nine_document_results
):
    """
    8문서로 통일한 결정의 근거다. 손해가 잡음 수준이므로, 나중에 파인튜닝과 나란히
    놓기 위한 비교 가능성을 사실상 공짜로 얻는다(§9.3).
    """
    eight = summarize(char_results)["macro_f1"]["fold_mean"]
    nine = summarize(nine_document_results)["macro_f1"]["fold_mean"]

    assert abs(nine - eight) < 0.01


def test_every_fold_result_carries_its_own_diagnostics(char_results):
    """
    fold 점수는 그 자체로 비교할 수 없다. 난이도와 점수 구성이 fold마다 다르고
    (issues/003, 006) 둘이 독립도 아니기 때문에, 점수만 떼어 비교하면 능력이 아니라
    구성을 보게 된다. 그래서 결과 객체가 진단값을 항상 함께 들고 있어야 한다.
    """
    for result in char_results:
        assert 0.0 <= result.oracle_majority_accuracy <= 1.0
        assert result.trained_majority_accuracy <= result.oracle_majority_accuracy
        assert 0.0 <= result.repeat_exposure_rate <= 1.0
        assert set(result.per_class_f1) == set(LABELS)


def test_three_way_scores_report_counts_so_tiny_subsets_are_visible(char_results):
    """
    결정 34의 세 갈래 보고에서 "반복만" 부분집합은 fold에 따라 1건에서 44건까지
    흔들린다. genai는 1건이라 점수가 0.000 아니면 1.000밖에 나올 수 없고, ccrs는
    0건이라 점수 자체가 없다. 비율만 적으면 이 사실이 보이지 않으므로 건수를 함께 낸다.
    """
    by_document = {r.test_document: r for r in char_results}

    ccrs = by_document["ccrs_ai_platform"]
    assert ccrs.repeat_count == 0
    assert ccrs.macro_f1_repeat_only is None  # 0.0으로 채우면 "성능 0"으로 오독된다

    mfds = by_document["mfds_drug_ai_review"]
    assert mfds.repeat_count > 0
    assert mfds.macro_f1_repeat_only is not None

    for result in char_results:
        assert result.repeat_count + result.non_repeat_count == result.test_size


def test_repeated_phrases_are_easier_than_the_rest_where_there_are_enough_of_them(
    char_results,
):
    """
    결정 34가 "점수에 두 능력이 섞인다"고 한 것의 실측이다. 반복 노출이 가장 높은
    mfds(22.9%)에서 반복 문구 점수와 나머지 점수가 크게 갈린다. 전체 점수 하나만
    보고 "처음 보는 RFP에 일반화된다"고 말하면 그 주장이 실제보다 강해 보인다.
    """
    mfds = next(r for r in char_results if r.test_document == "mfds_drug_ai_review")

    assert mfds.macro_f1_repeat_only > mfds.macro_f1_repeat_excluded
    assert mfds.macro_f1_repeat_excluded < mfds.macro_f1


def test_unknown_classifier_fails_loudly():
    """조용히 기본 분류기로 넘어가면 manifest와 실제 실행이 어긋난다."""
    with pytest.raises(ValueError, match="알 수 없는 분류기"):
        ModelSpec(name="오타", classifier="logisitc").build()


def test_summarize_reports_both_aggregations(char_results):
    """하나만 쓰면 어느 쪽이든 오해를 만든다(issues/003)."""
    summary = summarize(char_results)

    for metric in ("macro_f1", "accuracy", "review_recall", "lift_over_dummy"):
        assert set(summary[metric]) == {"fold_mean", "count_weighted"}


def _comparison(deltas):
    """차이만 지정해 Comparison을 만든다. 판정 로직만 보므로 학습은 하지 않는다."""
    return Comparison(
        baseline=CHAR_UNWEIGHTED,
        variant=CHAR_BALANCED,
        metric="macro_f1",
        per_fold=[(f"doc{i}", 0.5, 0.5 + d) for i, d in enumerate(deltas)],
    )


def test_a_small_mean_next_to_a_wide_spread_is_called_noise():
    """
    실측한 8문서 -> 9문서다. 평균 +0.002인데 fold별로는 -0.046 ~ +0.037이다.
    평균만 보면 "조금 올랐다"로 읽히지만 방향조차 일정하지 않다.
    """
    verdict = _comparison(
        [-0.011, 0.009, 0.002, 0.004, 0.037, 0.031, 0.012, -0.007, -0.046, -0.011]
    )

    assert verdict.looks_like_noise
    assert abs(verdict.mean_delta) < 0.01


def test_a_large_consistent_mean_is_called_an_effect():
    """
    실측한 class_weight None -> balanced다. 평균 +0.078에 8/10 fold에서 우세하다.
    편차 폭이 커도 평균이 그에 비해 충분히 크면 효과로 본다.
    """
    verdict = _comparison(
        [0.185, 0.178, 0.110, -0.001, 0.134, 0.055, 0.033, -0.072, 0.097, 0.059]
    )

    assert not verdict.looks_like_noise
    assert verdict.variant_wins == 8
