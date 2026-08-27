import numpy as np

from scripts.evaluation.candidate_ensemble import (
    agreement_summary,
    review_union_predictions,
    soft_vote,
)


def test_soft_vote_averages_aligned_probabilities():
    first = np.array([[0.8, 0.1, 0.1], [0.1, 0.7, 0.2]])
    second = np.array([[0.4, 0.4, 0.2], [0.2, 0.6, 0.2]])

    averaged = soft_vote([first, second])

    np.testing.assert_allclose(averaged, [[0.6, 0.25, 0.15], [0.15, 0.65, 0.2]])


def test_review_union_escalates_when_any_candidate_requests_review():
    predictions = [
        ["통상수용", "견적반영", "통상수용"],
        ["통상수용", "계약·질의검토", "견적반영"],
        ["통상수용", "견적반영", "통상수용"],
    ]

    result = review_union_predictions(predictions, ["통상수용"] * 3)

    assert result == ["통상수용", "계약·질의검토", "통상수용"]


def test_agreement_summary_reports_consensus_and_shared_errors():
    gold = ["통상수용", "견적반영", "계약·질의검토"]
    predictions = [
        ["통상수용", "통상수용", "견적반영"],
        ["통상수용", "계약·질의검토", "견적반영"],
        ["통상수용", "견적반영", "견적반영"],
    ]

    summary = agreement_summary(gold, predictions)

    assert summary["all_agree_count"] == 2
    assert summary["all_wrong_count"] == 1
    assert summary["at_least_one_correct_count"] == 2
