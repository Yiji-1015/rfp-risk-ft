import csv

from scripts.evaluation.binary_review import (
    collapse,
    collapse_rows,
    collapsed_three_class_reference,
    score,
)


def test_only_통상수용_stays_apart() -> None:
    assert collapse("통상수용") == "통상수용"
    assert collapse("견적반영") == "검토필요"
    assert collapse("계약·질의검토") == "검토필요"


def test_collapse_rows_does_not_touch_the_original() -> None:
    rows = [{"requirement_uid": "a", "primary_action": "견적반영"}]
    collapsed = collapse_rows(rows)

    assert collapsed[0]["primary_action"] == "검토필요"
    assert rows[0]["primary_action"] == "견적반영"


def test_score_fixes_the_denominator_when_a_class_is_missing() -> None:
    # 평가 문서에 검토필요가 하나도 없어도 macro 평균의 분모는 2로 유지된다.
    result = score(["통상수용", "통상수용"], ["통상수용", "통상수용"])

    assert result["accuracy"] == 1.0
    assert result["macro_f1"] == 0.5  # 검토필요 F1이 0으로 함께 평균된다
    assert result["review_recall"] == 0.0


def test_reference_collapses_the_saved_three_class_predictions(tmp_path) -> None:
    path = tmp_path / "oof.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["gold", "word_char_logistic_pred"])
        writer.writeheader()
        writer.writerows(
            [
                # 견적 ↔ 계약 혼동은 접으면 정답이 된다.
                {"gold": "견적반영", "word_char_logistic_pred": "계약·질의검토"},
                {"gold": "계약·질의검토", "word_char_logistic_pred": "견적반영"},
                {"gold": "통상수용", "word_char_logistic_pred": "통상수용"},
                # 통상수용과의 혼동만 오답으로 남는다.
                {"gold": "통상수용", "word_char_logistic_pred": "견적반영"},
            ]
        )

    result = collapsed_three_class_reference(path, "word_char_logistic")

    assert result["accuracy"] == 0.75
    assert result["review_recall"] == 1.0


def test_reference_is_absent_when_the_oof_file_is_not_there(tmp_path) -> None:
    assert collapsed_three_class_reference(tmp_path / "missing.csv", "x") is None
