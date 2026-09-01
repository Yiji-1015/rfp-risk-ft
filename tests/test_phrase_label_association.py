import pytest

from scripts.evaluation.phrase_label_association import (
    build_table,
    label_view,
    stem_approx,
    tokenize,
    write_xlsx,
)


def _row(uid: str, document: str, label: str, text: str) -> dict:
    return {
        "requirement_uid": uid,
        "document_id": document,
        "primary_action": label,
        "model_text": text,
    }


def test_tokenize_strips_edge_symbols_and_short_tokens() -> None:
    assert tokenize("- 제안사는 [성능] 기준을 A 제시하여야 한다.") == {
        "제안사는",
        "성능",
        "기준을",
        "제시하여야",
        "한다",
    }


def test_stem_approx_removes_particles_but_keeps_two_letter_words() -> None:
    assert stem_approx("제안사는") == "제안사"
    assert stem_approx("제안사가") == "제안사"
    assert stem_approx("공사에서") == "공사"
    assert stem_approx("정의") == "정의"  # "의"를 떼면 한 글자라 그대로 둔다
    assert stem_approx("구축") == "구축"


def test_lift_is_measured_against_the_base_rate() -> None:
    rows = [
        _row("a", "doc1", "계약·질의검토", "제안사는 시스템 하자보수를 이행하여야 한다"),
        _row("b", "doc2", "계약·질의검토", "제안사는 성능을 보장하여야 한다"),
        _row("c", "doc1", "통상수용", "시스템 사용자 관리 기능을 개발한다"),
        _row("d", "doc2", "통상수용", "표준 산출물을 제출한다"),
    ]
    table = {record["단어"]: record for record in build_table(rows, {}, min_count=1)}

    # 기본 비율 50%인 라벨에 100% 몰리면 lift는 2.0이다.
    assert table["제안사는"]["총건수"] == 2
    assert table["제안사는"]["문서수"] == 2
    assert table["제안사는"]["계약·질의검토_lift"] == 2.0
    assert table["제안사는"]["최대lift_라벨"] == "계약·질의검토"

    # 두 라벨에 고르게 걸친 단어는 lift 1.0이다.
    assert table["시스템"]["계약·질의검토_lift"] == 1.0
    assert table["시스템"]["통상수용_lift"] == 1.0


def test_model_contribution_is_joined_by_surface_form() -> None:
    rows = [
        _row("a", "doc1", "계약·질의검토", "제안사는 성능을 보장하여야 한다"),
        _row("b", "doc1", "통상수용", "산출물을 제출한다"),
    ]
    contributions = {"제안사는": {"계약·질의검토": 0.7}}
    table = {
        record["단어"]: record for record in build_table(rows, contributions, min_count=1)
    }

    assert table["제안사는"]["모델기여_계약·질의검토"] == 0.7
    assert table["제안사는"]["모델기여_합"] == 0.7
    assert table["산출물을"]["모델기여_합"] == 0.0
    # 정렬은 모델 기여가 큰 단어부터다.
    assert build_table(rows, contributions, min_count=1)[0]["단어"] == "제안사는"


def test_label_view_keeps_one_label_and_sorts_by_its_contribution() -> None:
    rows = [
        _row("a", "doc1", "계약·질의검토", "제안사는 성능을 보장하여야 한다"),
        _row("b", "doc1", "통상수용", "표준 산출물을 제출한다"),
    ]
    contributions = {"제안사는": {"계약·질의검토": 0.7}, "표준": {"통상수용": 0.4}}
    table = build_table(rows, contributions, min_count=1)

    review = label_view(table, "계약·질의검토")
    assert review[0]["단어"] == "제안사는"
    assert review[0]["모델기여"] == 0.7
    assert review[0]["lift"] == 2.0
    # 다른 라벨의 열은 남지 않는다.
    assert "통상수용_lift" not in review[0]

    accept = label_view(table, "통상수용")
    assert accept[0]["단어"] == "표준"
    assert accept[0]["모델기여"] == 0.4


def test_xlsx_has_one_sheet_per_label(tmp_path) -> None:
    openpyxl = pytest.importorskip("openpyxl")
    rows = [
        _row("a", "doc1", "계약·질의검토", "제안사는 성능을 보장하여야 한다"),
        _row("b", "doc1", "통상수용", "표준 산출물을 제출한다"),
    ]
    path = tmp_path / "association.xlsx"
    write_xlsx(build_table(rows, {}, min_count=1), path)

    book = openpyxl.load_workbook(path)
    assert book.sheetnames == ["통상수용", "견적반영", "계약·질의검토"]
    sheet = book["계약·질의검토"]
    assert sheet["A1"].value == "단어"
    assert sheet.freeze_panes == "A2"
