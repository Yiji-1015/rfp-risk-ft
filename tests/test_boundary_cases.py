import csv

import pytest

from scripts.evaluation.boundary_cases import collect_boundary_cases, write_xlsx

FIELDS = [
    "requirement_uid",
    "test_document",
    "gold",
    "word_char_logistic_pred",
    "word_char_logistic_p_통상수용",
    "word_char_logistic_p_견적반영",
    "word_char_logistic_p_계약·질의검토",
]


def _oof(tmp_path, records):
    path = tmp_path / "oof.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)
    return path


def _row(uid: str, label: str) -> dict:
    return {
        "requirement_uid": uid,
        "document_id": "doc1",
        "primary_action": label,
        "model_text": f"{uid} 본문",
        "requirement_name": f"{uid} 이름",
        "reasoning": f"{uid} 근거",
        "blockers": ["검수·성능기준"],
    }


def _saved(uid: str, gold: str, pred: str, p_quote: float, p_review: float) -> dict:
    return {
        "requirement_uid": uid,
        "test_document": "doc1",
        "gold": gold,
        "word_char_logistic_pred": pred,
        "word_char_logistic_p_통상수용": 0.1,
        "word_char_logistic_p_견적반영": p_quote,
        "word_char_logistic_p_계약·질의검토": p_review,
    }


def test_only_the_two_minority_labels_are_collected(tmp_path) -> None:
    rows = [
        _row("a", "견적반영"),
        _row("b", "계약·질의검토"),
        _row("c", "통상수용"),
        _row("d", "견적반영"),
    ]
    path = _oof(
        tmp_path,
        [
            _saved("a", "견적반영", "계약·질의검토", 0.30, 0.60),
            _saved("b", "계약·질의검토", "견적반영", 0.55, 0.35),
            # 통상수용이 섞인 오답과 정답은 제외된다.
            _saved("c", "통상수용", "견적반영", 0.60, 0.30),
            _saved("d", "견적반영", "견적반영", 0.70, 0.20),
        ],
    )

    cases = collect_boundary_cases(rows, path)

    assert [case["requirement_uid"] for case in cases] == ["b", "a"]  # 확률차 오름차순
    assert cases[0]["확률차"] == pytest.approx(0.20)
    assert cases[1]["확률차"] == pytest.approx(0.30)
    assert cases[0]["정답"] == "계약·질의검토"
    assert cases[0]["예측"] == "견적반영"
    assert cases[0]["reasoning"] == "b 근거"
    assert cases[0]["blockers"] == "검수·성능기준"


def test_xlsx_is_written_with_one_row_per_case(tmp_path) -> None:
    openpyxl = pytest.importorskip("openpyxl")
    rows = [_row("a", "견적반영")]
    path = _oof(tmp_path, [_saved("a", "견적반영", "계약·질의검토", 0.30, 0.60)])
    cases = collect_boundary_cases(rows, path)

    target = tmp_path / "boundary.xlsx"
    write_xlsx(cases, target)

    sheet = openpyxl.load_workbook(target).active
    assert sheet.title == "견적↔계약 혼동"
    assert sheet.max_row == 2
    assert sheet["A1"].value == "requirement_uid"
    assert sheet.freeze_panes == "A2"
