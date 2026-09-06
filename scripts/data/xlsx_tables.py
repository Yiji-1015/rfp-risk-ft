"""요구사항이 엑셀로 온 제안요청서를 표로 읽는다.

조달 문서 중에는 과업내용서를 `.xlsx`로 내는 것이 있다. 표가 이미 셀 격자라 변환
손실이 생길 자리가 없어 HWP·PDF보다 오히려 안전하다.

내보내는 형태는 `hwp_tables`와 같은 `표 → 행 → 셀 본문`이다. `build_dataset`의 뒷단이
그대로 재사용된다.

## 시트 구조

관찰된 양식은 두 시트로 나뉜다.

- **요구사항 총괄표** — 한 행에 `분류 | 고유번호 | 명칭`. 목록과 상세의 대조에 쓴다.
- **요구사항 정의서** — 요구사항 하나가 여러 행에 걸친 블록이다. 라벨 열과 값 열이
  따로 있고, 곁다리 열(`응락수준` 등)과 값을 복사한 미러 열이 함께 붙는다.

블록 경계는 **`요구사항 분류`가 다시 나오는 지점**이다. 고유번호로 자르면 그 앞 행인
분류가 이전 블록에 붙어 유형이 통째로 한 칸씩 밀린다. 상세표 한 장을 요구사항 하나로
보는 HWP 쪽과 달리, 여기서는 **행 묶음**이 그 단위다.
"""

from __future__ import annotations

import re
from pathlib import Path

import openpyxl

# 시트 이름은 문서마다 번호가 붙는다("1. 요구사항 총괄표"). 핵심 낱말로 찾는다.
SUMMARY_HINT = "총괄"
DETAIL_HINT = "정의서"

ID_PATTERN = re.compile(r"^[A-Z]{2,4}-\d{2,4}$")
ID_LABELS = frozenset({"요구사항고유번호", "요구사항번호", "요구사항id"})
# 블록은 `요구사항 분류`로 시작한다. 고유번호는 그 다음 행이라 그것으로 자르면
# 분류가 앞 블록에 붙어 유형이 통째로 밀린다.
BLOCK_START_LABELS = frozenset({"요구사항분류", "요구사항구분"})


def _norm(value: object) -> str:
    return re.sub(r"\s+", "", str(value)).lower() if value is not None else ""


def _cells(row: tuple) -> list[str]:
    return [str(cell).strip() if cell is not None else "" for cell in row]


def _label_and_value(cells: list[str]) -> list[str]:
    """`[라벨, 값]` 두 칸으로 줄인다.

    정의서 시트는 라벨·값 말고도 곁다리 열을 둔다 — 관찰된 양식에는 `응락수준`·`필수`
    같은 칸과, 값을 그대로 복사한 미러 열이 함께 있다. 그대로 두면 `row_value`가
    뒤 칸을 전부 이어 붙여 `'기능 요구사항\\n응락수준\\n필수\\n기능 요구사항'` 같은
    값이 되어 유형 정규화가 표기를 못 알아본다.

    중복을 걷어낸 뒤에도 후보가 여럿이면 **가장 긴 칸**을 값으로 본다. 라벨 행의 실제
    값(분류명·ID·본문)은 곁다리 칸보다 길다.
    """
    if len(cells) <= 2:
        return cells
    label, rest = cells[0], cells[1:]
    unique: list[str] = []
    for cell in rest:
        if cell not in unique:
            unique.append(cell)
    if len(unique) == 1:
        return [label, unique[0]]
    return [label, max(unique, key=len)]


def _find_sheet(book: openpyxl.Workbook, hint: str):
    for name in book.sheetnames:
        if hint in name.replace(" ", ""):
            return book[name]
    return None


def _summary_grid(sheet) -> list[list[str]]:
    """총괄표를 그대로 한 장의 표로 옮긴다. 빈 열은 버린다."""
    grid = []
    for row in sheet.iter_rows(values_only=True):
        cells = [cell for cell in _cells(row) if cell]
        if cells:
            grid.append(cells)
    return grid


def _detail_grids(sheet) -> list[list[list[str]]]:
    """정의서를 요구사항 하나당 표 한 장으로 자른다.

    라벨이 두 칸으로 갈라지는 행(`상세설명 | 정의 | 값`)은 셋을 그대로 한 행에 둔다.
    `find_labeled_value`가 첫 칸을 라벨로 보고 나머지에서 값을 찾으므로 이대로 맞는다.
    """
    grids: list[list[list[str]]] = []
    current: list[list[str]] | None = None
    for row in sheet.iter_rows(values_only=True):
        cells = [cell for cell in _cells(row) if cell]
        if not cells:
            continue
        if _norm(cells[0]) in BLOCK_START_LABELS:
            current = []
            grids.append(current)
        if current is not None:
            current.append(_label_and_value(cells))
    # 고유번호가 없는 묶음은 요구사항이 아니다(안내문·머리글).
    return [
        grid
        for grid in grids
        if any(
            _norm(row[0]) in ID_LABELS and any(ID_PATTERN.match(c) for c in row[1:])
            for row in grid
        )
    ]


def read_tables(path: str | Path) -> list[list[list[str]]]:
    """엑셀 과업내용서에서 표를 읽는다. 총괄표가 먼저, 요구사항 블록이 뒤따른다."""
    book = openpyxl.load_workbook(path, data_only=True, read_only=True)
    try:
        grids: list[list[list[str]]] = []
        summary = _find_sheet(book, SUMMARY_HINT)
        if summary is not None:
            grid = _summary_grid(summary)
            if grid:
                grids.append(grid)
        detail = _find_sheet(book, DETAIL_HINT)
        if detail is not None:
            grids.extend(_detail_grids(detail))
        return grids
    finally:
        book.close()
