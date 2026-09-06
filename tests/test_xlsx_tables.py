import pytest

openpyxl = pytest.importorskip("openpyxl")

from scripts.data.xlsx_tables import (  # noqa: E402
    _detail_grids,
    _label_and_value,
    _summary_grid,
    read_tables,
)


def sheet_from(rows):
    book = openpyxl.Workbook()
    sheet = book.active
    for row in rows:
        sheet.append(list(row))
    return sheet


class TestLabelAndValue:
    """정의서 시트의 곁다리 열과 미러 열을 걷어내는 자리다."""

    def test_two_cells_pass_through(self):
        assert _label_and_value(["요구사항 명칭", "계정 및 권한"]) == ["요구사항 명칭", "계정 및 권한"]

    def test_mirror_column_is_collapsed(self):
        assert _label_and_value(["요구사항 분류", "기능 요구사항", "기능 요구사항"]) == [
            "요구사항 분류",
            "기능 요구사항",
        ]

    def test_side_columns_lose_to_the_longest_value(self):
        # `응락수준`·`필수`가 섞이면 유형 표기가 정규화되지 않는다.
        row = ["요구사항 분류", "기능 요구사항", "응락수준", "필수", "기능 요구사항"]
        assert _label_and_value(row) == ["요구사항 분류", "기능 요구사항"]

    def test_identifier_beats_a_shorter_side_column(self):
        assert _label_and_value(["요구사항 고유번호", "ECR-001", "모바일", "ECR-001"]) == [
            "요구사항 고유번호",
            "ECR-001",
        ]

    def test_single_cell_row_is_untouched(self):
        assert _label_and_value(["비고"]) == ["비고"]


class TestSummaryGrid:
    def test_blank_rows_and_columns_are_dropped(self):
        sheet = sheet_from([
            [None, None, None],
            [None, "요구사항 분류", "요구사항 고유번호"],
            [None, "기능 요구사항", "SFR-001"],
        ])
        assert _summary_grid(sheet) == [
            ["요구사항 분류", "요구사항 고유번호"],
            ["기능 요구사항", "SFR-001"],
        ]


class TestDetailGrids:
    ROWS = [
        [None, "요구사항 분류", None, "기능 요구사항", "응락수준", "필수"],
        [None, "요구사항 고유번호", None, "SFR-001", None, None],
        [None, "요구사항 명칭", None, "계정 및 권한", None, None],
        [None, "세부내용", None, "○ 로그인 이력을 남긴다", None, None],
        [None, "요구사항 분류", None, "데이터 요구사항", None, None],
        [None, "요구사항 고유번호", None, "DAR-001", None, None],
        [None, "세부내용", None, "○ 표준 코드를 관리한다", None, None],
    ]

    def test_blocks_split_on_the_classification_row(self):
        grids = _detail_grids(sheet_from(self.ROWS))
        assert len(grids) == 2
        # 분류가 각 블록의 첫 행이어야 유형이 밀리지 않는다.
        assert grids[0][0] == ["요구사항 분류", "기능 요구사항"]
        assert grids[1][0] == ["요구사항 분류", "데이터 요구사항"]

    def test_each_block_keeps_its_own_identifier(self):
        grids = _detail_grids(sheet_from(self.ROWS))
        assert grids[0][1] == ["요구사항 고유번호", "SFR-001"]
        assert grids[1][1] == ["요구사항 고유번호", "DAR-001"]

    def test_groups_without_an_identifier_are_dropped(self):
        rows = [
            [None, "요구사항 분류", None, "안내", None],
            [None, "비고", None, "요구사항 아님", None],
        ]
        assert _detail_grids(sheet_from(rows)) == []


class TestReadTables:
    def test_summary_comes_first_then_one_grid_per_requirement(self, tmp_path):
        book = openpyxl.Workbook()
        summary = book.active
        summary.title = "1. 요구사항 총괄표"
        summary.append(["요구사항 분류", "요구사항 고유번호", "요구사항 명칭"])
        summary.append(["기능 요구사항", "SFR-001", "계정 및 권한"])
        detail = book.create_sheet("2. 요구사항 정의서")
        for row in TestDetailGrids.ROWS:
            detail.append(list(row))
        path = tmp_path / "과업내용서.xlsx"
        book.save(path)

        grids = read_tables(path)
        assert len(grids) == 3  # 총괄표 1 + 요구사항 2
        assert grids[0][0][0] == "요구사항 분류"
        assert grids[1][1] == ["요구사항 고유번호", "SFR-001"]

    def test_missing_sheets_yield_nothing_rather_than_raising(self, tmp_path):
        book = openpyxl.Workbook()
        book.active.title = "표지"
        path = tmp_path / "빈문서.xlsx"
        book.save(path)
        assert read_tables(path) == []
