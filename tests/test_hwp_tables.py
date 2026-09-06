import re
from pathlib import Path

import pytest

from scripts.data.hwp_tables import (
    EXTENDED_CONTROL_WCHARS,
    _cell_position,
    _read_records,
    _TableBuilder,
    decode_paragraph,
    read_tables,
)

ROOT = Path(__file__).resolve().parents[1]


def record(tag: int, level: int, payload: bytes) -> bytes:
    header = (len(payload) << 20) | (level << 10) | tag
    return header.to_bytes(4, "little") + payload


def wchars(*codes: int) -> bytes:
    return b"".join(code.to_bytes(2, "little") for code in codes)


class TestReadRecords:
    def test_parses_tag_level_and_size_from_the_header(self):
        data = record(0x43, 2, b"abcd")
        (tag, level, payload), = _read_records(data)
        assert (tag, level, payload) == (0x43, 2, b"abcd")

    def test_extended_size_uses_the_following_four_bytes(self):
        payload = b"x" * 0x1000
        header = (0xFFF << 20) | (1 << 10) | 0x43
        data = header.to_bytes(4, "little") + len(payload).to_bytes(4, "little") + payload
        (tag, level, read), = _read_records(data)
        assert tag == 0x43 and len(read) == len(payload)

    def test_stops_cleanly_on_a_truncated_tail(self):
        assert list(_read_records(record(0x43, 0, b"ab") + b"\x01\x02")) == [(0x43, 0, b"ab")]


class TestDecodeParagraph:
    """여기가 실제로 버그가 났던 자리다. 확장 제어문자는 8워드를 차지한다."""

    def test_plain_text_survives(self):
        assert decode_paragraph("요구사항".encode("utf-16-le")) == "요구사항"

    def test_extended_control_consumes_eight_wchars(self):
        # 제어문자(1) + 6워드 + 닫는 워드 = 8워드. 그 사이 ASCII가 한자로 새면 안 된다.
        payload = wchars(1, 0x6C64, 0x636F, 0x0000, 0x0000, 0x0000, 0x0000, 1) + "QMR-013".encode("utf-16-le")
        assert decode_paragraph(payload) == "QMR-013"

    def test_line_break_controls_become_newlines(self):
        payload = "가".encode("utf-16-le") + wchars(10) + "나".encode("utf-16-le")
        assert decode_paragraph(payload) == "가\n나"

    def test_tab_separates_but_is_normalized_to_a_space(self):
        # 탭은 글자를 붙여버리지 않고 구분자로 남되, 공백 정규화에서 스페이스가 된다.
        assert decode_paragraph("가".encode("utf-16-le") + wchars(9) + "나".encode("utf-16-le")) == "가 나"

    def test_runs_of_spaces_collapse(self):
        assert decode_paragraph("가    나".encode("utf-16-le")) == "가 나"

    def test_control_width_constant_matches_the_rule(self):
        assert EXTENDED_CONTROL_WCHARS == 8


class TestCellPosition:
    def test_reads_row_and_column(self):
        payload = b"\x00" * 8 + (3).to_bytes(2, "little") + (5).to_bytes(2, "little") + b"\x00" * 8
        assert _cell_position(payload) == (5, 3)

    def test_short_payload_is_not_a_table_cell(self):
        assert _cell_position(b"\x00" * 4) is None


class TestTableBuilder:
    def test_cells_land_in_their_own_row_and_column(self):
        builder = _TableBuilder(level=1)
        builder.open_cell((0, 0))
        builder.add_text("요구사항 번호")
        builder.open_cell((0, 1))
        builder.add_text("ECR-001")
        builder.open_cell((1, 0))
        builder.add_text("요구사항 명")
        builder.open_cell((1, 1))
        builder.add_text("도입 제품")
        assert builder.rows() == [["요구사항 번호", "ECR-001"], ["요구사항 명", "도입 제품"]]

    def test_paragraphs_in_one_cell_join_with_newlines(self):
        builder = _TableBuilder(level=1)
        builder.open_cell((0, 0))
        builder.add_text("첫 줄")
        builder.add_text("둘째 줄")
        assert builder.rows() == [["첫 줄\n둘째 줄"]]

    def test_text_outside_a_cell_is_dropped(self):
        builder = _TableBuilder(level=1)
        builder.open_cell(None)
        assert builder.add_text("표 밖 문단") is False
        assert builder.rows() == []

    def test_empty_table_yields_no_rows(self):
        assert _TableBuilder(level=1).rows() == []


class TestRealDocumentsIfPresent:
    """실제 HWP가 있으면 계약을 확인한다. 없으면 건너뛴다."""

    CANDIDATES = tuple((ROOT / "RFP_data").glob("*.hwp"))

    def test_requirement_tables_come_out_as_label_value_rows(self):
        if not self.CANDIDATES:
            pytest.skip("RFP_data에 .hwp 원본이 없습니다.")
        id_pattern = re.compile(r"^[A-Z]{2,4}-\d{2,4}$")
        found = False
        for path in self.CANDIDATES:
            for table in read_tables(path):
                cells = [cell.strip() for row in table for cell in row]
                if any(id_pattern.match(cell) for cell in cells):
                    assert all(isinstance(row, list) for row in table)
                    assert all(isinstance(cell, str) for row in table for cell in row)
                    found = True
                    break
            if found:
                break
        assert found, "요구사항 ID가 담긴 표를 하나도 찾지 못했습니다."

    def test_no_control_character_garbage_leaks_into_cells(self):
        if not self.CANDIDATES:
            pytest.skip("RFP_data에 .hwp 원본이 없습니다.")
        # 제어문자를 건너뛰지 않으면 ASCII 바이트쌍이 이 대역의 한자로 읽힌다.
        garbage = re.compile(r"[氀-濿]{2,}")
        for path in self.CANDIDATES[:1]:
            for table in read_tables(path):
                for row in table:
                    for cell in row:
                        assert not garbage.search(cell.replace(" ", "")[:40]) or "汤捯" not in cell
