"""HWP 5.0(`.hwp`) 본문에서 표를 **셀 경계를 살려서** 읽는다.

PIPELINE.md는 원본을 분석용 Markdown으로 바꾸는 단계를 "자동화돼 있지 않다"고 적어
두었고, decisions-01 §11.2는 그 수동 변환에서 **중첩 표의 셀 경계가 붙는** 위험을
경고했다. 이 모듈은 그 단계를 건너뛴다 — HWP 바이너리에서 직접 표를 읽으므로 변환
손실이 생길 자리가 없다.

내보내는 형태는 `list[list[str]]`(행 → 셀 본문)이다. HTML 표 파서가 만드는 것과 같은
모양이라 `build_dataset`의 뒷단(`split_requirement_blocks`·`find_requirement_id`·
`find_labeled_value`)을 그대로 재사용한다.

## 형식 메모

`.hwp`는 OLE2 복합 문서다. 본문은 `BodyText/Section*` 스트림에 들어 있고
`FileHeader`의 플래그 비트 0이 서면 zlib(raw deflate)로 압축돼 있다.

스트림은 레코드의 연속이며 헤더 4바이트가 `tag(0~9) | level(10~19) | size(20~31)`이다.
`size`가 `0xFFF`면 뒤따르는 4바이트가 실제 길이다. 표는 `CTRL_HEADER`의 ctrl_id가
`tbl `일 때 시작하고, 셀 하나가 `LIST_HEADER` 하나에 대응한다. 셀의 행·열 좌표는
그 payload에 들어 있어 표 구조를 그대로 복원할 수 있다.

**제어문자는 대부분 8워드를 차지한다.** 이를 건너뛰지 않으면 컨트롤 id의 ASCII
바이트쌍이 한자로 잘못 읽혀 `汤捯QMR-013` 같은 문자열이 본문에 섞인다. 실제로 이
처리를 넣기 전과 후에 한 문서의 요구사항 인식 건수가 37건에서 119건으로 달라졌다.
"""

from __future__ import annotations

import re
import zlib
from pathlib import Path

import olefile

PARA_TEXT = 0x43
CTRL_HEADER = 0x47
LIST_HEADER = 0x48

# 자기 자신 한 워드만 차지하는 제어문자. 나머지 0~31은 8워드를 차지한다.
SINGLE_WCHAR_CONTROLS = frozenset({0, 10, 13, 24, 25, 26, 27, 28, 29, 30, 31})
EXTENDED_CONTROL_WCHARS = 8

# 셀 좌표는 `LIST_HEADER` payload의 이 위치에 있다(공통부 8바이트 뒤).
CELL_COL_OFFSET = 8
CELL_MIN_PAYLOAD = 16


def _read_records(data: bytes):
    """(tag, level, payload) 를 순서대로 내놓는다."""
    position = 0
    while position + 4 <= len(data):
        header = int.from_bytes(data[position : position + 4], "little")
        tag = header & 0x3FF
        level = (header >> 10) & 0x3FF
        size = (header >> 20) & 0xFFF
        position += 4
        if size == 0xFFF:
            size = int.from_bytes(data[position : position + 4], "little")
            position += 4
        yield tag, level, data[position : position + size]
        position += size


def decode_paragraph(payload: bytes) -> str:
    """문단 텍스트에서 제어문자를 규칙대로 걷어낸다."""
    text = payload.decode("utf-16-le", errors="ignore")
    pieces: list[str] = []
    index = 0
    while index < len(text):
        code = ord(text[index])
        if code > 31 or code == 9:
            pieces.append(text[index])
            index += 1
        elif code in SINGLE_WCHAR_CONTROLS:
            pieces.append("\n" if code in (10, 13) else " ")
            index += 1
        else:
            pieces.append(" ")
            index += EXTENDED_CONTROL_WCHARS
    return re.sub(r"[ \t]+", " ", "".join(pieces)).strip()


def _cell_position(payload: bytes) -> tuple[int, int] | None:
    """`LIST_HEADER` payload에서 (행, 열)을 읽는다. 표 밖 목록이면 None."""
    if len(payload) < CELL_MIN_PAYLOAD:
        return None
    column = int.from_bytes(payload[CELL_COL_OFFSET : CELL_COL_OFFSET + 2], "little")
    row = int.from_bytes(payload[CELL_COL_OFFSET + 2 : CELL_COL_OFFSET + 4], "little")
    return row, column


class _TableBuilder:
    """열린 표 하나. 셀을 좌표에 담았다가 행 목록으로 편다."""

    def __init__(self, level: int) -> None:
        self.level = level
        self.cells: dict[tuple[int, int], list[str]] = {}
        self.current: list[str] | None = None

    def open_cell(self, position: tuple[int, int] | None) -> None:
        if position is None:
            self.current = None
            return
        self.current = self.cells.setdefault(position, [])

    def add_text(self, text: str) -> bool:
        if self.current is None:
            return False
        self.current.append(text)
        return True

    def rows(self) -> list[list[str]]:
        if not self.cells:
            return []
        last_row = max(row for row, _ in self.cells)
        table: list[list[str]] = []
        for row in range(last_row + 1):
            columns = sorted(col for r, col in self.cells if r == row)
            table.append(
                ["\n".join(p for p in self.cells[(row, col)] if p).strip() for col in columns]
            )
        return [row for row in table if row]


def read_tables(path: str | Path) -> list[list[list[str]]]:
    """HWP 파일의 표를 `표 → 행 → 셀 본문`으로 읽는다.

    표 밖 문단은 버린다. 요구사항은 모두 표 안에 있고, 표 밖 텍스트는 목차·안내문이라
    추출 대상이 아니다.
    """
    ole = olefile.OleFileIO(str(path))
    try:
        compressed = bool(ole.openstream("FileHeader").read()[36] & 1)
        streams = sorted(
            ("/".join(entry) for entry in ole.listdir() if entry[0] == "BodyText"),
            key=lambda name: int(re.sub(r"\D", "", name.rsplit("/", 1)[-1]) or 0),
        )
        tables: list[list[list[str]]] = []
        for name in streams:
            raw = ole.openstream(name).read()
            data = zlib.decompress(raw, -15) if compressed else raw
            tables.extend(_read_section(data))
        return tables
    finally:
        ole.close()


def _read_section(data: bytes) -> list[list[list[str]]]:
    builders: list[_TableBuilder] = []
    finished: list[tuple[int, list[list[str]]]] = []
    order = 0
    for tag, level, payload in _read_records(data):
        # level이 얕아지면 그 level 이상에서 열려 있던 표가 모두 닫힌 것이다.
        while builders and level <= builders[-1].level:
            closed = builders.pop()
            rows = closed.rows()
            if rows:
                finished.append((closed.order, rows))
        if tag == CTRL_HEADER and payload[:4][::-1] == b"tbl ":
            builder = _TableBuilder(level)
            builder.order = order  # type: ignore[attr-defined]
            order += 1
            builders.append(builder)
        elif tag == LIST_HEADER and builders:
            builders[-1].open_cell(_cell_position(payload))
        elif tag == PARA_TEXT:
            text = decode_paragraph(payload)
            if text and builders:
                builders[-1].add_text(text)
    while builders:
        closed = builders.pop()
        rows = closed.rows()
        if rows:
            finished.append((closed.order, rows))
    return [rows for _, rows in sorted(finished, key=lambda item: item[0])]
