#!/usr/bin/env python3
"""Build the requirement-level RFP dataset from converted Markdown files.

The source documents contain HTML tables embedded in Markdown.  This script
uses only the Python standard library so extraction is reproducible without a
package installation step.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "RFP_data" / "md"
OUTPUT_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"

ID_RE = re.compile(r"(?<![A-Z0-9])([A-Z]{2,8}(?:-[A-Z0-9]{2,8})+)(?![A-Z0-9])")

DOCUMENTS = {
    "1-3 (제안요청서) 신용회복위원회 통합 AI플랫폼 구축.md": {
        "document_id": "ccrs_ai_platform",
        "agency": "신용회복위원회",
        "domain": "금융·채무조정",
    },
    "강원랜드_생성형 AI 및 응용서비스 구축.md": {
        "document_id": "kangwon_land_genai",
        "agency": "강원랜드",
        "domain": "공기업·리조트",
    },
    "국방 지능형 플랫폼 고도화 구축.md": {
        "document_id": "defense_intelligent_platform",
        "agency": "국방부",
        "domain": "국방",
    },
    "남동발전_남동아이 인프라 증설 및 AI 연계표준 개발 용역.md": {
        "document_id": "koen_ai_infrastructure",
        "agency": "한국남동발전",
        "domain": "에너지·발전",
    },
    "붙임2. 제안요청서(AI 플랫폼(KEXIM AI) 구축)_F.md": {
        "document_id": "kexim_ai_platform",
        "agency": "한국수출입은행",
        "domain": "금융",
    },
    "생성형 AI 기반 침해대응체계 도입 및 구축 용역 제안요청서.md": {
        "document_id": "genai_incident_response",
        "agency": None,
        "domain": "사이버보안",
    },
    "식약처_의약품 AI 심사 및 산업지원 체계 구축.md": {
        "document_id": "mfds_drug_ai_review",
        "agency": "식품의약품안전처",
        "domain": "의약품·규제",
    },
    "인천공항_AI 디지털워크 전환 사업.md": {
        "document_id": "incheon_airport_digital_work",
        "agency": "인천국제공항공사",
        "domain": "공항·항공",
    },
    "한국공항공사_KAC AI 업무혁신 플랫폼 구축 용역.md": {
        "document_id": "kac_ai_work_platform",
        "agency": "한국공항공사",
        "domain": "공항·항공",
    },
    "한국철도공사_생성형 인공지능 시스템 구축 ISP·ISMP.md": {
        "document_id": "korail_genai_isp_ismp",
        "agency": "한국철도공사",
        "domain": "철도·교통",
    },
}


def clean_text(value: str) -> str:
    value = html.unescape(value).replace("\xa0", " ")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in value.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def key_text(value: str) -> str:
    value = clean_text(value)
    value = re.sub(r"^[①②③④⑤⑥⑦⑧⑨⑩\d.()\s]+", "", value)
    return re.sub(r"\s+", "", value)


@dataclass
class Cell:
    text: str = ""
    tag: str = "td"


@dataclass
class Table:
    line: int
    rows: list[list[Cell]] = field(default_factory=list)
    row_lines: list[int] = field(default_factory=list)


class TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[Table] = []
        self.table: Table | None = None
        self.table_depth = 0
        self.row: list[Cell] | None = None
        self.row_line = 0
        self.cell: Cell | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "table":
            if self.table is None:
                self.table = Table(line=self.getpos()[0])
                self.table_depth = 1
            else:
                # Some requirement bodies contain example tables.  They are
                # body content, not separate requirement tables.
                self.table_depth += 1
                if self.cell is not None:
                    self.cell.text += "\n"
        elif tag == "tr" and self.table is not None and self.table_depth == 1:
            self.row = []
            self.row_line = self.getpos()[0]
        elif tag in {"td", "th"} and self.row is not None and self.table_depth == 1:
            self.cell = Cell(tag=tag)
        elif tag == "br" and self.cell is not None:
            self.cell.text += "\n"

    def handle_endtag(self, tag: str) -> None:
        if (
            tag in {"td", "th"}
            and self.table_depth == 1
            and self.cell is not None
            and self.row is not None
        ):
            self.cell.text = clean_text(self.cell.text)
            self.row.append(self.cell)
            self.cell = None
        elif (
            tag == "tr"
            and self.table_depth == 1
            and self.row is not None
            and self.table is not None
        ):
            if self.row:
                self.table.rows.append(self.row)
                self.table.row_lines.append(self.row_line)
            self.row = None
        elif tag == "table" and self.table is not None:
            if self.table_depth > 1:
                self.table_depth -= 1
                if self.cell is not None:
                    self.cell.text += "\n"
            else:
                self.tables.append(self.table)
                self.table = None
                self.table_depth = 0

    def handle_data(self, data: str) -> None:
        if self.cell is not None:
            self.cell.text += data


def parse_tables(markdown: str) -> list[Table]:
    parser = TableParser()
    parser.feed(markdown)
    return parser.tables


def row_value(row: list[Cell]) -> str:
    return clean_text("\n".join(cell.text for cell in row[1:]))


def find_labeled_value(rows: list[list[Cell]], labels: set[str]) -> str | None:
    for row in rows:
        if not row:
            continue
        key = key_text(row[0].text)
        if key in labels:
            value = row_value(row)
            if value:
                return value
    return None


def find_requirement_id(table: Table) -> tuple[str | None, str | None]:
    """Return (requirement_id, adapter_name)."""
    for row in table.rows:
        if not row:
            continue
        key = key_text(row[0].text)
        if "요구사항고유번호" in key or key in {"요구사항ID", "고유번호"}:
            for cell in row[1:]:
                match = ID_RE.search(cell.text)
                if match:
                    return match.group(1), "labeled_id"

    # KAC document: the first table row contains only the ID, followed by
    # labeled classification/name/content rows.
    if table.rows and any(
        key_text(row[0].text) in {"요구사항명", "요구사항명칭"}
        for row in table.rows[1:]
        if row
    ):
        first_row_text = " ".join(cell.text for cell in table.rows[0])
        match = ID_RE.fullmatch(clean_text(first_row_text))
        if match:
            return match.group(1), "header_id"
    return None, None


def split_requirement_blocks(table: Table) -> list[Table]:
    """Split documents that place many requirement records in one HTML table."""
    marker_indexes: list[int] = []
    for index, row in enumerate(table.rows):
        if not row:
            continue
        key = key_text(row[0].text)
        if "요구사항고유번호" in key or key in {"요구사항ID", "고유번호"}:
            if any(ID_RE.search(cell.text) for cell in row[1:]):
                marker_indexes.append(index)

    if len(marker_indexes) <= 1:
        return [table]

    blocks: list[Table] = []
    for block_index, start in enumerate(marker_indexes):
        end = marker_indexes[block_index + 1] if block_index + 1 < len(marker_indexes) else len(table.rows)
        rows = table.rows[start:end]
        row_lines = table.row_lines[start:end]
        if start > 0 and table.rows[start - 1]:
            previous_key = key_text(table.rows[start - 1][0].text)
            if previous_key in {"요구사항분류", "요구사항구분", "구분"}:
                rows = [table.rows[start - 1], *rows]
                row_lines = [table.row_lines[start - 1], *row_lines]
        blocks.append(Table(line=table.row_lines[start], rows=rows, row_lines=row_lines))
    return blocks


def extract_document(path: Path, metadata: dict[str, str | None]) -> list[dict]:
    markdown = path.read_text(encoding="utf-8")
    source_sha256 = hashlib.sha256(markdown.encode("utf-8")).hexdigest()
    records: list[dict] = []

    for table_index, source_table in enumerate(parse_tables(markdown), start=1):
        for block_index, table in enumerate(split_requirement_blocks(source_table), start=1):
            requirement_id, adapter = find_requirement_id(table)
            if not requirement_id:
                continue

            name = find_labeled_value(
                table.rows, {"요구사항명", "요구사항명칭", "요구사항이름"}
            )
            requirement_type = find_labeled_value(
                table.rows, {"요구사항분류", "요구사항구분", "구분"}
            )
            body = find_labeled_value(
                table.rows,
                {
                    "요구사항내용",
                    "요구사항세부내용",
                    "세부내용",
                    "상세내용",
                    "요구내용",
                },
            )

            records.append(
                {
                    "dataset_version": "requirements_v0.1.0",
                    "document_id": metadata["document_id"],
                    "agency": metadata["agency"],
                    "domain": metadata["domain"],
                    "requirement_uid": f"{metadata['document_id']}:{requirement_id}",
                    "requirement_id": requirement_id,
                    "requirement_type": requirement_type,
                    "requirement_name": name,
                    "raw_requirement_text": body,
                    "normalized_requirement_text": body,
                    "source_file": path.name,
                    "source_location": f"markdown_line:{table.line}",
                    "source_table_index": table_index,
                    "source_block_index": block_index,
                    "source_sha256": source_sha256,
                    "extraction_adapter": adapter,
                }
            )
    return records


def validate(records: list[dict], source_files: Iterable[Path]) -> dict:
    uid_counts = Counter(row["requirement_uid"] for row in records)
    by_document = Counter(row["document_id"] for row in records)
    adapters = Counter(row["extraction_adapter"] for row in records)
    missing_name = [row["requirement_uid"] for row in records if not row["requirement_name"]]
    missing_body = [row["requirement_uid"] for row in records if not row["raw_requirement_text"]]
    duplicate_uids = sorted(uid for uid, count in uid_counts.items() if count > 1)
    empty_documents = sorted(
        DOCUMENTS[path.name]["document_id"]
        for path in source_files
        if by_document[DOCUMENTS[path.name]["document_id"]] == 0
    )
    return {
        "record_count": len(records),
        "document_count": len(by_document),
        "records_by_document": dict(sorted(by_document.items())),
        "records_by_adapter": dict(sorted(adapters.items())),
        "missing_name_count": len(missing_name),
        "missing_name_uids": missing_name,
        "missing_body_count": len(missing_body),
        "missing_body_uids": missing_body,
        "duplicate_uid_count": len(duplicate_uids),
        "duplicate_uids": duplicate_uids,
        "empty_documents": empty_documents,
    }


def write_outputs(records: list[dict], audit: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    jsonl_path = OUTPUT_DIR / "requirements_v0.1.0.jsonl"
    with jsonl_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    csv_path = OUTPUT_DIR / "requirements_v0.1.0.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    (REPORT_DIR / "extraction_audit_v0.1.0.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# 요구사항 추출 감사 v0.1.0",
        "",
        f"- 추출 행: {audit['record_count']:,}",
        f"- 포함 문서: {audit['document_count']}/10",
        f"- 요구사항명 누락: {audit['missing_name_count']:,}",
        f"- 본문 누락: {audit['missing_body_count']:,}",
        f"- 중복 UID: {audit['duplicate_uid_count']:,}",
        "",
        "## 문서별 행 수",
        "",
        "| document_id | rows |",
        "|---|---:|",
    ]
    lines.extend(
        f"| {document_id} | {count:,} |"
        for document_id, count in audit["records_by_document"].items()
    )
    lines.extend(
        [
            "",
            "## 자동 검토 대상",
            "",
            f"- 빈 문서: {', '.join(audit['empty_documents']) or '없음'}",
            f"- 본문 누락 UID: {', '.join(audit['missing_body_uids']) or '없음'}",
            f"- 중복 UID: {', '.join(audit['duplicate_uids']) or '없음'}",
            "",
            "> 이 결과는 상세 요구사항 표의 구조만 읽은 비라벨 원문 데이터셋이다. "
            "목록-상세표 대조와 원본 PDF/HWP 대조가 끝나기 전에는 학습 데이터로 확정하지 않는다.",
            "",
        ]
    )
    (REPORT_DIR / "extraction_audit_v0.1.0.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero when IDs are duplicated or required fields are missing",
    )
    args = parser.parse_args()

    source_files = sorted(SOURCE_DIR.glob("*.md"), key=lambda path: path.name)
    unknown = sorted(path.name for path in source_files if path.name not in DOCUMENTS)
    missing = sorted(name for name in DOCUMENTS if not (SOURCE_DIR / name).exists())
    if unknown or missing:
        raise SystemExit(f"source mapping mismatch: unknown={unknown}, missing={missing}")

    records: list[dict] = []
    for path in source_files:
        records.extend(extract_document(path, DOCUMENTS[path.name]))

    records.sort(key=lambda row: (row["document_id"], row["source_table_index"]))
    audit = validate(records, source_files)
    write_outputs(records, audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))

    if args.strict and (
        audit["duplicate_uid_count"]
        or audit["missing_name_count"]
        or audit["missing_body_count"]
        or audit["empty_documents"]
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
