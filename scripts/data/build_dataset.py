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


ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = ROOT / "RFP_data" / "md"
OUTPUT_DIR = ROOT / "data" / "processed"
REVIEW_DIR = ROOT / "data" / "review"
REPORT_DIR = ROOT / "reports" / "current"
VERSION = "v0.2.0"
DATASET_VERSION = f"requirements_{VERSION}"

ID_RE = re.compile(r"(?<![A-Z0-9])([A-Z]{2,8}(?:-+[A-Z0-9]{2,8})+)(?![A-Z0-9])")

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

REQUIREMENT_ID_ALIASES = {
    ("incheon_airport_digital_work", "CUR-CM--001"): "CUR-CM-001",
}

ACCEPTED_INDEX_EXCEPTIONS = {
    "defense_intelligent_platform": {
        "duplicate_index_ids": {"SFR-048"},
    },
    "kac_ai_work_platform": {
        "index_only_ids": {"PER-003"},
    },
}

ACCEPTED_DOCUMENT_EXCEPTIONS = [
    {
        "document_id": "koen_ai_infrastructure",
        "exception_type": "summary_detail_count_mismatch",
        "summary_count": 99,
        "detail_count": 101,
        "category_differences": {
            "PER": {"summary_count": 5, "detail_count": 4},
            "QUR": {"summary_count": 9, "detail_count": 12},
        },
        "policy": "상세 요구사항 표 101건을 유지하고 개수표 차이를 원문 예외로 기록",
    }
]


def clean_text(value: str) -> str:
    value = html.unescape(value).replace("\xa0", " ")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in value.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def canonicalize_requirement_id(document_id: str, source_requirement_id: str) -> str:
    return REQUIREMENT_ID_ALIASES.get(
        (document_id, source_requirement_id), source_requirement_id
    )


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
        self.nested_cell_index = 0

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
        elif tag == "tr" and self.table is not None:
            if self.table_depth == 1:
                self.row = []
                self.row_line = self.getpos()[0]
            elif self.cell is not None:
                self.cell.text += "\n"
                self.nested_cell_index = 0
        elif tag in {"td", "th"}:
            if self.row is not None and self.table_depth == 1:
                self.cell = Cell(tag=tag)
            elif self.cell is not None and self.table_depth > 1:
                if self.nested_cell_index:
                    self.cell.text += " | "
                self.nested_cell_index += 1
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
        elif tag == "tr" and self.table_depth > 1 and self.cell is not None:
            self.cell.text += "\n"
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


def _section_heading_text(line: str) -> str:
    """Return visible text for a standalone Markdown/HTML heading line."""
    if re.search(r"</?(?:tr|td|th)\b", line, flags=re.IGNORECASE):
        return ""
    return clean_text(re.sub(r"<[^>]+>", " ", line)).strip("#* |\t")


def extract_index_requirement_ids(markdown: str) -> list[str]:
    """Extract requirement IDs from the document's requirement-index section."""
    lines = markdown.splitlines()
    candidates: list[list[str]] = []
    start_pattern = re.compile(
        r"요구사항\s*(?:(?:정의\s*및\s*)?목록(?:표)?|총괄표)"
    )
    end_pattern = re.compile(
        r"요구사항\s*(?:세부(?:\s*내용)?|상세(?:\s*(?:내용|내역))?)"
    )

    for start, line in enumerate(lines):
        heading = _section_heading_text(line)
        if not heading or not start_pattern.search(heading):
            continue

        end = len(lines)
        for index in range(start + 1, len(lines)):
            next_heading = _section_heading_text(lines[index])
            if next_heading and end_pattern.search(next_heading):
                end = index
                break

        section = "\n".join(lines[start + 1 : end])
        ids: list[str] = []
        for table in parse_tables(section):
            if find_requirement_id(table)[0]:
                break
            ids.extend(
                match.group(1)
                for row in table.rows
                for cell in row
                for match in ID_RE.finditer(cell.text)
            )
        if ids:
            candidates.append(ids)

    return candidates[-1] if candidates else []


def compare_index_and_detail_ids(
    index_ids: list[str], detail_ids: list[str]
) -> dict[str, int | bool | list[str]]:
    """Compare IDs declared in an index with IDs extracted from detail tables."""
    index_counts = Counter(index_ids)
    index_set = set(index_ids)
    detail_set = set(detail_ids)
    duplicate_index_ids = sorted(
        requirement_id
        for requirement_id, count in index_counts.items()
        if count > 1
    )
    index_only_ids = sorted(index_set - detail_set)
    detail_only_ids = sorted(detail_set - index_set) if index_ids else []
    return {
        "index_count": len(index_ids),
        "detail_count": len(detail_ids),
        "index_only_ids": index_only_ids,
        "detail_only_ids": detail_only_ids,
        "duplicate_index_ids": duplicate_index_ids,
        "is_exact_match": bool(index_ids)
        and not index_only_ids
        and not detail_only_ids
        and not duplicate_index_ids,
    }


def apply_index_exception_policy(document_id: str, comparison: dict) -> dict:
    """Separate approved source-document exceptions from unresolved mismatches."""
    result = dict(comparison)
    accepted = ACCEPTED_INDEX_EXCEPTIONS.get(document_id, {})
    for key in ("index_only_ids", "detail_only_ids", "duplicate_index_ids"):
        accepted_ids = sorted(set(comparison[key]) & accepted.get(key, set()))
        unresolved_ids = sorted(set(comparison[key]) - set(accepted_ids))
        result[f"accepted_{key}"] = accepted_ids
        result[f"unresolved_{key}"] = unresolved_ids

    result["is_policy_resolved"] = bool(comparison["index_count"]) and not any(
        result[f"unresolved_{key}"]
        for key in ("index_only_ids", "detail_only_ids", "duplicate_index_ids")
    )
    return result


REVIEW_FIELDS = [
    "document_id",
    "requirement_uid",
    "requirement_id",
    "reason",
    "source_file",
    "source_location",
    "original_checked",
    "id_match",
    "name_match",
    "body_complete",
    "table_structure_ok",
    "review_note",
    "reviewer",
    "reviewed_at",
]


def build_review_queue(
    records: list[dict], comparisons: dict[str, dict]
) -> list[dict[str, str]]:
    """Build a deterministic queue for mismatch and source-document review."""
    records_by_document: defaultdict[str, list[dict]] = defaultdict(list)
    records_by_key: dict[tuple[str, str], dict] = {}
    for record in records:
        document_id = record["document_id"]
        records_by_document[document_id].append(record)
        records_by_key[(document_id, record["requirement_id"])] = record

    queue: list[dict[str, str]] = []

    def add(record: dict, reason: str) -> None:
        queue.append(
            {
                "document_id": str(record.get("document_id", "")),
                "requirement_uid": str(record.get("requirement_uid", "")),
                "requirement_id": str(record.get("requirement_id", "")),
                "reason": reason,
                "source_file": str(record.get("source_file", "")),
                "source_location": str(record.get("source_location", "")),
                "original_checked": "",
                "id_match": "",
                "name_match": "",
                "body_complete": "",
                "table_structure_ok": "",
                "review_note": "",
                "reviewer": "",
                "reviewed_at": "",
            }
        )

    for document_id in sorted(comparisons):
        comparison = comparisons[document_id]
        document_records = records_by_document.get(document_id, [])
        source_file = document_records[0].get("source_file", "") if document_records else ""
        for reason, key in (
            ("index_only", "index_only_ids"),
            ("detail_only", "detail_only_ids"),
            ("duplicate_index", "duplicate_index_ids"),
        ):
            for requirement_id in comparison.get(key, []):
                record = records_by_key.get((document_id, requirement_id))
                if record is None:
                    record = {
                        "document_id": document_id,
                        "requirement_uid": f"{document_id}:{requirement_id}",
                        "requirement_id": requirement_id,
                        "source_file": source_file,
                        "source_location": "",
                    }
                add(record, reason)

    for document_id in sorted(records_by_document):
        document_records = records_by_document[document_id]
        middle_index = (len(document_records) - 1) // 2
        longest = max(
            document_records,
            key=lambda row: len(row.get("raw_requirement_text") or ""),
        )
        for record, reason in (
            (document_records[0], "document_first"),
            (document_records[middle_index], "document_middle"),
            (document_records[-1], "document_last"),
            (longest, "document_longest"),
        ):
            add(record, reason)

    return queue


def write_review_queue(queue: list[dict[str, str]], output_path: Path) -> None:
    """Write the human source-review queue with a stable schema."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing_reviews: dict[tuple[str, str, str], dict[str, str]] = {}
    if output_path.exists():
        with output_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                key = (row["document_id"], row["requirement_id"], row["reason"])
                existing_reviews[key] = row

    review_fields = REVIEW_FIELDS[6:]
    for row in queue:
        key = (row["document_id"], row["requirement_id"], row["reason"])
        existing = existing_reviews.get(key, {})
        for field in review_fields:
            if not row[field] and existing.get(field):
                row[field] = existing[field]

    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(queue)


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
            source_requirement_id, adapter = find_requirement_id(table)
            if not source_requirement_id:
                continue
            requirement_id = canonicalize_requirement_id(
                metadata["document_id"], source_requirement_id
            )

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
                    "dataset_version": DATASET_VERSION,
                    "document_id": metadata["document_id"],
                    "agency": metadata["agency"],
                    "domain": metadata["domain"],
                    "requirement_uid": f"{metadata['document_id']}:{requirement_id}",
                    "requirement_id": requirement_id,
                    "source_requirement_id": source_requirement_id,
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


def validate(
    records: list[dict],
    source_files: Iterable[Path],
    index_comparisons: dict[str, dict] | None = None,
) -> dict:
    index_comparisons = index_comparisons or {}
    policy_comparisons = {
        document_id: apply_index_exception_policy(document_id, comparison)
        for document_id, comparison in index_comparisons.items()
    }
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
    exact_match_documents = sorted(
        document_id
        for document_id, comparison in index_comparisons.items()
        if comparison["is_exact_match"]
    )
    mismatch_documents = sorted(
        document_id
        for document_id, comparison in index_comparisons.items()
        if comparison["index_count"] and not comparison["is_exact_match"]
    )
    documents_without_index_ids = sorted(
        document_id
        for document_id, comparison in index_comparisons.items()
        if not comparison["index_count"]
    )
    policy_resolved_documents = sorted(
        document_id
        for document_id, comparison in policy_comparisons.items()
        if not comparison["is_exact_match"] and comparison["is_policy_resolved"]
    )
    unresolved_mismatch_documents = sorted(
        document_id
        for document_id, comparison in policy_comparisons.items()
        if comparison["index_count"] and not comparison["is_policy_resolved"]
    )
    accepted_document_exceptions = [
        exception
        for exception in ACCEPTED_DOCUMENT_EXCEPTIONS
        if exception["document_id"] in by_document
    ]
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
        "index_exact_match_document_count": len(exact_match_documents),
        "index_mismatch_document_count": len(mismatch_documents),
        "index_policy_resolved_document_count": len(policy_resolved_documents),
        "unresolved_index_mismatch_document_count": len(
            unresolved_mismatch_documents
        ),
        "documents_without_index_ids": documents_without_index_ids,
        "accepted_document_exceptions": accepted_document_exceptions,
        "index_comparisons": index_comparisons,
        "index_policy_comparisons": policy_comparisons,
    }


def write_outputs(
    records: list[dict], audit: dict, review_queue: list[dict[str, str]]
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    jsonl_path = OUTPUT_DIR / f"{DATASET_VERSION}.jsonl"
    with jsonl_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    csv_path = OUTPUT_DIR / f"{DATASET_VERSION}.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    (REPORT_DIR / f"extraction_audit_{VERSION}.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        f"# 요구사항 추출 감사 {VERSION}",
        "",
        f"- 추출 행: {audit['record_count']:,}",
        f"- 포함 문서: {audit['document_count']}/10",
        f"- 요구사항명 누락: {audit['missing_name_count']:,}",
        f"- 본문 누락: {audit['missing_body_count']:,}",
        f"- 중복 UID: {audit['duplicate_uid_count']:,}",
        f"- 목록-상세표 완전 일치 문서: {audit['index_exact_match_document_count']}",
        f"- 목록-상세표 불일치 문서: {audit['index_mismatch_document_count']}",
        f"- 승인 정책으로 해소된 문서: {audit['index_policy_resolved_document_count']}",
        f"- 미해결 목록-상세 불일치 문서: {audit['unresolved_index_mismatch_document_count']}",
        f"- ID 목록이 없는 문서: {len(audit['documents_without_index_ids'])}",
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
            "## 목록-상세표 ID 대조",
            "",
            "| document_id | 목록 | 상세 | 목록만 | 상세만 | 목록 중복 | 상태 |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for document_id, comparison in audit["index_policy_comparisons"].items():
        if not comparison["index_count"]:
            status = "목록 ID 없음"
        elif comparison["is_exact_match"]:
            status = "일치"
        elif comparison["is_policy_resolved"]:
            status = "승인 예외"
        else:
            status = "검토 필요"
        lines.append(
            f"| {document_id} | {comparison['index_count']} | "
            f"{comparison['detail_count']} | {len(comparison['index_only_ids'])} | "
            f"{len(comparison['detail_only_ids'])} | "
            f"{len(comparison['duplicate_index_ids'])} | {status} |"
        )
    lines.extend(["", "## 승인된 원문 예외", ""])
    for exception in audit["accepted_document_exceptions"]:
        lines.append(
            f"- `{exception['document_id']}`: 개수표 {exception['summary_count']}건, "
            f"상세표 {exception['detail_count']}건. {exception['policy']}"
        )
    lines.extend(
        [
            "",
            "## 자동 검토 대상",
            "",
            f"- 빈 문서: {', '.join(audit['empty_documents']) or '없음'}",
            f"- 본문 누락 UID: {', '.join(audit['missing_body_uids']) or '없음'}",
            f"- 중복 UID: {', '.join(audit['duplicate_uids']) or '없음'}",
            "- ID 목록 없는 문서: "
            f"{', '.join(audit['documents_without_index_ids']) or '없음'}",
            "",
            "> 이 결과는 상세 요구사항 표의 구조만 읽은 비라벨 원문 데이터셋이다. "
            "목록-상세표 대조와 원본 PDF/HWP 대조가 끝나기 전에는 학습 데이터로 확정하지 않는다.",
            "",
        ]
    )
    (REPORT_DIR / f"extraction_audit_{VERSION}.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    write_review_queue(
        review_queue,
        REVIEW_DIR / f"extraction_review_{VERSION}.csv",
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
    index_comparisons: dict[str, dict] = {}
    for path in source_files:
        metadata = DOCUMENTS[path.name]
        document_records = extract_document(path, metadata)
        records.extend(document_records)
        index_ids = extract_index_requirement_ids(path.read_text(encoding="utf-8"))
        detail_ids = [row["requirement_id"] for row in document_records]
        index_comparisons[metadata["document_id"]] = compare_index_and_detail_ids(
            index_ids, detail_ids
        )

    records.sort(key=lambda row: (row["document_id"], row["source_table_index"]))
    audit = validate(records, source_files, index_comparisons)
    review_queue = build_review_queue(records, index_comparisons)
    write_outputs(records, audit, review_queue)
    print(json.dumps(audit, ensure_ascii=False, indent=2))

    if args.strict and (
        audit["duplicate_uid_count"]
        or audit["missing_name_count"]
        or audit["missing_body_count"]
        or audit["empty_documents"]
        or audit["unresolved_index_mismatch_document_count"]
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
