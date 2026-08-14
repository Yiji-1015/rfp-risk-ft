import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.data.build_dataset import (
    apply_index_exception_policy,
    build_review_queue,
    compare_index_and_detail_ids,
    extract_document,
    extract_index_requirement_ids,
    find_requirement_id,
    parse_tables,
    validate,
    write_review_queue,
)


class RequirementIndexAuditTests(unittest.TestCase):
    def test_extracts_ids_only_from_requirement_index_section(self):
        markdown = """
## 2. 요구사항 목록

<table>
<tr><th>번호</th><th>요구사항 ID</th><th>명칭</th></tr>
<tr><td>1</td><td>SFR-001</td><td>검색</td></tr>
<tr><td>2</td><td>SFR-002</td><td>응답</td></tr>
</table>

## 3. 요구사항 세부내용

<table>
<tr><td>요구사항 고유번호</td><td>SFR-001</td></tr>
<tr><td>요구사항 명칭</td><td>검색</td></tr>
<tr><td>요구사항 내용</td><td>SFR-099 시스템과 연계한다.</td></tr>
</table>
"""

        self.assertEqual(
            extract_index_requirement_ids(markdown),
            ["SFR-001", "SFR-002"],
        )

    def test_uses_last_index_heading_and_stops_at_first_detail_table(self):
        markdown = """
**2. 요구사항 목록**

<table><tr><td>요구사항 분류 예시</td><td>SFR-000</td></tr></table>

2. 요구사항 목록

<table>
<tr><th>번호</th><th>요구사항 ID</th><th>명칭</th></tr>
<tr><td>1</td><td>SFR-001</td><td>검색</td></tr>
</table>
<table>
<tr><td>2</td><td>SFR-002</td><td>응답</td></tr>
</table>
<table>
<tr><td>요구사항 고유번호</td><td>SFR-001</td></tr>
<tr><td>요구사항 명칭</td><td>검색</td></tr>
<tr><td>요구사항 내용</td><td>SFR-099 시스템과 연계한다.</td></tr>
</table>
"""

        self.assertEqual(
            extract_index_requirement_ids(markdown),
            ["SFR-001", "SFR-002"],
        )

    def test_preserves_nested_table_row_and_cell_boundaries(self):
        markdown = """
<table>
<tr><td>요구사항 내용</td><td>장비 목록
<table>
<tr><th>번호</th><th>구분</th></tr>
<tr><td>1</td><td>HW</td></tr>
<tr><td>2</td><td>SW</td></tr>
</table>
완료</td></tr>
</table>
"""

        body = parse_tables(markdown)[0].rows[0][1].text

        self.assertEqual(
            body,
            "장비 목록\n번호 | 구분\n1 | HW\n2 | SW\n완료",
        )

    def test_compares_both_missing_directions_and_duplicate_index_ids(self):
        result = compare_index_and_detail_ids(
            ["SFR-001", "SFR-002", "SFR-002"],
            ["SFR-001", "SFR-003"],
        )

        self.assertEqual(result["index_count"], 3)
        self.assertEqual(result["detail_count"], 2)
        self.assertEqual(result["index_only_ids"], ["SFR-002"])
        self.assertEqual(result["detail_only_ids"], ["SFR-003"])
        self.assertEqual(result["duplicate_index_ids"], ["SFR-002"])
        self.assertFalse(result["is_exact_match"])

    def test_treats_missing_index_as_unavailable_instead_of_all_details_missing(self):
        result = compare_index_and_detail_ids([], ["SFR-001", "SFR-002"])

        self.assertEqual(result["index_count"], 0)
        self.assertEqual(result["detail_count"], 2)
        self.assertEqual(result["index_only_ids"], [])
        self.assertEqual(result["detail_only_ids"], [])
        self.assertFalse(result["is_exact_match"])

    def test_preserves_double_hyphen_in_source_requirement_id(self):
        markdown = """
<table>
<tr><td>요구사항 고유번호</td><td>CUR-CM--001</td></tr>
<tr><td>요구사항 명칭</td><td>컨설팅</td></tr>
<tr><td>요구사항 내용</td><td>업무 체계를 수립한다.</td></tr>
</table>
"""

        requirement_id, _ = find_requirement_id(parse_tables(markdown)[0])

        self.assertEqual(requirement_id, "CUR-CM--001")

    def test_extract_document_separates_canonical_and_source_requirement_ids(self):
        markdown = """
<table>
<tr><td>요구사항 고유번호</td><td>CUR-CM--001</td></tr>
<tr><td>요구사항 명칭</td><td>컨설팅</td></tr>
<tr><td>요구사항 내용</td><td>업무 체계를 수립한다.</td></tr>
</table>
"""
        metadata = {
            "document_id": "incheon_airport_digital_work",
            "agency": "인천국제공항공사",
            "domain": "공항·항공",
        }

        with TemporaryDirectory() as directory:
            path = Path(directory) / "source.md"
            path.write_text(markdown, encoding="utf-8")
            record = extract_document(path, metadata)[0]

        self.assertEqual(record["source_requirement_id"], "CUR-CM--001")
        self.assertEqual(record["requirement_id"], "CUR-CM-001")
        self.assertEqual(
            record["requirement_uid"],
            "incheon_airport_digital_work:CUR-CM-001",
        )

    def test_review_queue_contains_mismatches_and_document_coverage(self):
        records = [
            {
                "document_id": "doc_a",
                "requirement_uid": f"doc_a:SFR-00{number}",
                "requirement_id": f"SFR-00{number}",
                "raw_requirement_text": "본문" * length,
                "source_file": "doc_a.md",
                "source_location": f"markdown_line:{number}",
            }
            for number, length in [(1, 1), (2, 2), (3, 10), (4, 3), (5, 4)]
        ]
        comparisons = {
            "doc_a": {
                "index_only_ids": ["SFR-999"],
                "detail_only_ids": ["SFR-002"],
                "duplicate_index_ids": [],
            }
        }

        queue = build_review_queue(records, comparisons)

        queued = {(row["requirement_id"], row["reason"]) for row in queue}
        self.assertIn(("SFR-999", "index_only"), queued)
        self.assertIn(("SFR-002", "detail_only"), queued)
        self.assertIn(("SFR-001", "document_first"), queued)
        self.assertIn(("SFR-003", "document_middle"), queued)
        self.assertIn(("SFR-005", "document_last"), queued)
        self.assertIn(("SFR-003", "document_longest"), queued)

    def test_validate_summarizes_index_coverage(self):
        source_name = "강원랜드_생성형 AI 및 응용서비스 구축.md"
        records = [
            {
                "document_id": "kangwon_land_genai",
                "requirement_uid": "kangwon_land_genai:SFR-001",
                "requirement_name": "검색",
                "raw_requirement_text": "검색 기능을 구축한다.",
                "extraction_adapter": "labeled_id",
            }
        ]
        comparison = {
            "kangwon_land_genai": {
                "index_count": 2,
                "detail_count": 1,
                "index_only_ids": ["SFR-002"],
                "detail_only_ids": [],
                "duplicate_index_ids": [],
                "is_exact_match": False,
            }
        }

        with TemporaryDirectory() as directory:
            source_path = Path(directory) / source_name
            source_path.write_text("원문", encoding="utf-8")
            audit = validate(records, [source_path], comparison)

        self.assertEqual(audit["index_exact_match_document_count"], 0)
        self.assertEqual(audit["index_mismatch_document_count"], 1)
        self.assertEqual(audit["documents_without_index_ids"], [])
        self.assertEqual(audit["index_comparisons"], comparison)

    def test_accepted_index_exception_is_policy_resolved(self):
        comparison = {
            "index_count": 170,
            "detail_count": 169,
            "index_only_ids": [],
            "detail_only_ids": [],
            "duplicate_index_ids": ["SFR-048"],
            "is_exact_match": False,
        }

        result = apply_index_exception_policy(
            "defense_intelligent_platform", comparison
        )

        self.assertEqual(result["accepted_duplicate_index_ids"], ["SFR-048"])
        self.assertEqual(result["unresolved_duplicate_index_ids"], [])
        self.assertTrue(result["is_policy_resolved"])

    def test_unknown_index_exception_remains_unresolved(self):
        comparison = {
            "index_count": 2,
            "detail_count": 1,
            "index_only_ids": ["SFR-999"],
            "detail_only_ids": [],
            "duplicate_index_ids": [],
            "is_exact_match": False,
        }

        result = apply_index_exception_policy("unknown_document", comparison)

        self.assertEqual(result["accepted_index_only_ids"], [])
        self.assertEqual(result["unresolved_index_only_ids"], ["SFR-999"])
        self.assertFalse(result["is_policy_resolved"])

    def test_validate_counts_accepted_exception_as_policy_resolved(self):
        source_name = "국방 지능형 플랫폼 고도화 구축.md"
        records = [
            {
                "document_id": "defense_intelligent_platform",
                "requirement_uid": "defense_intelligent_platform:SFR-048",
                "requirement_name": "배포관리",
                "raw_requirement_text": "배포환경을 구축한다.",
                "extraction_adapter": "labeled_id",
            }
        ]
        comparisons = {
            "defense_intelligent_platform": {
                "index_count": 2,
                "detail_count": 1,
                "index_only_ids": [],
                "detail_only_ids": [],
                "duplicate_index_ids": ["SFR-048"],
                "is_exact_match": False,
            }
        }

        with TemporaryDirectory() as directory:
            source_path = Path(directory) / source_name
            source_path.write_text("원문", encoding="utf-8")
            audit = validate(records, [source_path], comparisons)

        self.assertEqual(audit["index_policy_resolved_document_count"], 1)
        self.assertEqual(audit["unresolved_index_mismatch_document_count"], 0)

    def test_validate_records_accepted_document_count_exception(self):
        source_name = "남동발전_남동아이 인프라 증설 및 AI 연계표준 개발 용역.md"
        records = [
            {
                "document_id": "koen_ai_infrastructure",
                "requirement_uid": "koen_ai_infrastructure:QUR-001",
                "requirement_name": "품질",
                "raw_requirement_text": "품질을 확보한다.",
                "extraction_adapter": "labeled_id",
            }
        ]
        comparisons = {
            "koen_ai_infrastructure": {
                "index_count": 0,
                "detail_count": 101,
                "index_only_ids": [],
                "detail_only_ids": [],
                "duplicate_index_ids": [],
                "is_exact_match": False,
            }
        }

        with TemporaryDirectory() as directory:
            source_path = Path(directory) / source_name
            source_path.write_text("원문", encoding="utf-8")
            audit = validate(records, [source_path], comparisons)

        exception = audit["accepted_document_exceptions"][0]
        self.assertEqual(exception["document_id"], "koen_ai_infrastructure")
        self.assertEqual(exception["summary_count"], 99)
        self.assertEqual(exception["detail_count"], 101)

    def test_write_review_queue_creates_reviewable_csv(self):
        queue = [
            {
                "document_id": "doc_a",
                "requirement_uid": "doc_a:SFR-001",
                "requirement_id": "SFR-001",
                "reason": "document_first",
                "source_file": "doc_a.md",
                "source_location": "markdown_line:10",
                "original_checked": "",
                "id_match": "",
                "name_match": "",
                "body_complete": "",
                "table_structure_ok": "",
                "review_note": "",
                "reviewer": "",
                "reviewed_at": "",
            }
        ]

        with TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "review.csv"
            write_review_queue(queue, output)
            with output.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["requirement_uid"], "doc_a:SFR-001")
        self.assertEqual(rows[0]["original_checked"], "")

    def test_write_review_queue_preserves_existing_human_review(self):
        row = {
            "document_id": "doc_a",
            "requirement_uid": "doc_a:SFR-001",
            "requirement_id": "SFR-001",
            "reason": "duplicate_index",
            "source_file": "doc_a.md",
            "source_location": "markdown_line:10",
            "original_checked": "",
            "id_match": "",
            "name_match": "",
            "body_complete": "",
            "table_structure_ok": "",
            "review_note": "",
            "reviewer": "",
            "reviewed_at": "",
        }

        with TemporaryDirectory() as directory:
            output = Path(directory) / "review.csv"
            write_review_queue([row], output)
            with output.open("r", encoding="utf-8-sig", newline="") as handle:
                saved = list(csv.DictReader(handle))
            saved[0]["original_checked"] = "true"
            saved[0]["review_note"] = "목록에 같은 ID가 두 번 기재됨"
            with output.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=saved[0].keys())
                writer.writeheader()
                writer.writerows(saved)

            refreshed = dict(row, source_location="markdown_line:20")
            write_review_queue([refreshed], output)
            with output.open("r", encoding="utf-8-sig", newline="") as handle:
                result = list(csv.DictReader(handle))[0]

        self.assertEqual(result["source_location"], "markdown_line:20")
        self.assertEqual(result["original_checked"], "true")
        self.assertEqual(result["review_note"], "목록에 같은 ID가 두 번 기재됨")


if __name__ == "__main__":
    unittest.main()
