#!/usr/bin/env python3
"""
scripts/data/eda_requirements.py 에 대한 단위 테스트
"""

import unittest
from scripts.data.eda_requirements import (
    normalize_requirement_type,
    calculate_stats,
    analyze_dataset,
    generate_markdown_report,
    generate_jupyter_notebook
)


class EDARequirementsTests(unittest.TestCase):

    def setUp(self):
        self.mock_records = [
            {
                "dataset_version": "requirements_v0.2.0",
                "document_id": "doc_a",
                "requirement_uid": "doc_a:REQ-001",
                "requirement_id": "REQ-001",
                "source_requirement_id": "REQ-001",
                "requirement_type": "기능 요구사항(SFR)",
                "requirement_name": "테스트 1",
                "raw_requirement_text": "첫 번째 테스트 본문입니다."
            },
            {
                "dataset_version": "requirements_v0.2.0",
                "document_id": "doc_a",
                "requirement_uid": "doc_a:REQ-002",
                "requirement_id": "REQ-002",
                "source_requirement_id": "REQ--002",  # ID mismatch
                "requirement_type": "성능 요구사항",
                "requirement_name": "테스트 2",
                "raw_requirement_text": "셀 구분자 | 중첩표가 포함된 | 두 번째 본문입니다."
            },
            {
                "dataset_version": "requirements_v0.2.0",
                "document_id": "doc_b",
                "requirement_uid": "doc_b:REQ-001",
                "requirement_id": "REQ-001",
                "source_requirement_id": "REQ-001",
                "requirement_type": "기능",
                "requirement_name": "테스트 3",
                "raw_requirement_text": "짧은 본문"
            }
        ]

    def test_normalize_requirement_type(self):
        self.assertEqual(normalize_requirement_type("기능 요구사항"), "기능 요구사항")
        self.assertEqual(normalize_requirement_type("기능"), "기능 요구사항")
        self.assertEqual(normalize_requirement_type("기능 요구사항(SFR)"), "기능 요구사항")
        self.assertEqual(normalize_requirement_type("성능 요구사항(PER)"), "성능 요구사항")
        self.assertEqual(normalize_requirement_type("보안"), "보안 요구사항")
        self.assertEqual(normalize_requirement_type("None"), "미지정 (None)")

    def test_calculate_stats(self):
        vals = [10, 20, 30, 40, 50]
        stats = calculate_stats(vals)
        self.assertEqual(stats["min"], 10)
        self.assertEqual(stats["max"], 50)
        self.assertEqual(stats["median"], 30.0)
        self.assertEqual(stats["mean"], 30.0)

    def test_analyze_dataset(self):
        res = analyze_dataset(self.mock_records)
        self.assertEqual(res["total_records"], 3)
        self.assertEqual(res["document_counts"]["doc_a"], 2)
        self.assertEqual(res["document_counts"]["doc_b"], 1)
        self.assertEqual(res["normalized_type_counts"]["기능 요구사항"], 2)
        self.assertEqual(res["normalized_type_counts"]["성능 요구사항"], 1)
        
        # ID mismatch
        self.assertEqual(len(res["id_mismatches"]), 1)
        self.assertEqual(res["id_mismatches"][0]["requirement_id"], "REQ-002")

        # Nested table
        self.assertEqual(res["nested_table_count"], 1)

    def test_generate_markdown_report(self):
        res = analyze_dataset(self.mock_records)
        md_text = generate_markdown_report(res)
        self.assertIn("# 요구사항 데이터셋 v0.2.0 EDA 보고서", md_text)
        self.assertIn("`기능 요구사항`", md_text)

    def test_generate_jupyter_notebook(self):
        res = analyze_dataset(self.mock_records)
        nb_json = generate_jupyter_notebook(res)
        self.assertEqual(nb_json["nbformat"], 4)
        self.assertTrue(len(nb_json["cells"]) >= 6)


if __name__ == "__main__":
    unittest.main()
