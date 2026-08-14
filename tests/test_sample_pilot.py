#!/usr/bin/env python3
"""
scripts/data/sample_pilot.py 에 대한 단위 테스트
"""

import unittest
from scripts.data.sample_pilot import sample_pilot_dataset


class SamplePilotTests(unittest.TestCase):

    def setUp(self):
        self.records = []
        for doc_index in range(10):
            for row_index in range(5):
                requirement_id = f"REQ-{doc_index:02d}-{row_index:02d}"
                self.records.append(
                    {
                        "document_id": f"doc-{doc_index:02d}",
                        "requirement_uid": f"doc-{doc_index:02d}:{requirement_id}",
                        "requirement_id": requirement_id,
                        "source_requirement_id": requirement_id,
                        "requirement_type": "기능 요구사항",
                        "raw_requirement_text": "요구사항 본문 " + ("가" * row_index),
                    }
                )

        self.records[0]["source_requirement_id"] = "SOURCE-EXCEPTION"
        self.records[1]["raw_requirement_text"] = "상위 셀 | 중첩 셀"

    def test_sample_pilot_coverage_and_size(self):
        pilot = sample_pilot_dataset(self.records, target_size=40)
        
        # 1. 크기 검증 (35 ~ 45건)
        self.assertTrue(35 <= len(pilot) <= 45, f"표본 크기 범위 이탈: {len(pilot)}")

        # 2. 10개 문서 100% 커버리지 검증
        docs = set(r["document_id"] for r in pilot)
        all_docs = set(r["document_id"] for r in self.records)
        self.assertEqual(docs, all_docs, "10개 문서가 모두 포함되어야 함")

        # 3. 승인 예외 ID 포함 여부 검증
        mismatches = [r for r in pilot if r.get("requirement_id") != r.get("source_requirement_id")]
        self.assertTrue(len(mismatches) >= 1, "승인 예외 ID(인천공항)가 포함되어야 함")

        # 4. 중첩표 포함 여부 검증
        nested = [r for r in pilot if " | " in r.get("raw_requirement_text", "")]
        self.assertTrue(len(nested) >= 1, "중첩표 포함 샘플이 수록되어야 함")


if __name__ == "__main__":
    unittest.main()
