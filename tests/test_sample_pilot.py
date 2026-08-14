#!/usr/bin/env python3
"""
scripts/data/sample_pilot.py 에 대한 단위 테스트
"""

import unittest
from pathlib import Path
from scripts.data.sample_pilot import load_dataset, sample_pilot_dataset


class SamplePilotTests(unittest.TestCase):

    def setUp(self):
        root_dir = Path(__file__).resolve().parent.parent
        self.dataset_path = root_dir / "data" / "processed" / "requirements_v0.2.0.jsonl"
        self.records = load_dataset(str(self.dataset_path))

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
