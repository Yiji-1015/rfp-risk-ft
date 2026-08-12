#!/usr/bin/env python3
"""
scripts/validate_label_schema.py 단위 테스트
"""

import unittest
from scripts.validate_label_schema import validate_label_output


class LabelSchemaValidationTests(unittest.TestCase):

    def setUp(self):
        self.valid_output = {
            "requirement_uid": "ccrs_ai_platform:SFR-001",
            "primary_action": "통상수용",
            "confidence": "높음",
            "reasoning": "기본적인 단일 UI 포털 구성 및 권한 관리 기능으로 통상적인 SI 구축 범위에 해당합니다.",
            "evidence": ["AI 서비스, MLOps 운영, 플랫폼 관리 기능 전체를 단일 웹 UI에서 접근"],
            "missing_information": {
                "is_missing": False,
                "missing_details": ""
            },
            "domain_dependency": {
                "level": "낮음",
                "domain_name": "일반 IT/UI",
                "support_status": "미지정"
            },
            "risk_factors": {
                "cost_driver": "표준 Web UI 개발",
                "scope_uncertainty": "없음",
                "responsibility_risk": "없음",
                "acceptance_risk": "통상적인 기능 테스트"
            }
        }

    def test_valid_label_output(self):
        is_valid, errors = validate_label_output(self.valid_output)
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)

    def test_invalid_primary_action(self):
        invalid_data = self.valid_output.copy()
        invalid_data["primary_action"] = "무조건거절"
        is_valid, errors = validate_label_output(invalid_data)
        self.assertFalse(is_valid)
        self.assertTrue(any("primary_action" in e for e in errors))

    def test_missing_required_key(self):
        invalid_data = self.valid_output.copy()
        del invalid_data["missing_information"]
        is_valid, errors = validate_label_output(invalid_data)
        self.assertFalse(is_valid)
        self.assertTrue(any("missing_information" in e for e in errors))


if __name__ == "__main__":
    unittest.main()
