#!/usr/bin/env python3
"""
scripts/labeling/validate_label_schema.py 단위 테스트
"""

import unittest
import pytest

from scripts.labeling.label_schema import (
    REASONING_MAX_LENGTH,
    LabelResult,
    derive_primary_action,
)
from scripts.labeling.validate_label_schema import validate_label_output


class LabelSchemaValidationTests(unittest.TestCase):

    def setUp(self):
        self.valid_output = {
            "requirement_uid": "ccrs_ai_platform:SFR-001",
            "primary_action": "통상수용",
            "blockers": [],
            "cost_basis": "없음",
            "domain_dependency": "낮음",
            "build_difficulty": "낮음",
            "reasoning": "단일 웹 UI 포털과 권한 관리로, 통상적인 SI 구축 범위에 해당한다.",
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
        del invalid_data["reasoning"]
        is_valid, errors = validate_label_output(invalid_data)
        self.assertFalse(is_valid)
        self.assertTrue(any("reasoning" in e for e in errors))

    def test_pydantic_schema_forbids_extra_fields(self):
        invalid_data = dict(self.valid_output, unexpected="nope")
        with pytest.raises(ValueError):
            LabelResult.model_validate(invalid_data)

    def test_removed_auxiliary_fields_are_rejected(self):
        invalid_data = dict(self.valid_output, blocker="기술범위")
        with pytest.raises(ValueError):
            LabelResult.model_validate(invalid_data)

    def test_removed_confidence_field_is_rejected(self):
        invalid_data = dict(self.valid_output, confidence="높음")
        with pytest.raises(ValueError):
            LabelResult.model_validate(invalid_data)

    def test_v2_output_without_auxiliary_axes_is_rejected(self):
        """스키마 v2로 생성된 결과가 v3 실행에 섞여 들어오면 검증에서 걸려야 한다."""
        v2_output = {
            "requirement_uid": "doc:SFR-001",
            "primary_action": "통상수용",
            "reasoning": "통상적인 SI 구축 범위다.",
        }
        is_valid, errors = validate_label_output(v2_output)
        self.assertFalse(is_valid)
        self.assertTrue(any("blockers" in e for e in errors))

    def test_build_difficulty_and_domain_dependency_are_independent_fields(self):
        """두 축을 섞지 않는 것이 결정 20의 요지이므로 각각 독립적으로 검증한다."""
        for field in ("domain_dependency", "build_difficulty"):
            invalid = dict(self.valid_output, **{field: "중간"})
            with pytest.raises(ValueError):
                LabelResult.model_validate(invalid)

    def test_pydantic_schema_limits_reasoning_length(self):
        invalid_data = dict(
            self.valid_output,
            reasoning="가" * (REASONING_MAX_LENGTH + 1),
        )
        with pytest.raises(ValueError):
            LabelResult.model_validate(invalid_data)


if __name__ == "__main__":
    unittest.main()


class DerivePrimaryActionTests(unittest.TestCase):
    """결정 21의 고정 규칙: blocker 있음 → 계약·질의검토, 원가 있음 → 견적반영."""

    def _label(self, **overrides):
        base = {
            "requirement_uid": "doc:SFR-001",
            "primary_action": "통상수용",
            "blockers": [],
            "cost_basis": "없음",
            "domain_dependency": "낮음",
            "build_difficulty": "낮음",
            "reasoning": "본문 근거.",
        }
        base.update(overrides)
        return LabelResult.model_validate(base)

    def test_any_blocker_forces_contract_review(self):
        label = self._label(blockers=["라이선스·공급"], cost_basis="장비·인프라")
        self.assertEqual(derive_primary_action(label), "계약·질의검토")

    def test_cost_without_blocker_is_quotable(self):
        self.assertEqual(
            derive_primary_action(self._label(cost_basis="고급·전문인력")),
            "견적반영",
        )

    def test_no_blocker_no_cost_is_routine(self):
        self.assertEqual(derive_primary_action(self._label()), "통상수용")

    def test_high_difficulty_alone_does_not_raise_the_label(self):
        """난이도가 높다는 사실만으로 주 라벨을 올리지 않는다(결정 21)."""
        label = self._label(build_difficulty="높음", domain_dependency="높음")
        self.assertEqual(derive_primary_action(label), "통상수용")
