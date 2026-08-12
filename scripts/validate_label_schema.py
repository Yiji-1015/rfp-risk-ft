#!/usr/bin/env python3
"""
LLM 라벨링 파일럿 입출력 JSON 스키마 검증 유틸리티
"""

import json
from typing import Dict, Any, Tuple, List


VALID_PRIMARY_ACTIONS = {"통상수용", "견적반영", "계약·질의검토"}
VALID_CONFIDENCE_LEVELS = {"높음", "중간", "낮음"}
VALID_DOMAIN_LEVELS = {"높음", "중간", "낮음"}
VALID_SUPPORT_STATUSES = {"발주처 제공", "공동 수행", "수행사 전담", "미지정"}


def validate_label_output(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """LLM 출력 JSON 데이터 무결성 검증"""
    errors = []
    
    # 필수 톱레벨 키 검사
    required_keys = [
        "requirement_uid",
        "primary_action",
        "confidence",
        "reasoning",
        "evidence",
        "missing_information",
        "domain_dependency",
        "risk_factors"
    ]
    for k in required_keys:
        if k not in data:
            errors.append(f"필수 키 누락: '{k}'")

    if errors:
        return False, errors

    # 1. 주 라벨 검사
    action = data.get("primary_action")
    if action not in VALID_PRIMARY_ACTIONS:
        errors.append(f"유효하지 않은 primary_action: '{action}' (허용: {VALID_PRIMARY_ACTIONS})")

    # 2. 확신도 검사
    conf = data.get("confidence")
    if conf not in VALID_CONFIDENCE_LEVELS:
        errors.append(f"유효하지 않은 confidence: '{conf}'")

    # 3. 근거 배열 검사
    ev = data.get("evidence")
    if not isinstance(ev, list):
        errors.append("evidence 필드는 리스트 형태여야 합니다.")

    # 4. 정보 부족 객체 검사
    mi = data.get("missing_information", {})
    if not isinstance(mi, dict) or "is_missing" not in mi:
        errors.append("missing_information 객체에 'is_missing' (boolean) 필드가 필요합니다.")

    # 5. 도메인 의존성 객체 검사
    dd = data.get("domain_dependency", {})
    if not isinstance(dd, dict) or "level" not in dd:
        errors.append("domain_dependency 객체에 'level' 필드가 필요합니다.")
    elif dd.get("level") not in VALID_DOMAIN_LEVELS:
        errors.append(f"유효하지 않은 domain_dependency.level: '{dd.get('level')}'")

    # 6. 세부 위험 요인 객체 검사
    rf = data.get("risk_factors", {})
    if not isinstance(rf, dict):
        errors.append("risk_factors 필드는 객체 형태여야 합니다.")

    return len(errors) == 0, errors
