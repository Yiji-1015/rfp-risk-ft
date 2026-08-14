#!/usr/bin/env python3
"""
LLM 라벨링 파일럿 입출력 JSON 스키마 검증 유틸리티
"""

from typing import Any, Dict, List, Tuple

from pydantic import ValidationError

from scripts.labeling.label_schema import LabelResult


def validate_label_output(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """LLM 출력 JSON 데이터 무결성 검증"""
    try:
        LabelResult.model_validate(data)
    except ValidationError as exc:
        errors = []
        for error in exc.errors(include_url=False):
            location = ".".join(str(part) for part in error["loc"])
            errors.append(f"{location}: {error['msg']}")
        return False, errors
    return True, []
