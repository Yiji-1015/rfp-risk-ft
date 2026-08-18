"""Canonical structured output schema for RFP risk labeling.

주 라벨은 가격 산정 가능성이다(PROJECT_DIRECTION §5.1). 보조 축은 그 판정의
이유를 분리해 기록한다 — 같은 `계약·질의검토`라도 범위가 열려서인지, 기술
실현성이 미확인이라서인지, 조달 자체가 불투명해서인지는 실무 조치가 다르다.

`blockers`는 비용 발생 여부가 아니라 **제안·입찰 전에 반드시 확인해야 안전하게
수용할 수 있는 조건**을 뜻한다(결정 21). 실무자 검토에서 한 요구사항이 복수의
blocker를 동시에 가질 수 있음이 확인되어 리스트로 둔다. 빈 리스트는 blocker 없음.

`domain_dependency`는 §7.2가 주 라벨과 분리하라고 한 보조 축이다. 문서 간
재출현률로 독립 검증한다.

`build_difficulty`는 도메인 난이도를 걷어낸 순수 구축 난이도다. 난이도가 높다는
사실만으로 주 라벨을 올리지 않으며, 고급·전문 인력 투입으로 이어질 때만
`견적반영`이 된다(결정 21).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "4.0.0"

REASONING_MAX_LENGTH = 300

PRIMARY_ACTIONS = ("통상수용", "견적반영", "계약·질의검토")

BLOCKER_TYPES = (
    "범위·책임",
    "검수·성능기준",
    "기술실현성",
    "라이선스·공급",
    "공급자종속",
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LabelResult(StrictModel):
    requirement_uid: str = Field(min_length=1)
    primary_action: Literal["통상수용", "견적반영", "계약·질의검토"]
    blockers: list[
        Literal[
            "범위·책임",
            "검수·성능기준",
            "기술실현성",
            "라이선스·공급",
            "공급자종속",
        ]
    ]
    cost_basis: Literal[
        "없음",
        "고급·전문인력",
        "장비·인프라",
        "라이선스",
        "외부인증",
        "외주·전문기관",
        "복합",
    ]
    domain_dependency: Literal["높음", "보통", "낮음"]
    build_difficulty: Literal["높음", "보통", "낮음"]
    reasoning: str = Field(min_length=1, max_length=REASONING_MAX_LENGTH)


def derive_primary_action(label: LabelResult) -> str:
    """결정 21의 고정 규칙으로 주 라벨을 도출한다.

    LLM이 직접 출력한 `primary_action`과 비교하면 §8.2의 "직접 3분류 vs 위험
    요인 분해" 비교가 추가 호출 없이 성립한다.
    """
    if label.blockers:
        return "계약·질의검토"
    if label.cost_basis != "없음":
        return "견적반영"
    return "통상수용"
