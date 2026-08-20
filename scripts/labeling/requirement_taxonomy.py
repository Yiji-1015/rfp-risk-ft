"""요구사항 유형 표기를 정본 분류로 정규화한다.

원본 `requirement_type`은 60종이다. 추출기가 문서의 표기를 그대로 옮기기 때문에
"기능 요구사항" / "기능" / "기능 요구사항(SFR)" / "기능요구사항"이 전부 별개로 남는다.
그 상태로는 유형별 분석이 성립하지 않는다. 표기가 문서와 얽혀 있어 유형 효과와
문서 효과가 분리되지 않기 때문이다(docs/issues/001).

정본은 **공공 SW 제안요청서 표준 11분류**를 쓴다. 표준 접두어(SFR·PER·ECR·INR·DAR·
TER·SER·QUR·COR·PMR·PSR)와 1:1로 대응해서 검증이 쉽다. 여기에 `컨설팅`만 더한다.
표준 11분류에는 없지만 ISP·ISMP 컨설팅 요구사항은 판정 기준이 뚜렷이 다르고
프롬프트에도 별도 보정 블록을 두고 있어(결정 21), 합치면 정보가 사라진다.

**표기가 접두어보다 우선한다.** 접두어는 문서를 넘어 충돌하기 때문이다. 실제로
`kexim_ai_platform` 한 문서가 표준 접두어를 다른 뜻으로 쓴다.

    INR -> 인터페이스(5개 문서) 이지만 kexim에서는 인프라
    ECR -> 시스템장비구성(5개 문서) 이지만 kexim에서는 ECM
    SER -> 보안(9개 문서) 이지만 kexim에서는 서비스

접두어는 표기가 아예 없는 문서(`koen_ai_infrastructure` 101건)에서만 대체로 쓴다.
그 문서는 접두어 충돌이 없어 안전하다.
"""

from __future__ import annotations

import re
from typing import Literal

CanonicalType = str

# 공공 SW 제안요청서 표준 11분류 + 컨설팅
CANONICAL_TYPES: tuple[str, ...] = (
    "기능",
    "성능",
    "시스템장비구성",
    "인터페이스",
    "데이터",
    "테스트",
    "보안",
    "품질",
    "제약사항",
    "프로젝트관리",
    "프로젝트지원",
    "컨설팅",
)
UNKNOWN = "기타"

# 정규화된 표기 -> 정본. 판단이 들어간 매핑은 주석으로 근거를 남긴다.
TEXT_TO_CANONICAL: dict[str, CanonicalType] = {
    "기능": "기능",
    "그룹웨어기능": "기능",          # GW-001~ 전자결재·포탈 등 업무 기능
    "AI활용업무기능": "기능",
    "AI기반솔루션": "기능",          # ASR-001~ 프롬프트 도구·RAG·Agent 기능
    "ECM": "기능",                  # kexim ECR-001~ 문서 중앙저장소 구축 기능
    "서비스": "기능",                # kexim SER-001~ AI 대화형·화면 서비스
    "성능": "성능",
    "시스템장비구성": "시스템장비구성",
    "시스템장비": "시스템장비구성",
    "시스템": "시스템장비구성",
    "시스템구성": "시스템장비구성",
    "AI플랫폼및인프라": "시스템장비구성",
    "인프라": "시스템장비구성",       # kexim INR-001~ 인프라 상세. 표준의 장비구성에 해당
    "인프라상세": "시스템장비구성",
    "인터페이스": "인터페이스",
    "데이터": "데이터",
    "테스트": "테스트",
    "보안": "보안",
    "프로젝트보안": "보안",
    "품질": "품질",
    "품질관리": "품질",
    "제약사항": "제약사항",
    "안전": "제약사항",              # SAR-001~ 안전보건관리체계. 법규 준수 제약
    "프로젝트관리": "프로젝트관리",
    "거버넌스및PMO": "프로젝트관리",  # GOV-001~ AI 거버넌스·위험관리 체계 수립
    "프로젝트지원": "프로젝트지원",
    "컨설팅": "컨설팅",
}

# 표기가 없는 문서에서만 쓰는 대체 경로. 현재는 koen_ai_infrastructure 뿐이다.
PREFIX_TO_CANONICAL: dict[str, CanonicalType] = {
    "SFR": "기능",
    "FUN": "기능",
    "PER": "성능",
    "ECR": "시스템장비구성",
    "SYS": "시스템장비구성",
    "INF": "시스템장비구성",
    "INR": "인터페이스",
    "SIR": "인터페이스",
    "INT": "인터페이스",
    "DAR": "데이터",
    "DAT": "데이터",
    "TER": "테스트",
    "TST": "테스트",
    "SER": "보안",
    "SEC": "보안",
    "QUR": "품질",
    "QMR": "품질",
    "COR": "제약사항",
    "SAR": "제약사항",
    "PMR": "프로젝트관리",
    "GOV": "프로젝트관리",
    "PSR": "프로젝트지원",
    "CNR": "컨설팅",
    "CUR": "컨설팅",
    "CSR": "컨설팅",
    "CON": "컨설팅",   # koen CON-001 "전사 AI 연계 표준체계 수립". 제약(COR)과 별개로 쓴다
}

_CODE_SUFFIX = re.compile(r"\([A-Z]{2,4}\)")

TypeSource = Literal["text", "prefix", "none"]


def normalize_text(requirement_type: str | None) -> str | None:
    """표기에서 괄호 코드·공백·`요구사항` 접미어를 떼어 비교 가능한 형태로 만든다."""
    if not requirement_type:
        return None
    stripped = (
        _CODE_SUFFIX.sub("", requirement_type).replace(" ", "").replace("요구사항", "")
    )
    return stripped or None


def prefix_of(requirement_id: str) -> str:
    """`SFR-001` 또는 `CUR-CM-001`에서 앞 토큰을 뽑는다."""
    return requirement_id.split("-", 1)[0]


def normalize_requirement_type(
    requirement_type: str | None,
    requirement_id: str,
) -> tuple[CanonicalType, TypeSource]:
    """정본 유형과 그 근거를 반환한다.

    :returns: (정본 유형, 'text' | 'prefix' | 'none')

    표기를 먼저 본다. 접두어는 문서를 넘어 충돌하므로 표기가 있으면 그쪽이 사실에
    가깝다. 어느 쪽으로도 결정되지 않으면 `기타`로 두고 근거를 `none`으로 남긴다.
    조용히 최빈값으로 채우지 않는다.
    """
    text = normalize_text(requirement_type)
    if text is not None:
        canonical = TEXT_TO_CANONICAL.get(text)
        if canonical is not None:
            return canonical, "text"

    canonical = PREFIX_TO_CANONICAL.get(prefix_of(requirement_id))
    if canonical is not None:
        return canonical, "prefix"

    return UNKNOWN, "none"
