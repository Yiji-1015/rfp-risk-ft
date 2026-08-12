# LLM 라벨링 파일럿 입출력 JSON 스키마 명세 (Draft v0.1.0)

> **기준일**: 2026-08-13  
> **상태**: 초안 (Draft)  
> **목적**: 1,024개 RFP 요구사항에 대해 LLM 라벨링 파일럿을 수행하기 위한 입력 프롬프트 구조, 주 라벨 가설, 보조 위험 요소 및 출력 JSON Schema를 정의한다.

---

## 1. 기본 원칙 및 범위

1. **학습·라벨링 입력 정보 고정**:
   - LLM 라벨러와 후속 ML/파인튜닝 모델은 **동일하게 `요구사항명 + 요구사항 본문`만** 입력받는다.
   - 전체 RFP 문맥, 사업기간, 타 요구사항의 예외 조건 등 요구사항 단독 본문에 포함되지 않은 정보는 추측하지 않는다.
2. **기준 사용자 (페르소나)**:
   - 식별자: `persona_ai_it_proposal_writer_v1`
   - 공공 AI·IT 구축사의 제안서 작성자 (일반 AI/IT/SI는 이해하지만 의약품·국방·금융 등 산업별 전문가는 아님).
3. **주 라벨 타깃**:
   - 추상적 손실 확률이 아니라 **견적·계약 실무 검토 조치 우선순위 (3분류)**를 예측한다.

---

## 2. 주 라벨 (Primary Label) 정의

| 주 라벨 (`primary_action`) | 의미 | 대표 판단 상황 |
|---|---|---|
| **`통상수용`** | 별도 추가 견적이나 계약 질의 없이 일반적인 수용 가능 | 표준 보고서 작성, 일반적인 시범운영, 통상적 테스트/교육 |
| **`견적반영`** | 수량·인력·기간·사양이 명확하여 제안 견적에 금액으로 반영 가능 | 상주 인력 수, 구체적 장비 구매/투입, 명시된 교육 횟수, 무상보증 기간 |
| **`계약·질의검토`** | 범위·책임·검수 기준이 모호하거나 수행사 포괄 책임이 포함되어 위험함 | 추가 개발 무상 요구, 불명확한 검수/성능합격 기준, 포괄적 손해배상/저작권 양도 |

---

## 3. 보조 위험 요소 (Risk Factors) 및 도메인 축

주 라벨의 이유를 정밀하게 설명하고 라벨 감사 및 오류 분석에 활용하기 위해 다음 보조 필드를 출력한다.

1. **`missing_information` (정보 부족 여부)**:
   - 본 요구사항 단독 본문만으로 비용/범위를 완결짓기 어려운 경우 `true`로 설정하고 사유 기록.
2. **`domain_dependency` (도메인 의존성 보조 축)**:
   - `level`: `"높음"` | `"중간"` | `"낮음"` (의약품 인허가, 국방 RMF, 채무조정 등 전문 도메인 지식 필요도)
   - `domain_support`: `"발주처 제공"` | `"공동 수행"` | `"수행사 전담"` | `"미지정"`
3. **`risk_factors` (세부 요인 분석)**:
   - `cost_driver`: 비용 발생 항목 (인력, HW/SW 라이선스, 외부 검증 등)
   - `scope_uncertainty`: 범위/수량 상한의 명확성
   - `responsibility_risk`: 수행사 독소/포괄 책임 위험 여부
   - `acceptance_risk`: 검수/성능 평가 기준의 명확성
4. **`evidence` (원문 인출 근거)**:
   - 해당 라벨 판단에 결정적 영향을 미친 요구사항 본문 내 원문 구문 추출 (1~3문장).

---

## 4. 입출력 JSON Schema 명세

### 4.1 LLM 프롬프트 입력 JSON (`input_schema`)

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "RFPRequirementInput",
  "type": "object",
  "properties": {
    "requirement_uid": {
      "type": "string",
      "description": "문서ID와 요구사항ID 결합 고유키 (예: ccrs_ai_platform:SFR-001)"
    },
    "document_id": {
      "type": "string",
      "description": "RFP 문서 식별자"
    },
    "requirement_id": {
      "type": "string",
      "description": "요구사항 Canonical ID"
    },
    "requirement_type": {
      "type": "string",
      "description": "원문/정규화 요구사항 유형"
    },
    "requirement_name": {
      "type": "string",
      "description": "원문 요구사항 명칭"
    },
    "requirement_text": {
      "type": "string",
      "description": "요구사항 본문 (정규화 불릿/중첩표 보존)"
    }
  },
  "required": ["requirement_uid", "document_id", "requirement_id", "requirement_name", "requirement_text"]
}
```

### 4.2 LLM 출력 JSON (`output_schema`)

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "RFPRiskAssessmentOutput",
  "type": "object",
  "properties": {
    "requirement_uid": {
      "type": "string",
      "description": "입력받은 requirement_uid 그대로 반환"
    },
    "primary_action": {
      "type": "string",
      "enum": ["통상수용", "견적반영", "계약·질의검토"],
      "description": "주 검토 조치 예측 타깃 (3분류)"
    },
    "confidence": {
      "type": "string",
      "enum": ["높음", "중간", "낮음"],
      "description": "LLM 라벨 판정의 자체 확신도"
    },
    "reasoning": {
      "type": "string",
      "description": "주 라벨 결정 이유 요약 (2-3문장)"
    },
    "evidence": {
      "type": "array",
      "items": { "type": "string" },
      "description": "라벨 결정의 근거가 된 원문 텍스트 인용구 (1~3개)"
    },
    "missing_information": {
      "type": "object",
      "properties": {
        "is_missing": { "type": "boolean" },
        "missing_details": { "type": "string", "description": "부족한 정보 항목 (예: 전체 문서 인프라 제공 조건 확인 필요 등)" }
      },
      "required": ["is_missing"]
    },
    "domain_dependency": {
      "type": "object",
      "properties": {
        "level": { "type": "string", "enum": ["높음", "중간", "낮음"] },
        "domain_name": { "type": "string", "description": "도메인 분야 (예: 금융·채무조정, 국방 RMF 등)" },
        "support_status": { "type": "string", "enum": ["발주처 제공", "공동 수행", "수행사 전담", "미지정"] }
      },
      "required": ["level"]
    },
    "risk_factors": {
      "type": "object",
      "properties": {
        "cost_driver": { "type": "string", "description": "비용 반영 필요 요소 (인력, 장비, 라이선스 등)" },
        "scope_uncertainty": { "type": "string", "description": "범위 모호성 및 상한 미지정 위험" },
        "responsibility_risk": { "type": "string", "description": "수행사 포괄/무상 책임 위험" },
        "acceptance_risk": { "type": "string", "description": "불명확한 검수/성능합격 기준 위험" }
      }
    }
  },
  "required": [
    "requirement_uid",
    "primary_action",
    "confidence",
    "reasoning",
    "evidence",
    "missing_information",
    "domain_dependency",
    "risk_factors"
  ]
}
```

---

## 5. 검증 및 다음 파일럿 적용 방안

1. **스키마 검증**:
   - `jsonschema` 패키지 또는 Python `pydantic` 모델을 작성하여 LLM 파이프라인 응답에 대한 무결성 테스트 구축.
2. **라벨링 실험 비교 준비**:
   - Zero-shot / Global Few-shot / Dynamic Few-shot / 위험 요인 분해 후 주 라벨 매핑 방식을 동일한 이 스키마 포맷으로 통일하여 비교 가능성 유지.
