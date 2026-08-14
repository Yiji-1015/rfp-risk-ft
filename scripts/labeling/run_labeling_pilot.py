#!/usr/bin/env python3
"""
RFP 요구사항 동결 데이터셋 v0.2.0 LLM 라벨링 파일럿 실행 스크립트 (자동 가용 모델 Fallback & Resume)

수행 작업:
1. .env 의 GEMINI_API_KEY 로드
2. data/samples/labeling_pilot_sample_v0.1.0.jsonl (40건) 데이터 읽기
3. docs/LABELING_SCHEMA_DRAFT.md 명세에 맞춘 System Prompt 및 JSON Schema 설정
4. 404 / 429 에러 시 가용 모델 순환 및 Exponential Backoff 재시도
5. 이미 라벨링된 건은 이어하기(Resume) 처리하여 40건 100% 완성
6. 결과를 reports/current/labeling_pilot_results_v0.1.0.jsonl 및 CSV로 저장
"""

import os
import sys
import json
import time
import csv
from pathlib import Path
from typing import Dict, Any

# Windows 인코딩 대응
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from dotenv import load_dotenv
import google.genai as genai
from google.genai import types

from scripts.labeling.validate_label_schema import validate_label_output

# .env 파일 로드
root_dir = Path(__file__).resolve().parents[2]
load_dotenv(root_dir / ".env")

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("오류: .env 파일에 GEMINI_API_KEY가 설정되어 있지 않습니다.")
    sys.exit(1)

client = genai.Client(api_key=api_key)

# 1. System Prompt 명세
SYSTEM_PROMPT = """너는 한국 공공 AI·IT 구축 RFP를 검토하여 견적 및 계약상 위험 조치를 판단하는 15년 차 수석 제안서 작성자(Persona: persona_ai_it_proposal_writer_v1)이다.

[검토 원칙]
1. 오직 입력으로 제공된 [요구사항명]과 [요구사항 내용]만 읽고 판단해라. 전체 RFP의 사업기간, 타 요구사항 예외, 예산 등 본문에 없는 정보는 추측하지 마라.
2. 주 타깃 라벨(primary_action)은 다음 3개 중 하나로만 판단해라:
   - "통상수용": 별도 추가 견적이나 계약 질의 없이 일반적인 SI 표준 수용 가능. (보고서 작성, 표준 시범운영, 통상적 테스트/교육 등)
   - "견적반영": 부담은 있으나 범위/수량/인력/장비/기간이 명확하여 제안 견적금액에 산정 가능한 경우. (상주인력 수, 명시된 장비 구매, 구체적 교육 횟수, 무상보증 기간 등)
   - "계약·질의검토": 범위나 책임이 모호하거나, 무상 추가개발 요구, 불명확한 검수/성능합격 기준, 포괄적 손해배상/저작권 양도 등 제안사 위험 조항인 경우.

3. 판단에 결정적 영향을 미친 요구사항 본문 내 문장을 evidence 리스트에 1~3개 인용해라.
4. 요구사항 단독 본문으로 비용/범위를 완성하기 어려운 경우 missing_information.is_missing을 true로 두고 부족한 사유를 기록해라.
5. 도메인 지식(의약품, 국방, 채무조정 등) 필요도(level: 높음/중간/낮음) 및 발주처 지원 상태(support_status: 발주처 제공/공동 수행/수행사 전담/미지정)를 작성해라.
6. 반드시 지정된 JSON Schema에 맞추어 답변해라."""

# 2. JSON Schema 출력 정의
RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "requirement_uid": {"type": "STRING"},
        "primary_action": {
            "type": "STRING",
            "enum": ["통상수용", "견적반영", "계약·질의검토"]
        },
        "confidence": {
            "type": "STRING",
            "enum": ["높음", "중간", "낮음"]
        },
        "reasoning": {"type": "STRING"},
        "evidence": {
            "type": "ARRAY",
            "items": {"type": "STRING"}
        },
        "missing_information": {
            "type": "OBJECT",
            "properties": {
                "is_missing": {"type": "BOOLEAN"},
                "missing_details": {"type": "STRING"}
            },
            "required": ["is_missing"]
        },
        "domain_dependency": {
            "type": "OBJECT",
            "properties": {
                "level": {"type": "STRING", "enum": ["높음", "중간", "낮음"]},
                "domain_name": {"type": "STRING"},
                "support_status": {"type": "STRING", "enum": ["발주처 제공", "공동 수행", "수행사 전담", "미지정"]}
            },
            "required": ["level"]
        },
        "risk_factors": {
            "type": "OBJECT",
            "properties": {
                "cost_driver": {"type": "STRING"},
                "scope_uncertainty": {"type": "STRING"},
                "responsibility_risk": {"type": "STRING"},
                "acceptance_risk": {"type": "STRING"}
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

CANDIDATE_MODELS = [
    "gemini-3.6-flash",
    "gemini-flash-latest",
    "gemini-2.5-flash-lite"
]


def run_single_labeling_with_fallback(sample: Dict[str, Any]) -> Dict[str, Any]:
    """가용 모델 순환 및 Rate limit 대기 재시도"""
    req_uid = sample["requirement_uid"]
    req_name = sample["requirement_name"]
    req_text = sample["raw_requirement_text"]
    
    user_prompt = f"""[요구사항 ID]: {req_uid}
[요구사항명]: {req_name}
[요구사항 내용]:
{req_text}"""

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        response_mime_type="application/json",
        response_schema=RESPONSE_SCHEMA,
        temperature=0.1
    )

    last_err = None
    for model_name in CANDIDATE_MODELS:
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=user_prompt,
                    config=config
                )
                res_json = json.loads(response.text)
                res_json["requirement_uid"] = req_uid
                return res_json
            except Exception as e:
                err_str = str(e)
                last_err = e
                if "404" in err_str or "NOT_FOUND" in err_str:
                    # 해당 모델 미지원 -> 다음 모델 후보로 스위치
                    break
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_msg:
                    print(f"    [429 Quota Exceeded] {model_name} -> 20초 대기 후 재시도...")
                    time.sleep(20)
                else:
                    time.sleep(3)

    raise last_err if last_err else RuntimeError("모든 모델 시도 실패")


def main():
    sample_file = root_dir / "data" / "samples" / "labeling_pilot_sample_v0.1.0.jsonl"
    if not sample_file.exists():
        print(f"오류: 표본 파일이 존재하지 않습니다 -> {sample_file}")
        sys.exit(1)

    with open(sample_file, "r", encoding="utf-8") as f:
        samples = [json.loads(line.strip()) for line in f if line.strip()]

    reports_dir = root_dir / "reports" / "current"
    reports_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl = reports_dir / "labeling_pilot_results_v0.1.0.jsonl"
    output_csv = reports_dir / "labeling_pilot_results_v0.1.0.csv"

    # 기존 결과 이어하기(Resume) 체크
    completed_results = {}
    if output_jsonl.exists():
        with open(output_jsonl, "r", encoding="utf-8") as f_prev:
            for line in f_prev:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    if "pilot_result" in item and item["pilot_result"].get("primary_action"):
                        completed_results[item["requirement_uid"]] = item

    print(f"=== LLM 라벨링 파일럿 시작 (기존 완료: {len(completed_results)}/{len(samples)}건) ===")

    final_results = []
    
    with open(output_jsonl, "w", encoding="utf-8") as f_jsonl:
        for idx, sample in enumerate(samples, 1):
            uid = sample["requirement_uid"]
            name = sample["requirement_name"]

            # 이미 완료된 항목은 스킵
            if uid in completed_results:
                print(f"[{idx}/{len(samples)}] [완료됨 - 스킵]: {uid}")
                res_item = completed_results[uid]
                final_results.append(res_item)
                f_jsonl.write(json.dumps(res_item, ensure_ascii=False) + "\n")
                f_jsonl.flush()
                continue

            print(f"[{idx}/{len(samples)}] API 호출 중: {uid} ({name[:20]}...)")
            try:
                res_dict = run_single_labeling_with_fallback(sample)
                is_valid, errors = validate_label_output(res_dict)
                
                merged = sample.copy()
                merged["pilot_result"] = res_dict
                final_results.append(merged)
                f_jsonl.write(json.dumps(merged, ensure_ascii=False) + "\n")
                f_jsonl.flush()
                
                if not is_valid:
                    print(f"  [경고] 스키마 검증 경고: {errors}")
            except Exception as e:
                print(f"  [오류] {uid} 처리 실패: {e}")

            time.sleep(3.5)

    print("\n=== 최종 라벨링 파일럿 완료 ===")
    print(f"- 완료 건수: {len(final_results)}/{len(samples)}건")
    print(f"- JSONL 저장 완료: {output_jsonl}")

    # CSV요약 저장
    csv_fieldnames = [
        "requirement_uid",
        "document_id",
        "agency",
        "requirement_name",
        "sample_category",
        "primary_action",
        "confidence",
        "reasoning",
        "evidence",
        "is_missing_info",
        "domain_level",
        "cost_driver",
        "responsibility_risk"
    ]

    with open(output_csv, "w", encoding="utf-8-sig", newline="") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=csv_fieldnames)
        writer.writeheader()
        for res in final_results:
            pr = res.get("pilot_result", {})
            writer.writerow({
                "requirement_uid": res.get("requirement_uid"),
                "document_id": res.get("document_id"),
                "agency": res.get("agency"),
                "requirement_name": res.get("requirement_name"),
                "sample_category": res.get("sample_category"),
                "primary_action": pr.get("primary_action"),
                "confidence": pr.get("confidence"),
                "reasoning": pr.get("reasoning"),
                "evidence": " | ".join(pr.get("evidence", [])) if isinstance(pr.get("evidence"), list) else pr.get("evidence"),
                "is_missing_info": pr.get("missing_information", {}).get("is_missing") if isinstance(pr.get("missing_information"), dict) else "",
                "domain_level": pr.get("domain_dependency", {}).get("level") if isinstance(pr.get("domain_dependency"), dict) else "",
                "cost_driver": pr.get("risk_factors", {}).get("cost_driver") if isinstance(pr.get("risk_factors"), dict) else "",
                "responsibility_risk": pr.get("risk_factors", {}).get("responsibility_risk") if isinstance(pr.get("risk_factors"), dict) else ""
            })
    print(f"- CSV 요약 저장 완료: {output_csv}")


if __name__ == "__main__":
    main()
