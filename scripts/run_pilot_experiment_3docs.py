#!/usr/bin/env python3
"""
대표 RFP 3개 문서 대상 Zero-shot vs Dynamic Few-shot (Pure TF-IDF 앵커링) 비교 실험 스크립트

대상 문서 (총 259건):
- kac_ai_work_platform (한국공항공사 - 86건)
- incheon_airport_digital_work (인천공항 - 78건)
- ccrs_ai_platform (신용회복위원회 - 95건)

수행 단계:
1. data/processed/requirements_v0.2.0.jsonl 중 3개 문서 요구사항 로드
2. [실험 A] Zero-shot 라벨링 실행 (Resume 지원)
3. [실험 B] Pure TF-IDF Dynamic Few-shot 라벨링 실행 (동일 문서 앵커 자동 제외, Resume 지원)
4. 두 실험 결과 간 라벨 일치율 및 변동(Shift) 분석 보고서 생성 (reports/pilot_3docs_comparison_v0.1.0.md)
"""

import os
import sys
import json
import time
import csv
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Windows 콘솔 utf-8 인코딩 대응
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from dotenv import load_dotenv
import google.genai as genai
from google.genai import types

from anchor_retriever import PureTfidfAnchorRetriever
from validate_label_schema import validate_label_output

# 환경 변수 및 Gemini Client 초기화
root_dir = Path(__file__).resolve().parent.parent
load_dotenv(root_dir / ".env")

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("오류: .env 파일에 GEMINI_API_KEY가 설정되어 있지 않습니다.")
    sys.exit(1)

client = genai.Client(api_key=api_key)

# 대상 RFP 문서 3개 지정
TARGET_DOCS = [
    "kac_ai_work_platform",
    "incheon_airport_digital_work",
    "ccrs_ai_platform"
]

# System Prompt 명세
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

TARGET_MODEL = "gemini-3.5-flash-lite"


def call_gemini_api(user_prompt: str) -> Dict[str, Any]:
    """단일 통제 모델(gemini-3.5-flash-lite) 사용 및 고속 호출 최적화"""
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        response_mime_type="application/json",
        response_schema=RESPONSE_SCHEMA,
        temperature=0.1
    )

    last_err = None
    for attempt in range(10):
        try:
            response = client.models.generate_content(
                model=TARGET_MODEL,
                contents=user_prompt,
                config=config
            )
            return json.loads(response.text)
        except Exception as e:
            err_str = str(e)
            last_err = e
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                print(f"  [429 Quota Exceeded] {TARGET_MODEL} (시도 {attempt+1}/10) -> 12초 대기...")
                time.sleep(12)
            else:
                time.sleep(2)
    raise last_err if last_err else RuntimeError("모델 호출 실패")


def format_fewshot_prompt(target: Dict[str, Any], anchors: List[Tuple[Dict[str, Any], float]]) -> str:
    """Dynamic Few-shot 앵커 예시를 조합한 사용자 프롬프트 생성"""
    prompt_parts = []
    
    if anchors:
        prompt_parts.append("[참고 앵커 판단 사례 (타 기관 요구사항 유사 예시)]")
        for i, (anc, score) in enumerate(anchors, 1):
            anc_uid = anc.get("requirement_uid", "")
            anc_name = anc.get("requirement_name", "")
            anc_text = anc.get("raw_requirement_text", "")
            anc_label = anc.get("zero_shot_result", {}).get("primary_action", "통상수용")
            anc_reason = anc.get("zero_shot_result", {}).get("reasoning", "")
            
            prompt_parts.append(f"""사례 #{i} (유사 요구사항):
- 요구사항명: {anc_name}
- 요구사항 내용: {anc_text[:150]}...
- 수석 작성자 판단 조치: {anc_label}
- 판단 사유: {anc_reason}""")
        prompt_parts.append("\n" + "="*50 + "\n")

    prompt_parts.append(f"""[검토 대상 요구사항]
[요구사항 ID]: {target['requirement_uid']}
[요구사항명]: {target['requirement_name']}
[요구사항 내용]:
{target.get('raw_requirement_text', '')}""")

    return "\n\n".join(prompt_parts)


def run_experiment():
    dataset_file = root_dir / "data" / "processed" / "requirements_v0.1.0.jsonl"
    if not dataset_file.exists():
        print(f"오류: 데이터셋 파일이 존재하지 않습니다 -> {dataset_file}")
        sys.exit(1)

    # 1. 3개 RFP 대상 데이터 필터링
    records = []
    with open(dataset_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line.strip())
                if item.get("document_id") in TARGET_DOCS:
                    records.append(item)

    print(f"=== 3개 RFP 대상 데이터 로드 완료: 총 {len(records)}건 ===")
    for doc in TARGET_DOCS:
        c = sum(1 for r in records if r["document_id"] == doc)
        print(f"  - {doc}: {c}건")

    reports_dir = root_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    zeroshot_file = reports_dir / "experiment_3docs_zeroshot.jsonl"
    fewshot_file = reports_dir / "experiment_3docs_fewshot.jsonl"

    # ==========================================
    # Step 1: Zero-shot 라벨링
    # ==========================================
    zeroshot_results = {}
    if zeroshot_file.exists():
        with open(zeroshot_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line.strip())
                    zeroshot_results[item["requirement_uid"]] = item

    print(f"\n--- [1/2] Zero-shot 라벨링 진행 (완료: {len(zeroshot_results)}/{len(records)}건) ---")
    
    with open(zeroshot_file, "a", encoding="utf-8") as f_out:
        for idx, r in enumerate(records, 1):
            uid = r["requirement_uid"]
            if uid in zeroshot_results:
                continue

            print(f"[{idx}/{len(records)}] [Zero-shot] {uid} ({r['requirement_name'][:15]}...)")
            user_prompt = f"""[요구사항 ID]: {uid}
[요구사항명]: {r['requirement_name']}
[요구사항 내용]:
{r.get('raw_requirement_text', '')}"""

            try:
                res = call_gemini_api(user_prompt)
                res["requirement_uid"] = uid
                item = r.copy()
                item["zero_shot_result"] = res
                # Data Lineage 메타데이터 기록 (llm-wiki 지식 반영)
                item["lineage"] = {
                    "source_dataset": "requirements_v0.1.0.jsonl",
                    "persona_id": "persona_ai_it_proposal_writer_v1",
                    "experiment_type": "zero_shot",
                    "model_used": "gemini-2.5-flash"
                }
                zeroshot_results[uid] = item
                f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
                f_out.flush()
            except Exception as e:
                print(f"  [오류] {uid} Zero-shot 처리 실패: {e}")

            time.sleep(1.0)

    # ==========================================
    # Step 2: Pure TF-IDF Dynamic Few-shot 라벨링
    # ==========================================
    # 앵커 풀 = Zero-shot 완료된 전체 3개 RFP 259건 (동일 문서 자동 마스킹됨)
    anchor_pool = list(zeroshot_results.values())
    retriever = PureTfidfAnchorRetriever(anchor_pool)

    fewshot_results = {}
    if fewshot_file.exists():
        with open(fewshot_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line.strip())
                    fewshot_results[item["requirement_uid"]] = item

    print(f"\n--- [2/2] Dynamic Few-shot 라벨링 진행 (완료: {len(fewshot_results)}/{len(records)}건) ---")

    with open(fewshot_file, "a", encoding="utf-8") as f_out:
        for idx, r in enumerate(records, 1):
            uid = r["requirement_uid"]
            if uid in fewshot_results:
                continue

            print(f"[{idx}/{len(records)}] [Dynamic Few-shot] {uid} ({r['requirement_name'][:15]}...)")
            
            # Pure TF-IDF 앵커 Top-3 인출 (동일 문서 자동 제외)
            top_anchors = retriever.get_top_k_anchors(r, top_k=3)
            user_prompt = format_fewshot_prompt(r, top_anchors)

            try:
                res = call_gemini_api(user_prompt)
                res["requirement_uid"] = uid
                item = r.copy()
                item["zero_shot_result"] = zeroshot_results.get(uid, {}).get("zero_shot_result")
                item["few_shot_result"] = res
                item["anchors_used"] = [
                    {"uid": anc["requirement_uid"], "doc_id": anc["document_id"], "score": score}
                    for anc, score in top_anchors
                ]
                fewshot_results[uid] = item
                f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
                f_out.flush()
            except Exception as e:
                print(f"  [오류] {uid} Few-shot 처리 실패: {e}")

            time.sleep(1.0)

    # ==========================================
    # Step 3: 비교 분석 및 마크다운 보고서 생성
    # ==========================================
    generate_comparison_report(fewshot_results, reports_dir / "pilot_3docs_comparison_v0.1.0.md")


def generate_comparison_report(results: Dict[str, Any], report_path: Path):
    """Zero-shot vs Dynamic Few-shot 비교 분석 보고서 작성"""
    total = len(results)
    if total == 0:
        return

    agree_count = 0
    shift_matrix = {
        "통상수용": {"통상수용": 0, "견적반영": 0, "계약·질의검토": 0},
        "견적반영": {"통상수용": 0, "견적반영": 0, "계약·질의검토": 0},
        "계약·질의검토": {"통상수용": 0, "견적반영": 0, "계약·질의검토": 0}
    }

    shifts = []

    for uid, item in results.items():
        z_action = item.get("zero_shot_result", {}).get("primary_action")
        f_action = item.get("few_shot_result", {}).get("primary_action")
        
        if z_action and f_action:
            if z_action == f_action:
                agree_count += 1
            if z_action in shift_matrix and f_action in shift_matrix[z_action]:
                shift_matrix[z_action][f_action] += 1
            
            if z_action != f_action:
                shifts.append({
                    "uid": uid,
                    "doc_id": item.get("document_id"),
                    "name": item.get("requirement_name"),
                    "zero_action": z_action,
                    "few_action": f_action,
                    "zero_reason": item.get("zero_shot_result", {}).get("reasoning", ""),
                    "few_reason": item.get("few_shot_result", {}).get("reasoning", "")
                })

    agree_rate = (agree_count / total) * 100 if total > 0 else 0.0

    report_content = f"""# 3개 대표 RFP 라벨링 비교 분석 보고서 (Zero-shot vs Dynamic Few-shot)

- **대상 문서 (3개)**: `kac_ai_work_platform`, `incheon_airport_digital_work`, `ccrs_ai_platform`
- **총 요구사항 수**: {total} 건
- **라벨 일치 건수**: {agree_count} / {total} 건
- **전체 라벨 일치율 (Agreement Rate)**: **{agree_rate:.2f}%**

---

## 1. Zero-shot ↔ Dynamic Few-shot 라벨 전이 행렬 (Shift Matrix)

| Zero-shot \\ Few-shot | 통상수용 | 견적반영 | 계약·질의검토 | 합계 |
|---|---:|---:|---:|---:|
| **통상수용** | {shift_matrix['통상수용']['통상수용']} | {shift_matrix['통상수용']['견적반영']} | {shift_matrix['통상수용']['계약·질의검토']} | {sum(shift_matrix['통상수용'].values())} |
| **견적반영** | {shift_matrix['견적반영']['통상수용']} | {shift_matrix['견적반영']['견적반영']} | {shift_matrix['견적반영']['계약·질의검토']} | {sum(shift_matrix['견적반영'].values())} |
| **계약·질의검토** | {shift_matrix['계약·질의검토']['통상수용']} | {shift_matrix['계약·질의검토']['견적반영']} | {shift_matrix['계약·질의검토']['계약·질의검토']} | {sum(shift_matrix['계약·질의검토'].values())} |

---

## 2. 앵커링 도입 후 판정이 변동된 주요 사례 (Top Shift Cases)

총 **{len(shifts)}건**의 요구사항에서 앵커 추가 후 판단이 변경되었습니다.

"""
    for idx, s in enumerate(shifts[:10], 1):
        report_content += f"""### {idx}. [{s['doc_id']}] {s['uid']} - {s['name']}
- **Zero-shot 판정**: `{s['zero_action']}` (사유: {s['zero_reason']})
- **Dynamic Few-shot 판정**: `{s['few_action']}` (사유: {s['few_reason']})

---
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"\n==========================================")
    print(f"비교 분석 보고서 생성 완료: {report_path}")
    print(f"- 라벨 일치율: {agree_rate:.2f}% ({agree_count}/{total})")
    print(f"- 라벨 변동 건수: {len(shifts)}건")
    print(f"==========================================")


if __name__ == "__main__":
    run_experiment()
