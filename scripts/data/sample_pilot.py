#!/usr/bin/env python3
"""
RFP 요구사항 동결 데이터셋 v0.2.0 라벨링 파일럿 표본 추출 스크립트

수행 작업:
1. 10개 문서 전체를 100% 커버하는 문서별 층화 추출
2. 정규화 요구사항 유형별(기능, 보안, 성능, 인프라 등) 균형 추출
3. 본문 길이 분포(단문/중문/장문) 및 특이케이스(중첩표, 승인 예외 ID) 포함
4. 약 40건의 라벨링 파일럿 표본 생성 (data/samples/labeling_pilot_sample_v0.1.0.jsonl / csv)
"""

import json
import csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Any


def normalize_requirement_type(raw_type: str) -> str:
    """원문 요구사항 유형 정규화"""
    if not raw_type or raw_type == "None" or raw_type is None:
        return "미지정 (None)"
    t = str(raw_type).strip()
    if any(k in t for k in ["기능", "SFR", "AI 활용 업무", "AI 기반 솔루션", "그룹웨어", "서비스"]):
        return "기능 요구사항"
    if "성능" in t or "PER" in t:
        return "성능 요구사항"
    if "보안" in t or "SER" in t:
        return "보안 요구사항"
    if "데이터" in t or "DAR" in t or "ECM" in t:
        return "데이터 요구사항"
    if "품질" in t or "QUR" in t:
        return "품질 요구사항"
    if "인터페이스" in t or "INR" in t:
        return "인터페이스 요구사항"
    if "제약" in t or "COR" in t:
        return "제약사항"
    if "테스트" in t or "TER" in t:
        return "테스트 요구사항"
    if any(k in t for k in ["장비", "인프라", "시스템", "AI 플랫폼 및 인프라"]):
        return "인프라·장비 요구사항"
    if any(k in t for k in ["프로젝트", "PMR", "PSR", "컨설팅", "CNR", "CUR", "거버넌스", "안전"]):
        return "프로젝트 관리·지원 요구사항"
    return "기타"


def load_dataset(filepath: str) -> List[Dict[str, Any]]:
    """jsonl 데이터셋 로드"""
    records = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def sample_pilot_dataset(records: List[Dict[str, Any]], target_size: int = 40) -> List[Dict[str, Any]]:
    """문서, 유형, 길이, 특이케이스를 반영한 파일럿 표본 층화 추출"""
    sampled_uids = set()
    sampled_records = []
    
    def add_sample(r: Dict[str, Any], category_label: str = ""):
        uid = r["requirement_uid"]
        if uid not in sampled_uids:
            sampled_uids.add(uid)
            rec = r.copy()
            rec["sample_category"] = category_label
            sampled_records.append(rec)

    # 1. 필수 특이케이스 1: 승인된 원문 예외 ID (인천공항 CUR-CM-001)
    for r in records:
        if r.get("requirement_id") != r.get("source_requirement_id"):
            add_sample(r, "승인예외_ID_불일치")

    # 2. 필수 특이케이스 2: 중첩표(' | ') 포함 사례 (문서별 3~4건)
    nested_table_recs = [r for r in records if " | " in r.get("raw_requirement_text", "")]
    for r in nested_table_recs[:4]:
        add_sample(r, "특이_중첩표포함")

    # 3. 본문 극단치 사례 (최단 본문 2건, 최장 본문 2건)
    sorted_by_len = sorted(records, key=lambda x: len(x.get("raw_requirement_text", "")))
    for r in sorted_by_len[:2]:
        add_sample(r, "극단치_최단본문")
    for r in sorted_by_len[-2:][::-1]:
        add_sample(r, "극단치_최장본문")

    # 4. 문서별 커버리지 보장 (10개 문서 각 최소 3건 이상 포함)
    docs = sorted(list(set(r.get("document_id") for r in records)))
    for doc_id in docs:
        doc_recs = [r for r in records if r.get("document_id") == doc_id]
        # 해당 문서에서 정규화 유형 다양하게 추출
        doc_sampled_count = sum(1 for r in sampled_records if r.get("document_id") == doc_id)
        if doc_sampled_count < 3:
            needed = 3 - doc_sampled_count
            for r in doc_recs:
                if r["requirement_uid"] not in sampled_uids:
                    norm_type = normalize_requirement_type(r.get("requirement_type", ""))
                    add_sample(r, f"문서표본_{doc_id}_{norm_type}")
                    needed -= 1
                    if needed <= 0:
                        break

    # 5. 목표 크기(약 40건)에 도달할 때까지 정규화 유형별 대표 사례 보충
    if len(sampled_records) < target_size:
        type_groups = {}
        for r in records:
            nt = normalize_requirement_type(r.get("requirement_type", ""))
            if nt not in type_groups:
                type_groups[nt] = []
            type_groups[nt].append(r)
            
        # 유형 순환하며 대표 사례 추가
        while len(sampled_records) < target_size:
            added_any = False
            for nt, r_list in type_groups.items():
                for r in r_list:
                    if r["requirement_uid"] not in sampled_uids:
                        add_sample(r, f"유형대표_{nt}")
                        added_any = True
                        break
                if len(sampled_records) >= target_size:
                    break
            if not added_any:
                break

    return sampled_records


def main():
    root_dir = Path(__file__).resolve().parent.parent
    dataset_path = root_dir / "data" / "processed" / "requirements_v0.2.0.jsonl"
    
    if not dataset_path.exists():
        print(f"오류: 데이터셋 파일을 찾을 수 없습니다 -> {dataset_path}")
        sys.exit(1)
        
    records = load_dataset(str(dataset_path))
    pilot_samples = sample_pilot_dataset(records, target_size=40)
    
    samples_dir = root_dir / "data" / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. JSONL 파일 저장
    jsonl_path = samples_dir / "labeling_pilot_sample_v0.1.0.jsonl"
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for r in pilot_samples:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"파일럿 표본 JSONL 생성 완료: {jsonl_path} ({len(pilot_samples)}건)")
    
    # 2. CSV 파일 저장 (사람 검수/확인용)
    csv_path = samples_dir / "labeling_pilot_sample_v0.1.0.csv"
    fieldnames = [
        "requirement_uid",
        "document_id",
        "agency",
        "requirement_id",
        "requirement_type",
        "requirement_name",
        "sample_category",
        "raw_requirement_text"
    ]
    with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for r in pilot_samples:
            writer.writerow(r)
    print(f"파일럿 표본 CSV 생성 완료: {csv_path}")


if __name__ == "__main__":
    main()
