"""Inspect distribution of requirement types and their corresponding labels."""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]

PREFIX_NAMES = {
    "SFR": "기능 요구사항 (System Feature / Functional)",
    "ECR": "인프라·장비 요구사항 (Equipment / Infrastructure)",
    "SER": "보안 요구사항 (Security Requirement)",
    "SEC": "보안 요구사항 (Security)",
    "INR": "인터페이스·연계 요구사항 (Interface)",
    "DAR": "데이터 요구사항 (Data Requirement)",
    "QUR": "품질 요구사항 (Quality Requirement)",
    "TER": "테스트 요구사항 (Test Requirement)",
    "PMR": "프로젝트 관리 요구사항 (Project Management)",
    "PSR": "프로젝트 지원·상주 요구사항 (Project Support)",
    "COR": "제약사항·법령 (Constraint / Compliance)",
    "CON": "컨설팅·전략수립 (Consulting)",
    "CSR": "컨설팅·전략과제 (Consulting Strategy)",
    "CUR": "공통/특수 요구사항 (Common/Custom)",
    "OTR": "기타·운영 요구사항 (Other/Operation)",
    "PER": "성능 요구사항 (Performance)",
}


def main():
    p_reqs = ROOT / "data" / "processed" / "requirements_v0.3.0.jsonl"
    reqs = [json.loads(l) for l in p_reqs.read_text(encoding="utf-8").splitlines() if l.strip()]

    print(f"=== 1. 전체 1,024건 요구사항 유형(ID Prefix) 분포 ===")
    type_counts = Counter()
    for r in reqs:
        req_id = r["requirement_id"]
        prefix = req_id.split("-")[0].strip() if "-" in req_id else req_id[:3]
        type_counts[prefix] += 1

    for prefix, count in type_counts.most_common():
        pname = PREFIX_NAMES.get(prefix, "기타")
        print(f"{prefix:<6s} ({pname:<35s}): {count:4d}건 ({count/len(reqs)*100:4.1f}%)")

    # 2. Gather all unique labeled items so far (from runs)
    labeled_data = {}
    pilot_runs = [
        ROOT / "reports" / "current" / "claude_runs" / "pilot_v0.1.0_fewshot_v5_pool23" / "results.jsonl",
        ROOT / "reports" / "current" / "claude_runs" / "standard_clause_v0.1.0_zeroshot_v5" / "results.jsonl",
        ROOT / "reports" / "current" / "claude_runs" / "anchor_cand_quote_rep1" / "results.jsonl",
        ROOT / "reports" / "current" / "claude_runs" / "anchor_cand_cq_rep2" / "results.jsonl",
        ROOT / "data" / "anchors" / "anchor_pool_v1.jsonl",
    ]

    for p in pilot_runs:
        if not p.exists():
            continue
        for l in p.read_text(encoding="utf-8").splitlines():
            if not l.strip():
                continue
            row = json.loads(l)
            uid = row["requirement_uid"]
            lab = row.get("primary_action") or row.get("label", {}).get("primary_action")
            if lab:
                labeled_data[uid] = lab

    print(f"\n현재까지 고유하게 라벨링된 요구사항 수: {len(labeled_data)}건")

    # 3. Label distribution by requirement prefix
    type_labels = defaultdict(Counter)
    req_lookup = {r["requirement_uid"]: r for r in reqs}

    for uid, lab in labeled_data.items():
        r = req_lookup.get(uid)
        if not r:
            continue
        req_id = r["requirement_id"]
        prefix = req_id.split("-")[0].strip() if "-" in req_id else req_id[:3]
        type_labels[prefix][lab] += 1

    print("\n" + "=" * 65)
    print("2. 현재까지 라벨링된 요구사항 유형별 라벨 분포 현황")
    print("=" * 65)
    labels_order = ["통상수용", "견적반영", "계약·질의검토"]
    header = f"{'유형(Prefix)':<14s}{'라벨링건수':<10s}" + "".join(f"{l:>12s}" for l in labels_order)
    print(header)
    for prefix, count in type_counts.most_common():
        if prefix in type_labels:
            c = type_labels[prefix]
            tot = sum(c.values())
            row = f"{prefix:<14s}{tot:<10d}" + "".join(f"{c.get(l, 0):>12d}" for l in labels_order)
            print(row)


if __name__ == "__main__":
    main()
