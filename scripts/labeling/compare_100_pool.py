"""Compare labeling results between 23-item pool and 100-item pool."""

import json
import sys
from collections import Counter
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]

RUN23_PATH = ROOT / "reports" / "current" / "claude_runs" / "pilot_v0.1.0_fewshot_v5_pool23" / "results.jsonl"
RUN100_PATH = ROOT / "reports" / "current" / "claude_runs" / "pilot_v0.1.0_fewshot_v5_pool100" / "results.jsonl"

HUMAN_11 = {
    "defense_intelligent_platform:QUR-001": "통상수용",
    "kexim_ai_platform:PMR-025": "통상수용",
    "korail_genai_isp_ismp:CSR-001": "통상수용",
    "korail_genai_isp_ismp:CSR-003": "통상수용",
    "ccrs_ai_platform:DAR-001": "통상수용",
    "ccrs_ai_platform:SFR-022": "견적반영",
    "kexim_ai_platform:SER-001": "견적반영",
    "mfds_drug_ai_review:SFR-001": "견적반영",
    "mfds_drug_ai_review:SFR-002": "통상수용",
    "kac_ai_work_platform:AIP-001": "계약·질의검토",
    "koen_ai_infrastructure:CON-003": "통상수용",
}

LABELS = ["통상수용", "견적반영", "계약·질의검토"]


def main():
    r23 = {
        json.loads(l)["requirement_uid"]: json.loads(l)
        for l in RUN23_PATH.read_text(encoding="utf-8").splitlines()
        if l.strip()
    }
    r100 = {
        json.loads(l)["requirement_uid"]: json.loads(l)
        for l in RUN100_PATH.read_text(encoding="utf-8").splitlines()
        if l.strip()
    }

    uids = list(r23.keys())

    # 1. Similarity comparison
    sim23 = [
        a["similarity"]
        for row in r23.values()
        for a in row.get("anchors_used", [])
        if a.get("similarity") is not None
    ]
    sim100 = [
        a["similarity"]
        for row in r100.values()
        for a in row.get("anchors_used", [])
        if a.get("similarity") is not None
    ]

    avg_sim23 = sum(sim23) / len(sim23) if sim23 else 0
    avg_sim100 = sum(sim100) / len(sim100) if sim100 else 0
    max_sim23 = max(sim23) if sim23 else 0
    max_sim100 = max(sim100) if sim100 else 0

    print("=" * 65)
    print("1. 앵커 인출 유사도(Cosine Similarity) 비교")
    print("=" * 65)
    print(f"23건 풀 평균 유사도 : {avg_sim23:.4f} (최고: {max_sim23:.4f})")
    print(f"100건 풀 평균 유사도: {avg_sim100:.4f} (최고: {max_sim100:.4f}) -> +{(avg_sim100-avg_sim23)/avg_sim23*100:.1f}% 대폭 상승!")

    # 2. Label distribution
    d23 = Counter(r["label"]["primary_action"] for r in r23.values())
    d100 = Counter(r["label"]["primary_action"] for r in r100.values())

    print("\n" + "=" * 65)
    print("2. 파일럿(N=40) 라벨 분포 비교")
    print("=" * 65)
    for l in LABELS:
        print(
            f"{l:<14s} | 23건 풀: {d23.get(l, 0):2d}건 ({d23.get(l, 0)/40*100:4.1f}%) | "
            f"100건 풀: {d100.get(l, 0):2d}건 ({d100.get(l, 0)/40*100:4.1f}%)"
        )

    # 3. Human 11 benchmark
    h_23 = sum(1 for u, lab in HUMAN_11.items() if r23[u]["label"]["primary_action"] == lab)
    h_100 = sum(1 for u, lab in HUMAN_11.items() if r100[u]["label"]["primary_action"] == lab)
    print("\n" + "=" * 65)
    print("3. 사람 확정 11건 골드 라벨 일치율")
    print("=" * 65)
    print(f"23건 풀  일치율: {h_23:2d}/11 ({h_23/11*100:4.1f}%)")
    print(f"100건 풀 일치율: {h_100:2d}/11 ({h_100/11*100:4.1f}%)")

    # 4. Shifts
    print("\n" + "=" * 65)
    print("4. 23건 풀 대비 판정 변동 항목 (23풀 != 100풀)")
    print("=" * 65)
    diff_count = 0
    for u in uids:
        l23 = r23[u]["label"]["primary_action"]
        l100 = r100[u]["label"]["primary_action"]
        if l23 != l100:
            diff_count += 1
            name = r23[u]["input"]["requirement_name"]
            print(f"[{diff_count}] {u} ({name})")
            print(f"    23풀 판정:   {l23}")
            print(f"    100풀 판정:  {l100}")
            print(f"    100풀 근거:  {r100[u]['label']['reasoning']}")
            anchors100 = [
                f"{a['requirement_uid']}({a['primary_action']}, sim={a.get('similarity', 0):.2f})"
                for a in r100[u].get("anchors_used", [])
            ]
            print(f"    100풀 주입 앵커: {', '.join(anchors100)}")
            print()
    if diff_count == 0:
        print("판정 변동 없음 (100% 동일 유지)")


if __name__ == "__main__":
    main()
