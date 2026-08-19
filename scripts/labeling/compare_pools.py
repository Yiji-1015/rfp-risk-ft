"""Compare labeling results between anchor pool v1 (20 items) and updated pool (23 items)."""

import json
import sys
from collections import Counter
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]

RUN_FEW20 = ROOT / "reports" / "current" / "claude_runs" / "pilot_v0.1.0_fewshot_v5" / "results.jsonl"
RUN_FEW23 = ROOT / "reports" / "current" / "claude_runs" / "pilot_v0.1.0_fewshot_v5_pool23" / "results.jsonl"

ZERO_RUNS = [
    ROOT / "reports" / "current" / "claude_runs" / "pilot_v0.1.0_zeroshot_v5" / "results.jsonl",
    ROOT / "reports" / "current" / "claude_runs" / "pilot_v0.1.0_zeroshot_v5_rep2" / "results.jsonl",
    ROOT / "reports" / "current" / "claude_runs" / "pilot_v0.1.0_zeroshot_v5_rep3" / "results.jsonl",
]

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
    z_data = [
        {
            json.loads(l)["requirement_uid"]: json.loads(l)
            for l in p.read_text(encoding="utf-8").splitlines()
            if l.strip()
        }
        for p in ZERO_RUNS
    ]
    uids = list(z_data[0].keys())

    zero_maj = {
        u: Counter([d[u]["label"]["primary_action"] for d in z_data]).most_common(1)[0][0]
        for u in uids
    }

    few20_data = {
        json.loads(l)["requirement_uid"]: json.loads(l)
        for l in RUN_FEW20.read_text(encoding="utf-8").splitlines()
        if l.strip()
    }
    few23_data = {
        json.loads(l)["requirement_uid"]: json.loads(l)
        for l in RUN_FEW23.read_text(encoding="utf-8").splitlines()
        if l.strip()
    }

    print(f"총 분석 요구사항 수: {len(uids)}건\n")

    print("=" * 60)
    print("1. 라벨 분포 비교 (N=40)")
    print("=" * 60)
    z_dist = Counter(zero_maj.values())
    f20_dist = Counter(r["label"]["primary_action"] for r in few20_data.values())
    f23_dist = Counter(r["label"]["primary_action"] for r in few23_data.values())

    for lab in LABELS:
        print(
            f"{lab:<12s} | Zero-shot 다수결: {z_dist.get(lab, 0):2d}건 ({z_dist.get(lab, 0)/40*100:4.1f}%) | "
            f"Few-shot(20풀): {f20_dist.get(lab, 0):2d}건 ({f20_dist.get(lab, 0)/40*100:4.1f}%) | "
            f"Few-shot(23풀): {f23_dist.get(lab, 0):2d}건 ({f23_dist.get(lab, 0)/40*100:4.1f}%)"
        )

    print("\n" + "=" * 60)
    print("2. 전이 행렬 (v1 20개풀 -> v2 23개풀)")
    print("=" * 60)
    shift = {l1: {l2: 0 for l2 in LABELS} for l1 in LABELS}
    for u in uids:
        shift[few20_data[u]["label"]["primary_action"]][few23_data[u]["label"]["primary_action"]] += 1

    header = f"{'20풀 \\ 23풀':<14s}" + "".join(f"{l:>12s}" for l in LABELS)
    print(header)
    for l1 in LABELS:
        row = f"{l1:<14s}" + "".join(f"{shift[l1][l2]:>12d}" for l2 in LABELS)
        print(row)

    print("\n" + "=" * 60)
    print("3. 판정이 변경된 항목 상세 (20풀 != 23풀)")
    print("=" * 60)
    diff_count = 0
    for u in uids:
        l20 = few20_data[u]["label"]["primary_action"]
        l23 = few23_data[u]["label"]["primary_action"]
        if l20 != l23:
            diff_count += 1
            name = few20_data[u]["input"]["requirement_name"]
            print(f"[{diff_count}] {u} ({name})")
            print(f"    Zero-shot 다수결: {zero_maj[u]}")
            print(f"    v1 (20풀) 판정:   {l20}")
            print(f"       -> 근거: {few20_data[u]['label']['reasoning']}")
            print(f"    v2 (23풀) 판정:   {l23}")
            print(f"       -> 근거: {few23_data[u]['label']['reasoning']}")
            anchors23 = [
                f"{a['requirement_uid']}({a['primary_action']}, sim={a.get('similarity', 0):.2f})"
                for a in few23_data[u].get("anchors_used", [])
            ]
            print(f"    v2 주입 앵커: {', '.join(anchors23)}")
            print()

    print("=" * 60)
    print("4. 사람 확정 11건 벤치마크 일치율")
    print("=" * 60)
    h_z = sum(1 for u, lab in HUMAN_11.items() if zero_maj[u] == lab)
    h_20 = sum(1 for u, lab in HUMAN_11.items() if few20_data[u]["label"]["primary_action"] == lab)
    h_23 = sum(1 for u, lab in HUMAN_11.items() if few23_data[u]["label"]["primary_action"] == lab)
    print(f"Zero-shot 다수결 대 사람: {h_z:2d}/11 ({h_z/11*100:5.1f}%)")
    print(f"Few-shot (20풀)  대 사람: {h_20:2d}/11 ({h_20/11*100:5.1f}%)")
    print(f"Few-shot (23풀)  대 사람: {h_23:2d}/11 ({h_23/11*100:5.1f}%)")


if __name__ == "__main__":
    main()
