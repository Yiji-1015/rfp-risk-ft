"""Analyze repeated consistency between zero-shot and few-shot labeling runs."""

import json
import sys
from collections import Counter
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]

RUNS = {
    "zero_rep1": ROOT / "reports" / "current" / "claude_runs" / "pilot_v0.1.0_zeroshot_v5" / "results.jsonl",
    "zero_rep2": ROOT / "reports" / "current" / "claude_runs" / "pilot_v0.1.0_zeroshot_v5_rep2" / "results.jsonl",
    "zero_rep3": ROOT / "reports" / "current" / "claude_runs" / "pilot_v0.1.0_zeroshot_v5_rep3" / "results.jsonl",
    "few_rep1": ROOT / "reports" / "current" / "claude_runs" / "pilot_v0.1.0_fewshot_v5" / "results.jsonl",
    "few_rep2": ROOT / "reports" / "current" / "claude_runs" / "pilot_v0.1.0_fewshot_v5_rep2" / "results.jsonl",
    "few_rep3": ROOT / "reports" / "current" / "claude_runs" / "pilot_v0.1.0_fewshot_v5_rep3" / "results.jsonl",
}

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


def load_runs():
    data = {}
    for name, p in RUNS.items():
        records = {}
        with open(p, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("status") == "ok":
                    records[row["requirement_uid"]] = row
        data[name] = records
    return data


def main():
    data = load_runs()
    uids = list(data["zero_rep1"].keys())
    print(f"총 분석 요구사항 수: {len(uids)}건")

    print("\n" + "=" * 60)
    print("1. 3회 반복 일치율 (3/3 Complete Agreement)")
    print("=" * 60)
    fields = ["primary_action", "blockers", "cost_basis", "domain_dependency", "build_difficulty"]
    for field in fields:
        z_match = sum(
            1
            for u in uids
            if data["zero_rep1"][u]["label"][field]
            == data["zero_rep2"][u]["label"][field]
            == data["zero_rep3"][u]["label"][field]
        )
        f_match = sum(
            1
            for u in uids
            if data["few_rep1"][u]["label"][field]
            == data["few_rep2"][u]["label"][field]
            == data["few_rep3"][u]["label"][field]
        )
        print(
            f"{field:<20s} | Zero-shot: {z_match:2d}/40 ({z_match/40*100:5.1f}%) | "
            f"Few-shot: {f_match:2d}/40 ({f_match/40*100:5.1f}%)"
        )

    zero_maj = {
        u: Counter([data[f"zero_rep{i}"][u]["label"]["primary_action"] for i in (1, 2, 3)]).most_common(1)[0][0]
        for u in uids
    }
    few_maj = {
        u: Counter([data[f"few_rep{i}"][u]["label"]["primary_action"] for i in (1, 2, 3)]).most_common(1)[0][0]
        for u in uids
    }

    print("\n" + "=" * 60)
    print("2. 다수결(Majority Vote) 기준 라벨 분포")
    print("=" * 60)
    z_dist = Counter(zero_maj.values())
    f_dist = Counter(few_maj.values())
    for lab in LABELS:
        print(f"{lab:<12s} | Zero-shot: {z_dist.get(lab, 0):2d}건 ({z_dist.get(lab, 0)/40*100:5.1f}%) | Few-shot: {f_dist.get(lab, 0):2d}건 ({f_dist.get(lab, 0)/40*100:5.1f}%)")

    print("\n" + "=" * 60)
    print("3. 전이 행렬 (Zero 다수결 -> Few 다수결)")
    print("=" * 60)
    shift = {l1: {l2: 0 for l2 in LABELS} for l1 in LABELS}
    for u in uids:
        shift[zero_maj[u]][few_maj[u]] += 1

    header = f"{'Zero \\ Few':<14s}" + "".join(f"{l:>12s}" for l in LABELS)
    print(header)
    for l1 in LABELS:
        row = f"{l1:<14s}" + "".join(f"{shift[l1][l]:>12d}" for l in LABELS)
        print(row)

    print("\n" + "=" * 60)
    print("4. 판정 변동 항목 상세 분석 (Zero != Few 다수결)")
    print("=" * 60)
    shift_count = 0
    for u in uids:
        if zero_maj[u] != few_maj[u]:
            shift_count += 1
            z_acts = [data[f"zero_rep{i}"][u]["label"]["primary_action"] for i in (1, 2, 3)]
            f_acts = [data[f"few_rep{i}"][u]["label"]["primary_action"] for i in (1, 2, 3)]
            print(f"[{shift_count}] {u}")
            print(f"    Zero 3회: {z_acts} -> 다수결: {zero_maj[u]}")
            print(f"    Few  3회: {f_acts} -> 다수결: {few_maj[u]}")
            print(f"    Zero-shot 근거 (rep1): {data['zero_rep1'][u]['label']['reasoning']}")
            print(f"    Few-shot 근거 (rep1):  {data['few_rep1'][u]['label']['reasoning']}")
            print()

    print("=" * 60)
    print("5. 사람 확정 11건 벤치마크 일치율")
    print("=" * 60)
    for name in ["zero_rep1", "zero_rep2", "zero_rep3", "few_rep1", "few_rep2", "few_rep3"]:
        c = sum(1 for u, lab in HUMAN_11.items() if data[name][u]["label"]["primary_action"] == lab)
        print(f"{name:<12s}: {c:2d}/11 ({c/11*100:5.1f}%)")
    zm_h = sum(1 for u, lab in HUMAN_11.items() if zero_maj[u] == lab)
    fm_h = sum(1 for u, lab in HUMAN_11.items() if few_maj[u] == lab)
    print(f"Zero 다수결 대 사람: {zm_h:2d}/11 ({zm_h/11*100:5.1f}%)")
    print(f"Few  다수결 대 사람: {fm_h:2d}/11 ({fm_h/11*100:5.1f}%)")


if __name__ == "__main__":
    main()
