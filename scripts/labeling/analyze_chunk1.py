"""Analyze results of Chunk 1 (requirements 1-100)."""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]

RUN_PATH = ROOT / "reports" / "current" / "claude_runs" / "full_requirements_v0.2.0_fewshot_v5" / "results.jsonl"


def get_category(req_id: str) -> str:
    prefix = req_id.split("-")[0].strip() if "-" in req_id else req_id[:3]
    if prefix in ["SFR", "FUN", "AIP", "AIF", "SYS"]:
        return "1_기능"
    elif prefix in ["ECR", "INF"]:
        return "2_인프라"
    elif prefix in ["SER", "SEC"]:
        return "3_보안"
    elif prefix in ["DAR", "DAT", "INR", "INT", "GW"]:
        return "4_데이터연계"
    else:
        return "5_관리품질제약"


def main():
    raw_rows = [json.loads(l) for l in RUN_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows_map = {}
    for r in raw_rows:
        if r.get("status") == "ok":
            rows_map[r["requirement_uid"]] = r
    rows = list(rows_map.values())

    print("=" * 65)
    print(f"Chunk 1 (1~100번) 라벨링 완료 분석 (총 {len(rows)}건)")
    print("=" * 65)

    dist = Counter(r["label"]["primary_action"] for r in rows)
    for lab in ["통상수용", "견적반영", "계약·질의검토"]:
        cnt = dist.get(lab, 0)
        print(f"{lab:<14s}: {cnt:2d}건 ({cnt/len(rows)*100:4.1f}%)")

    print("\n" + "=" * 65)
    print("카테고리별 라벨 분포")
    print("=" * 65)
    cat_dist = defaultdict(Counter)
    for r in rows:
        cat = get_category(r["input"]["requirement_id"])
        act = r["label"]["primary_action"]
        cat_dist[cat][act] += 1

    labels = ["통상수용", "견적반영", "계약·질의검토"]
    for cat, d in sorted(cat_dist.items()):
        tot = sum(d.values())
        print(
            f"{cat:<16s} (합계: {tot:2d}건) -> "
            f"통상: {d.get('통상수용', 0):2d} ({d.get('통상수용', 0)/tot*100:4.1f}%), "
            f"견적: {d.get('견적반영', 0):2d} ({d.get('견적반영', 0)/tot*100:4.1f}%), "
            f"계약: {d.get('계약·질의검토', 0):2d} ({d.get('계약·질의검토', 0)/tot*100:4.1f}%)"
        )

    print("\n" + "=" * 65)
    print("원가 발생 근거(cost_basis) 분포")
    print("=" * 65)
    cost_dist = Counter(r["label"]["cost_basis"] for r in rows)
    for cost, cnt in cost_dist.most_common():
        print(f"  {cost:<16s}: {cnt:2d}건 ({cnt/len(rows)*100:4.1f}%)")


if __name__ == "__main__":
    main()
