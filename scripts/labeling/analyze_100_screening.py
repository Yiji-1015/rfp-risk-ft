"""Analyze 100-candidate 3-rep screening results and report consensus metrics."""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]

RUNS = [
    ROOT / "reports" / "current" / "claude_runs" / "anchor_pool_100_rep1" / "results.jsonl",
    ROOT / "reports" / "current" / "claude_runs" / "anchor_pool_100_rep2" / "results.jsonl",
    ROOT / "reports" / "current" / "claude_runs" / "anchor_pool_100_rep3" / "results.jsonl",
]

LABELS = ["통상수용", "견적반영", "계약·질의검토"]


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
    data = []
    for p in RUNS:
        m = {}
        for l in p.read_text(encoding="utf-8").splitlines():
            if not l.strip():
                continue
            r = json.loads(l)
            if r.get("status") == "ok":
                m[r["requirement_uid"]] = r
        data.append(m)

    uids = list(data[0].keys())
    print(f"총 스크리닝 대상 건수: {len(uids)}건\n")

    consensus_items = []
    unstable_items = []

    for uid in uids:
        r1, r2, r3 = data[0][uid], data[1][uid], data[2][uid]
        act1 = r1["label"]["primary_action"]
        act2 = r2["label"]["primary_action"]
        act3 = r3["label"]["primary_action"]

        if act1 == act2 == act3:
            consensus_items.append((uid, act1, r1))
        else:
            unstable_items.append((uid, [act1, act2, act3], r1))

    print("=" * 65)
    print("1. 3회 반복 일관성(Consensus) 통과율")
    print("=" * 65)
    print(f"3/3 만장일치 일관성 통과: {len(consensus_items):2d}/100 ({len(consensus_items)/100*100:4.1f}%)")
    print(f"일관성 불일치 (흔들림)  : {len(unstable_items):2d}/100 ({len(unstable_items)/100*100:4.1f}%)")

    print("\n" + "=" * 65)
    print(f"2. 만장일치 통과 {len(consensus_items)}건의 라벨 분포")
    print("=" * 65)
    consensus_dist = Counter(act for _, act, _ in consensus_items)
    for lab in LABELS:
        cnt = consensus_dist.get(lab, 0)
        print(f"{lab:<14s}: {cnt:2d}건 ({cnt/len(consensus_items)*100:4.1f}%)")

    print("\n" + "=" * 65)
    print(f"3. 카테고리별 라벨 분포 (만장일치 {len(consensus_items)}건)")
    print("=" * 65)
    cat_dist = defaultdict(Counter)
    for uid, act, r in consensus_items:
        cat = get_category(r["input"]["requirement_id"])
        cat_dist[cat][act] += 1

    header = f"{'카테고리':<16s}{'합계':<8s}" + "".join(f"{l:>12s}" for l in LABELS)
    print(header)
    for cat, dist in sorted(cat_dist.items()):
        tot = sum(dist.values())
        row = f"{cat:<16s}{tot:<8d}" + "".join(f"{dist.get(l, 0):>12d}" for l in LABELS)
        print(row)

    print("\n" + "=" * 65)
    print(f"4. 10개 기관 문서별 만장일치 건수 (총 {len(consensus_items)}건)")
    print("=" * 65)
    doc_dist = defaultdict(Counter)
    for uid, act, r in consensus_items:
        doc = r["input"]["document_id"]
        doc_dist[doc][act] += 1

    for doc, dist in sorted(doc_dist.items()):
        tot = sum(dist.values())
        print(
            f"  {doc:<32s}: {tot:2d}건 (통상: {dist.get('통상수용', 0):2d}, "
            f"견적: {dist.get('견적반영', 0):2d}, 계약: {dist.get('계약·질의검토', 0):2d})"
        )


if __name__ == "__main__":
    main()
