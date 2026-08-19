"""Analyze distribution of Anchor Pool + Chunk 1."""

import json
import sys
from collections import Counter
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]


def main():
    # 1. Anchor pool
    p_pool = ROOT / "data" / "anchors" / "anchor_pool_v1.jsonl"
    anchors = [json.loads(l) for l in p_pool.read_text(encoding="utf-8").splitlines() if l.strip()]
    anchor_map = {r["requirement_uid"]: r["primary_action"] for r in anchors}

    # 2. Chunk 1
    p_chunk1 = ROOT / "reports" / "current" / "claude_runs" / "full_requirements_v0.2.0_fewshot_v5" / "results.jsonl"
    chunk1_raw = [json.loads(l) for l in p_chunk1.read_text(encoding="utf-8").splitlines() if l.strip()]
    chunk1_map = {
        r["requirement_uid"]: r["label"]["primary_action"]
        for r in chunk1_raw
        if r.get("status") == "ok"
    }

    overlap_uids = set(anchor_map.keys()) & set(chunk1_map.keys())

    labels = ["통상수용", "견적반영", "계약·질의검토"]
    dist_anchor = Counter(anchor_map.values())
    dist_chunk1 = Counter(chunk1_map.values())

    combined_unique = dict(anchor_map)
    combined_unique.update(chunk1_map)
    dist_unique = Counter(combined_unique.values())

    dist_sum = Counter()
    for l in labels:
        dist_sum[l] = dist_anchor[l] + dist_chunk1[l]

    print("=" * 75)
    print(f"{'구분':<20s}{'총 건수':<10s}" + "".join(f"{l:>14s}" for l in labels))
    print("=" * 75)

    print(
        f"{'1. 앵커 풀':<20s}{len(anchor_map):<10d}"
        + "".join(f"{dist_anchor[l]:>6d} ({dist_anchor[l]/len(anchor_map)*100:4.1f}%)" for l in labels)
    )
    print(
        f"{'2. Chunk 1 (1~100)':<20s}{len(chunk1_map):<10d}"
        + "".join(f"{dist_chunk1[l]:>6d} ({dist_chunk1[l]/len(chunk1_map)*100:4.1f}%)" for l in labels)
    )
    print("-" * 75)
    print(
        f"{'3. 고유 요구사항 합집합':<20s}{len(combined_unique):<10d}"
        + "".join(f"{dist_unique[l]:>6d} ({dist_unique[l]/len(combined_unique)*100:4.1f}%)" for l in labels)
    )
    print(
        f"{'4. 단순 합산':<20s}{len(anchor_map) + len(chunk1_map):<10d}"
        + "".join(f"{dist_sum[l]:>6d} ({dist_sum[l]/(len(anchor_map)+len(chunk1_map))*100:4.1f}%)" for l in labels)
    )
    print("=" * 75)
    print(f"\n* 앵커 풀과 Chunk 1 간 중복 요구사항 수: {len(overlap_uids)}건")


if __name__ == "__main__":
    main()
