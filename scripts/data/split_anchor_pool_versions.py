"""Finalize clean separation of anchor pools v1 (20 items), v2 (100 items), and v3 (192 items)."""

import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]


def main():
    p_v3 = ROOT / "data" / "anchors" / "anchor_pool_v3.jsonl"
    rows_192 = [json.loads(l) for l in p_v3.read_text(encoding="utf-8").splitlines() if l.strip()]

    # 1. v1: initial 20 items (Decision 22)
    p_v1 = ROOT / "data" / "anchors" / "anchor_pool_v1.jsonl"
    v1_uids = [
        "defense_intelligent_platform:QUR-001",
        "kexim_ai_platform:PMR-025",
        "korail_genai_isp_ismp:CSR-001",
        "korail_genai_isp_ismp:CSR-003",
        "ccrs_ai_platform:DAR-001",
        "ccrs_ai_platform:SFR-022",
        "kexim_ai_platform:SER-001",
        "mfds_drug_ai_review:SFR-001",
        "mfds_drug_ai_review:SFR-002",
        "kac_ai_work_platform:AIP-001",
        "koen_ai_infrastructure:CON-003",
        "incheon_airport_digital_work:CUR-CM-001",
        "ccrs_ai_platform:SFR-001",
        "ccrs_ai_platform:SFR-002",
        "genai_incident_response:SFR-001",
        "incheon_airport_digital_work:SFR-001",
        "kangwon_land_genai:ECR-001",
        "kangwon_land_genai:ECR-002",
        "mfds_drug_ai_review:SFR-003",
        "ccrs_ai_platform:SFR-003",
    ]
    rows_20 = [r for r in rows_192 if r["requirement_uid"] in v1_uids]
    for r in rows_20:
        r["pool_version"] = "anchor_pool_v1"
    with open(p_v1, "w", encoding="utf-8") as f:
        for r in rows_20:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 2. v2: 100 items (Decision 25)
    p_v2 = ROOT / "data" / "anchors" / "anchor_pool_v2.jsonl"
    rows_100 = [r for r in rows_192 if r.get("source_run") != "full_chunk1_fewshot"]
    for r in rows_100:
        r["pool_version"] = "anchor_pool_v2"
    with open(p_v2, "w", encoding="utf-8") as f:
        for r in rows_100:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 3. v3: 192 items (Decision 26)
    for r in rows_192:
        r["pool_version"] = "anchor_pool_v3"
    with open(p_v3, "w", encoding="utf-8") as f:
        for r in rows_192:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    from scripts.labeling.anchor_pool import load_anchor_pool

    _, m1 = load_anchor_pool(p_v1)
    _, m2 = load_anchor_pool(p_v2)
    _, m3 = load_anchor_pool(p_v3)

    print("=== 앵커 풀 3개 버전 분리 완료 ===")
    print(f"1. [anchor_pool_v1.jsonl]: {m1['reviewed_count']}건 (초기 기준 풀 — 결정 22)")
    print(f"2. [anchor_pool_v2.jsonl]: {m2['reviewed_count']}건 (10개 기관 층화 10% 대표 풀 — 결정 25)")
    print(f"3. [anchor_pool_v3.jsonl]: {m3['reviewed_count']}건 (Chunk 1 누적 확장 풀 192건 — 결정 26)")


if __name__ == "__main__":
    main()
