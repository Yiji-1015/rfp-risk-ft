"""Sample 100 anchor candidates stratified by document and requirement category."""

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]


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
    random.seed(42)

    p_reqs = ROOT / "data" / "processed" / "requirements_v0.3.0.jsonl"
    reqs = [json.loads(l) for l in p_reqs.read_text(encoding="utf-8").splitlines() if l.strip()]

    by_doc = defaultdict(list)
    for r in reqs:
        by_doc[r["document_id"]].append(r)

    print(f"총 문서 수: {len(by_doc)}개")

    selected = []
    for doc_id, doc_reqs in sorted(by_doc.items()):
        doc_by_cat = defaultdict(list)
        for r in doc_reqs:
            cat = get_category(r["requirement_id"])
            doc_by_cat[cat].append(r)

        for cat in doc_by_cat:
            random.shuffle(doc_by_cat[cat])

        # Target allocation per document (total 10 items)
        targets = {"1_기능": 3, "2_인프라": 2, "3_보안": 2, "4_데이터연계": 1, "5_관리품질제약": 2}
        doc_selected = []

        for cat, n in targets.items():
            doc_selected.extend(doc_by_cat[cat][:n])
            doc_by_cat[cat] = doc_by_cat[cat][n:]

        if len(doc_selected) < 10:
            remainders = []
            for cat, r_list in doc_by_cat.items():
                remainders.extend(r_list)
            random.shuffle(remainders)
            doc_selected.extend(remainders[: 10 - len(doc_selected)])

        doc_selected = doc_selected[:10]
        selected.extend(doc_selected)

        cat_counts = dict(Counter(get_category(r["requirement_id"]) for r in doc_selected))
        print(f"문서 {doc_id:<32s}: {len(doc_selected)}건 표집 (카테고리: {cat_counts})")

    print(f"\n총 표집된 후보 건수: {len(selected)}건")
    overall_cats = Counter(get_category(r["requirement_id"]) for r in selected)
    for cat, cnt in sorted(overall_cats.items()):
        print(f"  {cat:<15s}: {cnt:2d}건")

    out_path = ROOT / "data" / "samples" / "anchor_pool_100_candidates_v0.1.0.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in selected:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n파일 저장 완료: {out_path}")


if __name__ == "__main__":
    main()
