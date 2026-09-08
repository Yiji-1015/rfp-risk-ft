"""`label_dataset_v7` 조립 — v5 위에 v7 재라벨링 결과를 덮는다.

v7은 프롬프트 v5에 `[v7 보정]`(과다 판정 방지, 판정을 낮추는 방향만)을 붙인 것이라
`통상수용`은 다시 매기지 않았다(decisions-09 2026-09-08 11:30, 단조성은 50건 표본으로 확인).
그래서 v7 = v5의 통상수용 행 그대로 + 나머지 행은 `relabel_v7_*` 배치 결과로 교체.
배치에서 실패한 건은 v5 라벨을 유지하고 manifest에 적는다.

    python -m scripts.labeling.assemble_label_dataset_v7
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from scripts.labeling.label_schema import LabelResult, SCHEMA_VERSION, derive_primary_action

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data/labels/label_dataset_v5.jsonl"
OUT = ROOT / "data/labels/label_dataset_v7.jsonl"
RUNS = ("claude_batches/relabel_v7_nonnormal", "claude_batches/relabel_v7_retry")
PROMPT = ROOT / "notebooks/prompts/system_prompt_v7.txt"
POOL = ROOT / "data/anchors/anchor_pool_v3.jsonl"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()] if path.exists() else []


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    base = read_jsonl(BASE)
    results: dict[str, tuple[dict, str]] = {}
    for run in RUNS:
        for r in read_jsonl(ROOT / "reports/current" / run / "results.jsonl"):
            if r.get("status") == "ok":
                results[r["requirement_uid"]] = (r["label"], Path(run).name)
    targets = [r for r in base if r["primary_action"] != "통상수용"]
    missing = [r["requirement_uid"] for r in targets if r["requirement_uid"] not in results]
    rows, corrections, changed = [], [], Counter()
    for r in base:
        uid = r["requirement_uid"]
        if r["primary_action"] == "통상수용" or uid not in results:
            rows.append({**r, "source_run": r["source_run"] if uid not in results else r["source_run"],
                         "label_origin": "v5"})
            continue
        label_dict, run_name = results[uid]
        label = LabelResult.model_validate(label_dict)
        derived = derive_primary_action(label)
        if derived != label.primary_action:
            corrections.append({"requirement_uid": uid, "model": label.primary_action, "rule": derived,
                                "blockers": list(label.blockers), "cost_basis": label.cost_basis})
        changed[(r["primary_action"], derived)] += 1
        rows.append({**r, "primary_action": derived, "primary_action_model": label.primary_action,
                     "rule_corrected": derived != label.primary_action, "blockers": list(label.blockers),
                     "cost_basis": label.cost_basis, "domain_dependency": label.domain_dependency,
                     "build_difficulty": label.build_difficulty, "reasoning": label.reasoning,
                     "execution_path": "배치", "source_run": run_name, "schema_version": SCHEMA_VERSION,
                     "label_origin": "v7"})
    rows.sort(key=lambda x: x["requirement_uid"])
    OUT.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")
    manifest = {
        "dataset_version": "label_dataset_v7", "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "row_count": len(rows), "schema_version": SCHEMA_VERSION,
        "base": {"path": "data/labels/label_dataset_v5.jsonl", "sha256": sha(BASE),
                 "kept_rows": sum(1 for x in rows if x["label_origin"] == "v5")},
        "relabeled_rows": sum(1 for x in rows if x["label_origin"] == "v7"),
        "not_relabeled_non_normal": missing,
        "source_runs": [Path(r).name for r in RUNS],
        "labeling_conditions": {
            "prompt_version": "system_prompt_v7", "prompt_file": str(PROMPT.relative_to(ROOT)), "prompt_sha256": sha(PROMPT),
            "retrieval": "stratified", "anchor_pool": str(POOL.relative_to(ROOT)), "anchor_pool_sha256": sha(POOL),
            "anchors_per_request": 3, "hints": False,
            "note": "v5의 통상수용 행은 다시 매기지 않았다(v7 규칙은 판정을 낮추는 방향만). 통상수용 50건 표본에서 49건 유지 확인.",
        },
        "transitions_from_v5": {f"{a}→{b}": n for (a, b), n in sorted(changed.items(), key=lambda kv: -kv[1])},
        "primary_action_counts": dict(Counter(x["primary_action"] for x in rows)),
        "rule_corrections": corrections, "output_sha256": sha(OUT),
    }
    OUT.with_name(OUT.stem + "_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{len(rows)}건 -> {OUT}  sha256 {manifest['output_sha256']}")
    print("  재라벨", manifest["relabeled_rows"], "/ v5 유지", manifest["base"]["kept_rows"], "/ 미회수", missing)
    print("  분포", manifest["primary_action_counts"])
    print("  전이", manifest["transitions_from_v5"])


if __name__ == "__main__":
    main()
