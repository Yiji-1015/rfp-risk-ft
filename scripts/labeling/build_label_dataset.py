"""전수 라벨링 실행 결과를 균일한 스키마의 라벨 데이터셋으로 합친다.

세 실행 디렉터리에서 직접 읽는다. 기존 병합 파일(`labels_v0.2.0_fewshot_v5.jsonl`)은
동기 실행분만 요구사항 원문을 갖고 배치분은 라벨만 갖는 불균일 스키마라, 문서·유형
속성을 쓰려면 매번 조인해야 하고 실행 경로도 `input` 필드 유무로 추측해야 했다.

두 가지를 여기서 정리한다.

1. 고정 규칙(`derive_primary_action`) 위반을 보정한다. 모델이 보조 축과 어긋나는
   주 라벨을 낸 행이 있다. 원본은 `primary_action_model`에 남긴다.
2. 실행 경로와 출처 실행을 명시 필드로 기록한다. 추측할 필요가 없어진다.
3. 요구사항 유형을 정본 분류로 정규화한다. 원본 표기 60종은 문서와 얽혀 있어
   유형별 분석이 성립하지 않는다(docs/issues/001). 원본 표기는 그대로 남긴다.

원본 실행 결과는 건드리지 않는다. 이 스크립트는 언제든 다시 돌릴 수 있다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.data.preprocess_text import make_model_text
from scripts.labeling.label_schema import (
    LabelResult,
    SCHEMA_VERSION,
    derive_primary_action,
)
from scripts.labeling.requirement_taxonomy import normalize_requirement_type

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REQUIREMENTS = ROOT / "data" / "processed" / "requirements_v0.3.0.jsonl"
DATASET_VERSION = "label_dataset_v4"
DEFAULT_OUTPUT = ROOT / "data" / "labels" / f"{DATASET_VERSION}.jsonl"
RUNS_DIR = ROOT / "reports" / "current" / "claude_runs"

# 전수 1,024건을 만든 실행들. Chunk 1 재실행 배치를 취소해서 경로가 섞여 있다(결정 27).
SOURCE_RUNS: tuple[tuple[str, str], ...] = (
    ("full_requirements_v0.2.0_fewshot_v5", "동기"),
    ("batch_full_101_1024", "배치"),
    ("batch_retry_4", "배치"),
)

REQUIREMENT_FIELDS = (
    "document_id",
    "agency",
    "domain",
    "requirement_id",
    "requirement_type",
    "requirement_name",
    "raw_requirement_text",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"파일이 없습니다: {path}")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_rows(
    requirements: dict[str, dict[str, Any]],
    runs: tuple[tuple[str, str], ...] = SOURCE_RUNS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """균일 스키마 행과 규칙 보정 내역을 반환한다."""
    rows: list[dict[str, Any]] = []
    corrections: list[dict[str, Any]] = []
    seen: dict[str, str] = {}

    for run_name, execution_path in runs:
        results = read_jsonl(RUNS_DIR / run_name / "results.jsonl")
        for result in results:
            if result.get("status") != "ok":
                continue
            uid = result["requirement_uid"]
            if uid in seen:
                raise ValueError(
                    f"{uid}가 {seen[uid]}와 {run_name} 양쪽에 있습니다. "
                    "실행 범위가 겹치면 어느 라벨이 최종인지 정해지지 않습니다."
                )
            seen[uid] = run_name

            requirement = requirements.get(uid)
            if requirement is None:
                raise KeyError(f"요구사항 데이터셋에 없는 uid: {uid} ({run_name})")

            label = LabelResult.model_validate(result["label"])
            derived = derive_primary_action(label)
            corrected = derived != label.primary_action
            if corrected:
                corrections.append(
                    {
                        "requirement_uid": uid,
                        "source_run": run_name,
                        "model": label.primary_action,
                        "rule": derived,
                        "blockers": list(label.blockers),
                        "cost_basis": label.cost_basis,
                        "reasoning": label.reasoning,
                    }
                )

            canonical_type, type_source = normalize_requirement_type(
                requirement.get("requirement_type"), requirement["requirement_id"]
            )
            normalized_text = make_model_text(
                None, requirement["raw_requirement_text"], "normalized-list"
            )
            model_text = make_model_text(
                requirement["requirement_name"],
                requirement["raw_requirement_text"],
                "normalized-list",
            )

            rows.append(
                {
                    "requirement_uid": uid,
                    **{f: requirement[f] for f in REQUIREMENT_FIELDS},
                    "normalized_requirement_text": normalized_text,
                    "model_text": model_text,
                    "requirement_type_normalized": canonical_type,
                    "requirement_type_source": type_source,
                    "primary_action": derived,
                    "primary_action_model": label.primary_action,
                    "rule_corrected": corrected,
                    "blockers": list(label.blockers),
                    "cost_basis": label.cost_basis,
                    "domain_dependency": label.domain_dependency,
                    "build_difficulty": label.build_difficulty,
                    "reasoning": label.reasoning,
                    "execution_path": execution_path,
                    "source_run": run_name,
                    "schema_version": SCHEMA_VERSION,
                }
            )

    rows.sort(key=lambda r: r["requirement_uid"])
    return rows, corrections


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requirements", type=Path, default=DEFAULT_REQUIREMENTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--expect",
        type=int,
        default=1024,
        help="기대 건수. 다르면 실패한다. 0이면 검사하지 않는다.",
    )
    args = parser.parse_args()

    requirements = {r["requirement_uid"]: r for r in read_jsonl(args.requirements)}
    rows, corrections = build_rows(requirements)

    if args.expect and len(rows) != args.expect:
        raise SystemExit(
            f"건수가 {len(rows)}입니다. {args.expect}건을 기대했습니다. "
            "실행 디렉터리가 빠졌거나 실패 건이 남아 있습니다."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )

    manifest = {
        "dataset_version": DATASET_VERSION,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "row_count": len(rows),
        "schema_version": SCHEMA_VERSION,
        "requirements_source": {
            "path": str(
                args.requirements.relative_to(ROOT)
                if args.requirements.is_relative_to(ROOT)
                else args.requirements
            ),
            "sha256": sha256_of(args.requirements),
        },
        "source_runs": [
            {"name": name, "execution_path": path,
             "row_count": sum(1 for r in rows if r["source_run"] == name)}
            for name, path in SOURCE_RUNS
        ],
        "execution_path_counts": dict(Counter(r["execution_path"] for r in rows)),
        "requirement_type_counts": dict(
            Counter(r["requirement_type_normalized"] for r in rows)
        ),
        "requirement_type_source_counts": dict(
            Counter(r["requirement_type_source"] for r in rows)
        ),
        "primary_action_counts": dict(Counter(r["primary_action"] for r in rows)),
        "model_input": {
            "field": "model_text",
            "variant": "normalized-list",
            "composition": "requirement_name + newline + normalized_requirement_text",
        },
        "rule_corrections": corrections,
        "output_sha256": sha256_of(args.output),
    }
    manifest_path = args.output.with_name(args.output.stem + "_manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"라벨 데이터셋 {len(rows)}건 -> {args.output}")
    print(f"  실행 경로: {manifest['execution_path_counts']}")
    print(f"  주 라벨:   {manifest['primary_action_counts']}")
    print(f"  유형 근거: {manifest['requirement_type_source_counts']}")
    print(f"  규칙 보정: {len(corrections)}건")
    for c in corrections:
        print(f"    {c['requirement_uid']}: {c['model']} -> {c['rule']} "
              f"(blockers={c['blockers']}, cost_basis={c['cost_basis']})")
    print(f"  manifest:  {manifest_path}")


if __name__ == "__main__":
    main()
