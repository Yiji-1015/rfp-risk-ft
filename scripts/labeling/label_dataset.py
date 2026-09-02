"""확정된 라벨 데이터셋을 감사와 함께 읽는다.

이 파일은 **동결 데이터셋**이다. 앵커 풀과 같은 이유로 고정한다(§11.13). 분석·실험이
서로 다른 라벨을 보고 있으면 결과를 비교할 수 없고, 조용히 바뀌면 눈치채기 어렵다.
그래서 읽을 때마다 SHA-256을 대조하고, 다르면 실패시킨다.

바꿔야 할 이유가 생기면 파일을 고치는 게 아니라 **새 버전을 만든다.** `FROZEN_SHA256`을
갱신하는 것은 데이터셋을 교체하겠다는 선언이고, 그 판단은 `docs/history/`에 남긴다.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from scripts.data.preprocess_text import apply_mask, make_model_text
from scripts.labeling.label_schema import (
    BLOCKER_TYPES,
    PRIMARY_ACTIONS,
    SCHEMA_VERSION,
)
from scripts.labeling.requirement_taxonomy import CANONICAL_TYPES, UNKNOWN

ROOT = Path(__file__).resolve().parents[2]

DATASET_VERSION = "label_dataset_v4"
DEFAULT_PATH = ROOT / "data" / "labels" / "label_dataset_v4.jsonl"
FROZEN_SHA256 = "f8c1eb25e31ea28dc11ed3eb51faaf6cbbe61f128959e857e122c8fea1167b79"
EXPECTED_ROWS = 1024

DATASET_SPECS = {
    "v3": {
        "dataset_version": "label_dataset_v3",
        "path": ROOT / "data" / "labels" / "label_dataset_v3.jsonl",
        "sha256": "cacd75695b01b2d6a51bc2933041488c375c4212e7d43bb7dd113321cb7c7684",
        "has_model_text": False,
    },
    "v4": {
        "dataset_version": DATASET_VERSION,
        "path": DEFAULT_PATH,
        "sha256": FROZEN_SHA256,
        "has_model_text": True,
    },
}
DEFAULT_DATASET_KEY = "v4"
DATASET_VERSION_ENV = "RFP_DATASET_VERSION"
TEXT_MASK_ENV = "RFP_TEXT_MASK"

COST_BASES = ("없음", "고급·전문인력", "장비·인프라", "라이선스", "외부인증", "외주·전문기관", "복합")
LEVELS = ("높음", "보통", "낮음")

REQUIRED_FIELDS = (
    "requirement_uid",
    "document_id",
    "requirement_id",
    "requirement_name",
    "raw_requirement_text",
    "primary_action",
    "primary_action_model",
    "rule_corrected",
    "blockers",
    "cost_basis",
    "domain_dependency",
    "build_difficulty",
    "reasoning",
    "execution_path",
    "source_run",
    "schema_version",
    "requirement_type_normalized",
    "requirement_type_source",
)

V4_REQUIRED_FIELDS = ("normalized_requirement_text", "model_text")

TYPE_SOURCES = ("text", "prefix", "none")

# 값이 비어 있어도 통과시키는 필드. 일부 원본 RFP에 해당 항목이 없다.
NULLABLE_FIELDS = ("agency", "domain", "requirement_type")


class LabelDatasetError(RuntimeError):
    """데이터셋이 감사를 통과하지 못했을 때 올린다."""


def _fail(message: str) -> None:
    raise LabelDatasetError(message)


def get_model_text(row: dict[str, Any]) -> str:
    """v4는 명시적 모델 입력을, 이전 버전은 원문을 반환한다.

    `RFP_TEXT_MASK`가 설정돼 있으면 전처리 ablation의 마스킹 규칙을 하나 적용한다.
    데이터셋 파일은 그대로 두고 읽는 시점에만 바꾸므로 동결과 SHA-256 대조는 유지된다.
    모든 평가 코드가 이 함수를 거치므로 스위치는 여기 하나만 둔다.
    """
    text = row["model_text"] if "model_text" in row else row["raw_requirement_text"]
    return apply_mask(text, os.getenv(TEXT_MASK_ENV))


def load_label_dataset(
    path: Path | None = None,
    *,
    version: str | None = None,
    verify_hash: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """데이터셋을 읽고 구조를 감사한다.

    :param verify_hash: SHA-256 대조 여부. 새 버전을 만드는 중이라 해시가 아직
        확정되지 않았을 때만 False로 둔다. 분석에서는 끄지 않는다.
    :returns: (행 목록, 메타데이터)
    """
    selected = version or (
        os.getenv(DATASET_VERSION_ENV, DEFAULT_DATASET_KEY)
        if path is None
        else DEFAULT_DATASET_KEY
    )
    if selected not in DATASET_SPECS:
        _fail(
            f"알 수 없는 데이터셋 버전 {selected!r}. "
            f"가능한 값: {', '.join(DATASET_SPECS)}"
        )
    spec = DATASET_SPECS[selected]
    path = path or spec["path"]
    if not path.exists():
        _fail(
            f"라벨 데이터셋이 없습니다: {path}\n"
            "  python -m scripts.labeling.build_label_dataset"
        )

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if verify_hash and digest != spec["sha256"]:
        _fail(
            f"데이터셋이 동결 상태와 다릅니다.\n"
            f"  기대: {spec['sha256']}\n"
            f"  실제: {digest}\n"
            "파일을 되돌리거나, 교체가 의도라면 새 버전을 만들고 FROZEN_SHA256을 갱신하세요."
        )

    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if len(rows) != EXPECTED_ROWS:
        _fail(f"{EXPECTED_ROWS}건을 기대했으나 {len(rows)}건입니다.")

    uids = [r.get("requirement_uid") for r in rows]
    if len(set(uids)) != len(uids):
        dupes = [u for u, n in Counter(uids).items() if n > 1]
        _fail(f"requirement_uid가 중복됩니다: {dupes[:5]}")

    keysets = {tuple(sorted(r)) for r in rows}
    if len(keysets) != 1:
        _fail(
            "행마다 키 집합이 다릅니다. 균일 스키마가 이 데이터셋의 존재 이유입니다."
        )

    for row in rows:
        uid = row.get("requirement_uid", "?")
        required_fields = REQUIRED_FIELDS + (
            V4_REQUIRED_FIELDS if spec["has_model_text"] else ()
        )
        for field in required_fields:
            if field not in row:
                _fail(f"{uid}: 필수 필드 누락 {field}")
            if field not in ("blockers", "rule_corrected") and not row[field]:
                _fail(f"{uid}: {field}가 비어 있습니다")
        if row["primary_action"] not in PRIMARY_ACTIONS:
            _fail(f"{uid}: 알 수 없는 primary_action {row['primary_action']!r}")
        if row["cost_basis"] not in COST_BASES:
            _fail(f"{uid}: 알 수 없는 cost_basis {row['cost_basis']!r}")
        for axis in ("domain_dependency", "build_difficulty"):
            if row[axis] not in LEVELS:
                _fail(f"{uid}: 알 수 없는 {axis} {row[axis]!r}")
        for blocker in row["blockers"]:
            if blocker not in BLOCKER_TYPES:
                _fail(f"{uid}: 알 수 없는 blocker {blocker!r}")
        if row["requirement_type_normalized"] not in (*CANONICAL_TYPES, UNKNOWN):
            _fail(
                f"{uid}: 알 수 없는 정본 유형 {row['requirement_type_normalized']!r}"
            )
        if row["requirement_type_source"] not in TYPE_SOURCES:
            _fail(f"{uid}: 알 수 없는 유형 근거 {row['requirement_type_source']!r}")
        if spec["has_model_text"]:
            expected_normalized = make_model_text(
                None, row["raw_requirement_text"], "normalized-list"
            )
            if row["normalized_requirement_text"] != expected_normalized:
                _fail(f"{uid}: normalized_requirement_text가 원문과 일치하지 않습니다")
            expected_model_text = make_model_text(
                row["requirement_name"], row["raw_requirement_text"], "normalized-list"
            )
            if row["model_text"] != expected_model_text:
                _fail(f"{uid}: model_text가 요구사항명과 전처리 본문에 일치하지 않습니다")
        if row["schema_version"] != SCHEMA_VERSION:
            _fail(
                f"{uid}: 스키마 버전이 {row['schema_version']}입니다. "
                f"코드는 {SCHEMA_VERSION}입니다."
            )

    meta = {
        "dataset_version": spec["dataset_version"],
        "model_text_field": "model_text" if spec["has_model_text"] else "raw_requirement_text",
        "path": str(path),
        "sha256": digest,
        "row_count": len(rows),
        "schema_version": SCHEMA_VERSION,
        "document_count": len({r["document_id"] for r in rows}),
        "label_counts": dict(Counter(r["primary_action"] for r in rows)),
        "execution_path_counts": dict(Counter(r["execution_path"] for r in rows)),
        "rule_corrected_count": sum(1 for r in rows if r["rule_corrected"]),
        "requirement_type_counts": dict(
            Counter(r["requirement_type_normalized"] for r in rows)
        ),
        "requirement_type_source_counts": dict(
            Counter(r["requirement_type_source"] for r in rows)
        ),
        # 정본에 실리지 못한 행. 새 표기가 들어오면 여기서 드러난다.
        "unmapped_type_count": sum(
            1 for r in rows if r["requirement_type_normalized"] == UNKNOWN
        ),
        "nullable_missing": {
            f: sum(1 for r in rows if not r.get(f)) for f in NULLABLE_FIELDS
        },
    }
    return rows, meta


if __name__ == "__main__":
    _, meta = load_label_dataset()
    print(json.dumps(meta, ensure_ascii=False, indent=2))
