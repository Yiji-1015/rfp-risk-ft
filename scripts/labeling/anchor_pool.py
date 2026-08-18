"""앵커 풀 로딩과 무결성 검증.

PROJECT_DIRECTION §8.3, §11.13에 따라 감사·버전 관리된 앵커만 프롬프트에 들어간다.
검토되지 않은 LLM 출력이 그대로 앵커로 재사용되면 초기 오류가 확산되므로,
`review_status`가 검토완료인 행만 사용하고 나머지는 로딩 단계에서 떨어뜨린다.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from scripts.labeling.label_schema import LabelResult

REQUIRED_FIELDS = (
    "requirement_uid",
    "document_id",
    "requirement_name",
    "raw_requirement_text",
    "primary_action",
    "reasoning",
    "review_status",
    "pool_version",
)

REVIEWED_STATUS = "검토완료"
KNOWN_REVIEW_STATUSES = {"후보", REVIEWED_STATUS, "제외"}

VALID_ACTIONS = set(LabelResult.model_fields["primary_action"].annotation.__args__)


class AnchorPoolError(ValueError):
    """앵커 풀 파일이 실험에 쓸 수 없는 상태일 때 발생한다."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def load_anchor_pool(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """앵커 풀 JSONL을 읽고 검토완료 앵커와 manifest용 메타데이터를 반환한다."""
    if not path.exists():
        raise AnchorPoolError(
            f"앵커 풀 파일이 없습니다: {path}\n"
            "dynamic few-shot 전략은 감사 완료된 앵커 풀을 먼저 요구합니다."
        )

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AnchorPoolError(f"{path}:{line_number} JSON 파싱 실패: {exc}") from exc

            missing = [field for field in REQUIRED_FIELDS if not item.get(field)]
            if missing:
                raise AnchorPoolError(f"{path}:{line_number} 필수 필드 누락: {missing}")
            if item["review_status"] not in KNOWN_REVIEW_STATUSES:
                raise AnchorPoolError(
                    f"{path}:{line_number} 알 수 없는 review_status: {item['review_status']}"
                )
            if item["primary_action"] not in VALID_ACTIONS:
                raise AnchorPoolError(
                    f"{path}:{line_number} 허용되지 않는 primary_action: {item['primary_action']}"
                )
            rows.append(item)

    if not rows:
        raise AnchorPoolError(f"앵커 풀이 비어 있습니다: {path}")

    versions = {row["pool_version"] for row in rows}
    if len(versions) > 1:
        raise AnchorPoolError(
            f"한 파일에 여러 pool_version이 섞여 있습니다: {sorted(versions)}"
        )

    reviewed = [row for row in rows if row["review_status"] == REVIEWED_STATUS]
    if not reviewed:
        raise AnchorPoolError(
            f"{path}에 검토완료 앵커가 없습니다. 사람 감사를 마친 뒤 사용하세요."
        )

    uids = [row["requirement_uid"] for row in reviewed]
    duplicates = sorted({uid for uid, count in Counter(uids).items() if count > 1})
    if duplicates:
        raise AnchorPoolError(f"중복 requirement_uid: {duplicates}")

    label_counts = Counter(row["primary_action"] for row in reviewed)
    missing_labels = sorted(VALID_ACTIONS - set(label_counts))

    metadata = {
        "path": str(path.resolve()),
        "pool_version": versions.pop(),
        "sha256": _sha256(path),
        "total_rows": len(rows),
        "reviewed_count": len(reviewed),
        "label_counts": dict(sorted(label_counts.items())),
        "document_count": len({row["document_id"] for row in reviewed}),
        "labels_without_anchor": missing_labels,
    }
    return reviewed, metadata
