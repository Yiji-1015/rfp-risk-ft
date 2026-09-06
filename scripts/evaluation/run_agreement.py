"""같은 요구사항을 두 번 라벨링한 실행 사이의 일치율을 잰다.

라벨은 LLM이 만든다. 그래서 "이 라벨이 맞는가"를 직접 물을 수는 없고, 대신 **같은
입력에 같은 답을 주는가**를 물을 수 있다. 두 실행이 갈리는 비율은 라벨 품질의 상한을
알려준다 — 조건을 하나 바꿨을 때 라벨이 흔들린다면, 그 조건이 우리가 정답이라고
부르는 값을 만들고 있다는 뜻이다.

**혼자서는 아무 뜻도 없는 숫자다.** 실행 간 불일치에는 두 성분이 섞여 있다.

1. 바꾼 조건이 만든 차이 (앵커 풀, 인출 전략, 프롬프트 버전 …)
2. 조건을 하나도 안 바꿔도 나오는 표집 변동

그래서 이 모듈은 일치율과 함께 **Wilson 95% 신뢰구간**을 항상 돌려준다. 결정 23이
같은 조건 반복에서 잰 37/40(92.5%)과 구간이 겹치면, 관측된 차이를 바꾼 조건 탓이라고
말할 수 없다. 결정 34가 골드 11건을 우열 판정에서 물린 것과 같은 검사다.

정규 근사(`p ± 1.96·√(p(1-p)/n)`)가 아니라 Wilson을 쓰는 이유는 n이 100 근처이고
p가 0.9 근처라 정규 근사의 구간 상단이 1을 넘어가기 때문이다.

`anchor_jaccard`는 그 두 성분을 갈라 보려는 시도다. 두 실행이 같은 앵커를 인출한
건만 모으면 앵커 차이가 제거되므로, 거기 남는 불일치는 표집 변동에 가깝다. 층이
얇아지면 구간이 넓어져 결론이 안 나올 수 있고, **그 경우 "모른다"가 결과다.**
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

# 보조 축까지 함께 재는 이유는 주 라벨만 보면 안정성을 과대평가하기 때문이다.
# 주 라벨이 같아도 그 판정을 뒷받침하는 근거가 다르면 같은 라벨이라고 보기 어렵다.
LABEL_FIELDS = ("primary_action", "cost_basis", "domain_dependency", "build_difficulty")

PRIMARY_ACTIONS = ("통상수용", "견적반영", "계약·질의검토")

# 결정 23이 같은 조건 3회 반복에서 잰 few-shot 주 라벨 3/3 일치율.
# 조건을 바꾼 실행의 일치율은 이 값과 구간이 겹치는지로 판단한다.
REPEAT_BASELINE = (37, 40)

Z_95 = 1.96


def wilson_interval(successes: int, total: int, *, z: float = Z_95) -> tuple[float, float]:
    """이항 비율의 Wilson 점수 신뢰구간.

    `total`이 0이면 `(0.0, 1.0)`을 돌려준다 — 아무것도 모른다는 뜻을 구간으로
    표현한 것이고, 0/0을 0%로 적어 층이 나쁜 것처럼 보이게 하지 않기 위해서다.
    """
    if successes < 0 or total < 0 or successes > total:
        raise ValueError(f"0 <= successes <= total이어야 합니다: {successes}/{total}")
    if total == 0:
        return (0.0, 1.0)

    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return (max(0.0, center - half), min(1.0, center + half))


@dataclass(frozen=True)
class Agreement:
    """한 필드(또는 한 층)의 일치 건수와 그 구간.

    :param label: 무엇을 잰 것인지. 표에 그대로 찍힌다.
    :param agreed: 두 실행이 같은 값을 준 건수.
    :param total: 두 실행 모두 성공한 건수.
    """

    label: str
    agreed: int
    total: int

    @property
    def rate(self) -> float:
        return self.agreed / self.total if self.total else 0.0

    @property
    def interval(self) -> tuple[float, float]:
        return wilson_interval(self.agreed, self.total)

    def overlaps(self, other: "Agreement") -> bool:
        """두 구간이 겹치는가 — 겹치면 그 차이를 신호라고 부를 수 없다."""
        low, high = self.interval
        other_low, other_high = other.interval
        return low <= other_high and other_low <= high


def repeat_baseline_agreement() -> Agreement:
    """결정 23의 동일 조건 반복 기준선을 `Agreement`로 돌려준다."""
    agreed, total = REPEAT_BASELINE
    return Agreement("결정23 동일조건 반복", agreed, total)


@dataclass(frozen=True)
class Disagreement:
    """한 건이 어떻게 갈렸는가.

    :param anchor_jaccard: 두 실행이 인출한 앵커 집합의 자카드 지수. 1.0이면
        앵커가 같았는데도 갈렸다는 뜻이라 앵커로는 설명되지 않는다.
    """

    requirement_uid: str
    left: str
    right: str
    anchor_jaccard: float


@dataclass(frozen=True)
class RunComparison:
    """두 실행의 대조 결과.

    :param left_name: 왼쪽 실행 이름. 혼동표의 행이다.
    :param right_name: 오른쪽 실행 이름. 혼동표의 열이다.
    :param uids: 양쪽 모두 성공한 `requirement_uid`. 정렬되어 있다.
    :param agreements: 필드명 → `Agreement`. `blockers`는 집합 비교다.
    :param confusion: (왼쪽 주라벨, 오른쪽 주라벨) → 건수.
    :param disagreements: 주 라벨이 갈린 건들.
    :param anchor_jaccard: uid → 앵커 집합 자카드 지수.
    """

    left_name: str
    right_name: str
    uids: tuple[str, ...]
    agreements: dict[str, Agreement]
    confusion: dict[tuple[str, str], int]
    disagreements: tuple[Disagreement, ...]
    anchor_jaccard: dict[str, float]

    @property
    def primary(self) -> Agreement:
        return self.agreements["primary_action"]

    @property
    def identical_anchor_uids(self) -> tuple[str, ...]:
        """두 실행이 완전히 같은 앵커를 본 건 — 앵커 차이가 제거된 층이다."""
        return tuple(u for u in self.uids if self.anchor_jaccard[u] == 1.0)


def load_run_results(path: str | Path) -> dict[str, dict[str, Any]]:
    """`results.jsonl`에서 성공 행만 uid로 색인해 읽는다.

    실패 행을 조용히 빼는 것이 아니라 아예 넣지 않는다. 실패는 라벨 불일치가 아니라
    호출이 끝나지 않은 것이고, 둘을 한 분모에 담으면 일치율이 실패율에 오염된다.
    """
    results: dict[str, dict[str, Any]] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") == "ok":
                results[row["requirement_uid"]] = row
    return results


def anchor_uids(record: dict[str, Any]) -> frozenset[str]:
    """한 건의 프롬프트에 실린 앵커들의 uid."""
    return frozenset(a["requirement_uid"] for a in record.get("anchors_used", []))


def jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    """두 집합의 자카드 지수. 둘 다 비었으면 1.0(같다)으로 본다."""
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def same_document_anchor_count(results: dict[str, dict[str, Any]]) -> int:
    """타깃과 같은 문서에서 온 앵커의 총 개수.

    결정 10이 유일하게 차단하기로 한 것이 이것이다. 0이 아니면 그 실행은 자기 문서의
    정답을 예시로 보고 라벨을 만든 것이므로 대조 자체가 성립하지 않는다.
    """
    total = 0
    for uid, record in results.items():
        document = uid.split(":")[0]
        total += sum(1 for a in anchor_uids(record) if a.split(":")[0] == document)
    return total


def compare_runs(
    left: dict[str, dict[str, Any]],
    right: dict[str, dict[str, Any]],
    *,
    left_name: str = "left",
    right_name: str = "right",
    fields: Sequence[str] = LABEL_FIELDS,
) -> RunComparison:
    """양쪽 모두 성공한 건에 대해 필드별 일치율과 혼동표를 만든다.

    분모는 언제나 **교집합**이다. 한쪽에만 있는 건은 비교 대상이 아니다.
    """
    uids = tuple(sorted(set(left) & set(right)))
    if not uids:
        raise ValueError("두 실행에 공통으로 성공한 요구사항이 없습니다.")

    overlap = {u: jaccard(anchor_uids(left[u]), anchor_uids(right[u])) for u in uids}

    agreements: dict[str, Agreement] = {}
    for field in fields:
        agreed = sum(1 for u in uids if left[u]["label"][field] == right[u]["label"][field])
        agreements[field] = Agreement(field, agreed, len(uids))

    # blocker는 순서에 의미가 없는 목록이라 집합으로 비교한다.
    blocker_agreed = sum(
        1
        for u in uids
        if set(left[u]["label"]["blockers"]) == set(right[u]["label"]["blockers"])
    )
    agreements["blockers"] = Agreement("blockers", blocker_agreed, len(uids))

    confusion: dict[tuple[str, str], int] = {}
    disagreements: list[Disagreement] = []
    for u in uids:
        a = left[u]["label"]["primary_action"]
        b = right[u]["label"]["primary_action"]
        confusion[(a, b)] = confusion.get((a, b), 0) + 1
        if a != b:
            disagreements.append(Disagreement(u, a, b, overlap[u]))

    return RunComparison(
        left_name=left_name,
        right_name=right_name,
        uids=uids,
        agreements=agreements,
        confusion=confusion,
        disagreements=tuple(disagreements),
        anchor_jaccard=overlap,
    )


ANCHOR_STRATA = ("앵커 동일 (J=1)", "부분 겹침 (0<J<1)", "앵커 상이 (J=0)")


def stratify_by_anchor_overlap(
    comparison: RunComparison,
    left: dict[str, dict[str, Any]],
    right: dict[str, dict[str, Any]],
) -> list[Agreement]:
    """앵커 중복도로 층을 갈라 주 라벨 일치율을 잰다.

    앵커가 판정을 움직인다면 중복도가 높은 층의 일치율이 높아야 한다. 층이 얇으면
    구간이 넓어 방향만 보이고 유의하지 않을 수 있는데, 그것도 결과다.
    """
    buckets: dict[str, list[str]] = {name: [] for name in ANCHOR_STRATA}
    for u in comparison.uids:
        j = comparison.anchor_jaccard[u]
        if j == 1.0:
            key = ANCHOR_STRATA[0]
        elif j == 0.0:
            key = ANCHOR_STRATA[2]
        else:
            key = ANCHOR_STRATA[1]
        buckets[key].append(u)

    strata = []
    for name in ANCHOR_STRATA:
        members = buckets[name]
        agreed = sum(
            1
            for u in members
            if left[u]["label"]["primary_action"] == right[u]["label"]["primary_action"]
        )
        strata.append(Agreement(name, agreed, len(members)))
    return strata


def confusion_rows(
    comparison: RunComparison, labels: Iterable[str] = PRIMARY_ACTIONS
) -> list[dict[str, Any]]:
    """혼동표를 `DataFrame`에 바로 넣을 수 있는 행 목록으로 편다."""
    labels = list(labels)
    return [
        {"": row, **{col: comparison.confusion.get((row, col), 0) for col in labels}}
        for row in labels
    ]


def agreement_rows(agreements: Iterable[Agreement]) -> list[dict[str, Any]]:
    """`Agreement` 목록을 표 행으로 편다. 구간은 항상 함께 싣는다."""
    rows = []
    for a in agreements:
        low, high = a.interval
        rows.append(
            {
                "항목": a.label,
                "일치": f"{a.agreed}/{a.total}",
                "일치율": a.rate,
                "Wilson 95% 하한": low,
                "Wilson 95% 상한": high,
            }
        )
    return rows
