"""문서 단위 leave-one-document-out(LODO) fold를 만들고 fold별 난이도를 진단한다.

행을 무작위로 섞지 않는다. 기관이 달라도 보안·품질·유지보수 문구가 반복되므로
무작위 분할은 같은 RFP의 비슷한 요구사항을 학습과 평가 양쪽에 갈라 넣는다(§10.1).

문서가 10개뿐이라 8:2 한 번으로는 결과가 "어느 2개가 걸렸는가"에 좌우된다. fold
기준선이 34.3%(defense)에서 79.2%(koen)까지 벌어져 있어, 쉬운 두 문서가 평가에
걸리면 같은 모델이 두 배 가까이 높은 점수를 낸다. LODO는 **모든 문서가 정확히 한 번씩
처음 보는 문서가 되게** 해서 §8.1의 다섯 번째 연구질문(처음 보는 기관과 도메인에도
일반화되는가)에 직접 답한다.

## 세 갈래 분할

파인튜닝은 early stopping과 best-checkpoint 선택을 요구한다(§9.4). 평가 문서를 보고
멈출 시점을 고르면 그 문서는 더 이상 처음 보는 문서가 아니므로, 평가와 별개의 검증
문서가 필요하다. 그래서 fold마다 셋으로 나눈다.

    학습 8문서 / 검증 1문서 / 평가 1문서

검증 문서는 **회전**으로 정한다. fold i의 평가가 docs[i]이면 검증은 docs[i+1]이다.
고정된 한 문서를 계속 검증에 쓰면 그 문서의 난이도가 모든 fold의 조기 종료 시점에
같은 방향으로 스며든다. 회전시키면 모든 문서가 검증도 정확히 한 번씩 맡아 그 편향이
fold 사이에서 상쇄된다. 무작위가 아니라 회전이므로 seed 없이 재현된다.

## 모든 모델이 같은 8문서로 학습한다

Dummy나 TF-IDF는 early stopping이 필요 없어 9문서를 다 쓸 수 있다. 그래도 기본은
`fit_documents`(8문서)다. §9.3의 통제 비교는 **분할을 고정하고 모델만 바꾸는** 것인데,
TF-IDF가 9문서를 보고 인코더가 8문서를 보면 두 결과의 차이에 학습량 차이가 섞인다.
"복잡하게 만든 값을 했는가"를 물으려면 본 데이터가 같아야 한다.

9문서를 다 쓰고 싶으면 `Fold.train_documents`가 그대로 있다. 다만 그렇게 쓴 결과는
파인튜닝과 나란히 놓지 않는다.

## 기준선이 둘인 이유

`trained_majority_accuracy`는 학습 집합의 최빈 클래스를 그대로 찍는 모델이다.
§9.2의 DummyClassifier이고 실제로 배포할 수 있다. 이 데이터에서는 어느 fold에서나
`통상수용`을 찍는다.

`oracle_majority_accuracy`는 **평가 문서 안에서의** 최빈 클래스 비율이다. 정답을 미리
본 값이라 배포할 수 없고, 모델이 아니라 **그 fold가 얼마나 쉬운가**를 재는 눈금이다.
issues/003이 "fold별 다수 클래스 기준선을 병기하라"고 한 것이 이쪽이다.

둘은 같지 않다. 문서마다 최빈 클래스가 달라서, defense·ccrs·genai에서는 `견적반영`이,
kangwon에서는 `계약·질의검토`가 최빈이다. 그 fold에서는 배포 가능한 Dummy가 oracle보다
낮게 나온다. 두 값을 섞어 하나로 보고하면 모델이 기준선을 넘었는지 판단이 흐려진다.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from scripts.evaluation.duplication import (
    DEFAULT_THRESHOLD,
    cross_document_similarity,
)


@dataclass(frozen=True)
class Fold:
    """LODO fold 하나.

    :param index: 0부터. 정렬된 문서 순서를 따른다.
    :param test_document: 이 fold에서 처음 보는 문서
    :param validation_document: early stopping용. 학습 문서 중 하나다
    :param fit_documents: 실제로 학습에 쓰는 8문서
    :param train_documents: 평가 문서를 뺀 9문서 (검증 문서 포함)
    """

    index: int
    test_document: str
    validation_document: str
    fit_documents: tuple[str, ...]
    train_documents: tuple[str, ...]

    def split(
        self, rows: Sequence[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        """행 목록을 (학습, 검증, 평가)로 가른다."""
        return (
            [r for r in rows if r["document_id"] in self.fit_documents],
            [r for r in rows if r["document_id"] == self.validation_document],
            [r for r in rows if r["document_id"] == self.test_document],
        )


@dataclass(frozen=True)
class FoldDiagnostics:
    """fold 하나의 난이도와 점수 구성.

    학습 전에 계산할 수 있는 값만 담는다. 모델 성능이 아니라 **그 성능을 어떻게 읽어야
    하는지**를 알려주는 값들이다.
    """

    fold: Fold
    test_size: int
    fit_size: int
    validation_size: int
    test_label_counts: dict[str, int]
    trained_majority_label: str
    trained_majority_accuracy: float
    oracle_majority_accuracy: float
    repeat_exposure_rate: float
    repeat_threshold: float
    rare_value_coverage: dict[str, dict[str, int]] = field(default_factory=dict)


def make_lodo_folds(rows: Sequence[dict[str, Any]]) -> list[Fold]:
    """문서 하나씩을 평가로 돌리는 fold 목록을 만든다.

    문서 이름을 정렬해 순서를 고정하므로 입력 행의 순서가 바뀌어도 같은 fold가 나온다.
    """
    documents = sorted({r["document_id"] for r in rows})
    if len(documents) < 3:
        raise ValueError(
            f"문서가 {len(documents)}개입니다. "
            "학습·검증·평가로 가르려면 최소 3개가 필요합니다."
        )

    folds: list[Fold] = []
    for i, test_document in enumerate(documents):
        # 회전. 평가 문서 바로 다음 문서를 검증으로 쓴다.
        validation_document = documents[(i + 1) % len(documents)]
        train_documents = tuple(d for d in documents if d != test_document)
        fit_documents = tuple(d for d in train_documents if d != validation_document)
        folds.append(
            Fold(
                index=i,
                test_document=test_document,
                validation_document=validation_document,
                fit_documents=fit_documents,
                train_documents=train_documents,
            )
        )
    return folds


def _majority(labels: Iterable[str]) -> tuple[str, int]:
    counts = Counter(labels)
    label, n = counts.most_common(1)[0]
    return label, n


def _repeat_flags(
    rows: Sequence[dict[str, Any]], threshold: float
) -> dict[str, bool]:
    """uid -> 반복 문구 여부.

    LODO에서 학습 집합은 곧 "자기 문서를 뺀 나머지 전부"이므로, 문서 간 최근접
    유사도를 한 번 구하면 모든 fold에 그대로 쓸 수 있다.
    """
    result = cross_document_similarity(rows, threshold=threshold)
    return {
        row["requirement_uid"]: bool(flag)
        for row, flag in zip(rows, result.is_repeat)
    }


def diagnose_fold(
    fold: Fold,
    rows: Sequence[dict[str, Any]],
    *,
    label_field: str = "primary_action",
    repeat_threshold: float = DEFAULT_THRESHOLD,
    rare_value_fields: Sequence[str] = ("cost_basis",),
    repeat_flags: dict[str, bool] | None = None,
) -> FoldDiagnostics:
    """fold 하나의 난이도 지표를 계산한다.

    :param repeat_flags: uid -> 반복 문구 여부. 미리 계산한 것을 넘기면 재사용한다.
        fold마다 다시 계산하면 같은 값을 열 번 구하게 된다.
    """
    fit_rows, validation_rows, test_rows = fold.split(rows)
    if not test_rows:
        raise ValueError(f"{fold.test_document}에 해당하는 행이 없습니다.")

    trained_label, _ = _majority(r[label_field] for r in fit_rows)
    trained_hits = sum(1 for r in test_rows if r[label_field] == trained_label)
    _, oracle_hits = _majority(r[label_field] for r in test_rows)

    if repeat_flags is None:
        repeat_flags = _repeat_flags(rows, repeat_threshold)
    exposed = sum(1 for r in test_rows if repeat_flags[r["requirement_uid"]])

    # 희소 값이 학습과 평가 양쪽에 얼마나 있는가. 한쪽이 0이면 그 값은 이 fold에서
    # 배울 수 없거나 잴 수 없다.
    coverage: dict[str, dict[str, int]] = {}
    for field_name in rare_value_fields:
        for value in sorted({r[field_name] for r in rows}):
            in_fit = sum(1 for r in fit_rows if r[field_name] == value)
            in_test = sum(1 for r in test_rows if r[field_name] == value)
            if in_fit == 0 or in_test == 0:
                coverage.setdefault(field_name, {})[value] = in_test

    return FoldDiagnostics(
        fold=fold,
        test_size=len(test_rows),
        fit_size=len(fit_rows),
        validation_size=len(validation_rows),
        test_label_counts=dict(Counter(r[label_field] for r in test_rows)),
        trained_majority_label=trained_label,
        trained_majority_accuracy=trained_hits / len(test_rows),
        oracle_majority_accuracy=oracle_hits / len(test_rows),
        repeat_exposure_rate=exposed / len(test_rows),
        repeat_threshold=repeat_threshold,
        rare_value_coverage=coverage,
    )


def diagnose_all(
    rows: Sequence[dict[str, Any]],
    *,
    label_field: str = "primary_action",
    repeat_threshold: float = DEFAULT_THRESHOLD,
) -> list[FoldDiagnostics]:
    """모든 fold를 진단한다. 반복 문구 계산은 한 번만 한다."""
    repeat_flags = _repeat_flags(rows, repeat_threshold)
    return [
        diagnose_fold(
            fold,
            rows,
            label_field=label_field,
            repeat_threshold=repeat_threshold,
            repeat_flags=repeat_flags,
        )
        for fold in make_lodo_folds(rows)
    ]


def aggregate(values: Sequence[float], sizes: Sequence[int]) -> dict[str, float]:
    """fold 점수를 두 방식으로 집계한다.

    **둘 다 보고한다.** 하나만 쓰면 어느 쪽이든 오해를 만든다(issues/003).

    - `fold_mean`: 49건짜리 fold가 192건짜리와 같은 무게를 갖는다. 문서 단위 일반화를
      보는 관점이지만 작은 fold의 분산이 그대로 실린다.
    - `count_weighted`: 건수로 가중한다. 큰 문서 셋(mfds·defense·kexim, 전체의 48.6%)이
      결과를 지배한다.
    """
    if len(values) != len(sizes):
        raise ValueError("값과 건수의 길이가 다릅니다.")
    if not values:
        raise ValueError("집계할 fold가 없습니다.")
    total = sum(sizes)
    return {
        "fold_mean": sum(values) / len(values),
        "count_weighted": sum(v * n for v, n in zip(values, sizes)) / total,
    }


def _main() -> None:
    from scripts.labeling.label_dataset import load_label_dataset

    rows, meta = load_label_dataset()
    diagnostics = diagnose_all(rows)

    print(
        f"데이터셋 {meta['dataset_version']} / {meta['row_count']}건 / "
        f"문서 {meta['document_count']}개"
    )
    print(f"sha256   {meta['sha256'][:16]}...")
    print(f"분할     LODO {len(diagnostics)}겹 (학습 8 / 검증 1 / 평가 1 문서)")
    print(f"반복 임계값 {diagnostics[0].repeat_threshold}")
    print()

    header = (
        f"{'#':>2} {'평가 문서':<30}{'건수':>5}{'Dummy':>8}{'oracle':>8}"
        f"{'반복노출':>9}  {'검증 문서':<30}"
    )
    print(header)
    print("-" * 100)
    for d in diagnostics:
        print(
            f"{d.fold.index:>2} {d.fold.test_document:<30}{d.test_size:>5}"
            f"{d.trained_majority_accuracy:>8.1%}{d.oracle_majority_accuracy:>8.1%}"
            f"{d.repeat_exposure_rate:>9.1%}  {d.fold.validation_document:<30}"
        )

    sizes = [d.test_size for d in diagnostics]
    for name, values in (
        ("Dummy(배포 가능)", [d.trained_majority_accuracy for d in diagnostics]),
        ("oracle(난이도 눈금)", [d.oracle_majority_accuracy for d in diagnostics]),
        ("반복 노출률", [d.repeat_exposure_rate for d in diagnostics]),
    ):
        agg = aggregate(values, sizes)
        print(
            f"\n{name:<20} fold 평균 {agg['fold_mean']:.1%} / "
            f"건수 가중 {agg['count_weighted']:.1%}"
        )

    gaps = [
        d
        for d in diagnostics
        if d.trained_majority_accuracy < d.oracle_majority_accuracy - 1e-9
    ]
    print(
        f"\n두 기준선이 갈리는 fold {len(gaps)}개 — 평가 문서의 최빈 클래스가 "
        f"전체 최빈({diagnostics[0].trained_majority_label})과 다르다:"
    )
    for d in gaps:
        top = max(d.test_label_counts.items(), key=lambda kv: kv[1])[0]
        print(
            f"  {d.fold.test_document:<30} 최빈 {top} "
            f"({d.trained_majority_accuracy:.1%} -> {d.oracle_majority_accuracy:.1%})"
        )

    print("\n학습 또는 평가에 없는 희소 값:")
    for d in diagnostics:
        for field_name, values in d.rare_value_coverage.items():
            for value, in_test in values.items():
                print(
                    f"  {d.fold.test_document:<30} {field_name}={value} "
                    f"평가 {in_test}건"
                )


if __name__ == "__main__":
    _main()
