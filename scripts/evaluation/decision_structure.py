"""결정 구조를 바꾸면 `견적반영`↔`계약·질의검토` 경계가 움직이는가.

오답의 3분의 1이 이 경계 하나에서 나온다(2026-09-01 17:59). 파인튜닝으로 모델 계열을
바꾸고 용량을 세 배 키우고 입력을 마스킹해도 경계 혼동은 98건에서 97건이 됐을 뿐이다
(2026-09-02 22:06). 남은 의심은 **분류기가 세 라벨을 한 번에 가르는 구조 자체**였다.

이 모듈은 그 의심을 네 가지로 나눠 검사한다.

1. `multinomial` — 현행. softmax 하나로 세 라벨을 동시에 가른다.
2. `OvR` — 라벨마다 "이것 대 나머지" 분류기를 두고 가장 확신하는 것을 고른다.
3. `OvO` — 세 짝마다 분류기를 둔다. **`견적 대 계약`만 전담하는 분류기가 생긴다.**
4. 캐스케이드 — 1단계로 `통상수용` 대 `검토필요`, 2단계로 그 안에서 `견적` 대 `계약`.

`boundary_ceiling`은 다른 질문에 답한다 — 경계 두 라벨만 남기고 학습하면 얼마나
맞히는가. 이것은 **오라클 라우팅**을 가정한 값이다. 진짜 경계 건만 골라 넣어주므로
실제로는 얻을 수 없고, 캐스케이드가 왜 그 값에 못 미치는지를 설명하는 상한이다.

fold·전처리·클래스 가중치·평가 지표는 `baselines`의 것을 그대로 쓴다. 바뀌는 것은
파이프라인 마지막 단계뿐이라, 나온 숫자를 기존 기준선과 바로 견줄 수 있다.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier

from scripts.evaluation.baselines import (
    DEFAULT_THRESHOLD,
    FoldResult,
    _fold_result_from_predictions,
    _model_input,
    _repeat_flags,
    _resolved_class_weight,
)
from scripts.evaluation.folds import make_lodo_folds

ACCEPT = "통상수용"
BOUNDARY = frozenset({"견적반영", "계약·질의검토"})
REVIEW = "검토필요"


@dataclass(frozen=True)
class StructureResult:
    """한 구성의 결과.

    :param name: 표에 찍히는 이름.
    :param folds: fold별 결과. 지표 계산은 `baselines`와 같은 함수를 쓴다.
    :param errors: 오답 총 건수.
    :param boundary_errors: 그중 `견적반영`↔`계약·질의검토` 상호 혼동.
    """

    name: str
    folds: tuple[FoldResult, ...]
    errors: int
    boundary_errors: int

    @property
    def macro_f1(self) -> float:
        return float(np.mean([f.macro_f1 for f in self.folds]))

    @property
    def review_recall(self) -> float:
        return float(np.mean([f.review_recall for f in self.folds]))

    @property
    def boundary_share(self) -> float:
        """경계 혼동이 오답에서 차지하는 비율. 구성을 바꿔도 이 값이 안 움직인다."""
        return self.boundary_errors / self.errors if self.errors else 0.0


def _fit(spec, rows: Sequence[dict[str, Any]], labels: Sequence[str], wrapper=None):
    """학습 fold의 분포로 가중치를 정하고, 필요하면 분류기를 감싼다."""
    pipeline = replace(spec, class_weight=_resolved_class_weight(spec, labels)).build()
    if wrapper is not None:
        name, classifier = pipeline.steps[-1]
        pipeline.steps[-1] = (name, wrapper(classifier))
    pipeline.fit(_model_input(spec, rows), labels)
    return pipeline


def _collect(name, rows, predict_fold) -> StructureResult:
    """fold를 돌며 예측을 모으고 `baselines`와 같은 방식으로 채점한다."""
    folds, errors, boundary = [], 0, 0
    for fold in make_lodo_folds(rows):
        fit_rows, _, test_rows = fold.split(rows)
        pred = predict_fold(fit_rows, test_rows)
        for gold, guess in zip((r["primary_action"] for r in test_rows), pred):
            if gold != guess:
                errors += 1
                if {gold, guess} == BOUNDARY:
                    boundary += 1
        folds.append(
            _fold_result_from_predictions(
                fold, rows, test_rows, list(pred),
                train_size=len(fit_rows),
                repeat_flags=_repeat_flags(fold, rows, DEFAULT_THRESHOLD),
                repeat_threshold=DEFAULT_THRESHOLD,
                review_weight_multiplier=0.0,
                type_feature_weight=0.0,
            )
        )
    return StructureResult(name, tuple(folds), errors, boundary)


def run_flat(rows, spec, *, wrapper: Callable | None = None, name: str = "multinomial") -> StructureResult:
    """세 라벨을 한 분류기로 가른다. `wrapper`로 OvR·OvO를 씌운다."""

    def predict(fit_rows, test_rows):
        model = _fit(spec, fit_rows, [r["primary_action"] for r in fit_rows], wrapper)
        return model.predict(_model_input(spec, test_rows))

    return _collect(name, rows, predict)


def run_cascade(rows, spec, *, name: str = "캐스케이드") -> StructureResult:
    """1단계 `통상수용` 대 `검토필요`, 2단계 `견적` 대 `계약`.

    2단계는 **실제로 검토필요인 건만** 학습하지만, 평가 때는 1단계가 보낸 건을 받는다.
    1단계가 잘못 보낸 건이 섞이고 놓친 건은 아예 2단계에 오지 않으므로 오차가 곱해진다.
    이것이 `boundary_ceiling`의 오라클 값과 벌어지는 이유다.
    """

    def predict(fit_rows, test_rows):
        stage1 = _fit(
            spec, fit_rows,
            [ACCEPT if r["primary_action"] == ACCEPT else REVIEW for r in fit_rows],
        )
        routed = list(stage1.predict(_model_input(spec, test_rows)))

        boundary_rows = [r for r in fit_rows if r["primary_action"] in BOUNDARY]
        stage2 = _fit(spec, boundary_rows, [r["primary_action"] for r in boundary_rows])

        need = [i for i, label in enumerate(routed) if label == REVIEW]
        pred = [ACCEPT] * len(test_rows)
        if need:
            sub = [test_rows[i] for i in need]
            for i, label in zip(need, stage2.predict(_model_input(spec, sub))):
                pred[i] = label
        return pred

    return _collect(name, rows, predict)


VARIANTS: tuple[tuple[str, Callable | None], ...] = (
    ("multinomial (현행)", None),
    ("OvR", OneVsRestClassifier),
    ("OvO", OneVsOneClassifier),
)


def run_all(rows, spec) -> list[StructureResult]:
    """네 구성을 같은 fold로 돌린다."""
    results = [run_flat(rows, spec, wrapper=w, name=n) for n, w in VARIANTS]
    results.append(run_cascade(rows, spec))
    return results


def boundary_ceiling(rows, spec) -> dict[str, float]:
    """경계 두 라벨만 남기고 학습·평가한다 — 오라클 라우팅 상한.

    다수 클래스만 찍는 기준선과 함께 봐야 뜻이 있다. 경계가 아예 학습되지 않는다면
    이 값이 기준선 근처에 머문다.
    """
    pair_rows = [r for r in rows if r["primary_action"] in BOUNDARY]
    if not pair_rows:
        raise ValueError("경계 라벨을 가진 행이 없습니다.")

    counts = {label: sum(1 for r in pair_rows if r["primary_action"] == label) for label in BOUNDARY}
    gold_all: list[str] = []
    pred_all: list[str] = []
    fold_accuracy: list[float] = []
    for fold in make_lodo_folds(rows):
        fit_all, _, test_all = fold.split(rows)
        fit = [r for r in fit_all if r["primary_action"] in BOUNDARY]
        test = [r for r in test_all if r["primary_action"] in BOUNDARY]
        if len(test) < 5 or len({r["primary_action"] for r in fit}) < 2:
            continue
        model = _fit(spec, fit, [r["primary_action"] for r in fit])
        pred = list(model.predict(_model_input(spec, test)))
        gold = [r["primary_action"] for r in test]
        fold_accuracy.append(accuracy_score(gold, pred))
        gold_all += gold
        pred_all += pred

    return {
        "n": len(pair_rows),
        "majority_baseline": max(counts.values()) / len(pair_rows),
        "fold_mean_accuracy": float(np.mean(fold_accuracy)),
        "pooled_accuracy": accuracy_score(gold_all, pred_all),
        "pooled_macro_f1": float(f1_score(gold_all, pred_all, average="macro")),
        "folds": len(fold_accuracy),
    }


def result_rows(results: Sequence[StructureResult]) -> list[dict[str, Any]]:
    """`DataFrame`에 바로 넣을 행 목록."""
    return [
        {
            "구성": r.name,
            "fold평균 macro F1": r.macro_f1,
            "검토 recall": r.review_recall,
            "오답": r.errors,
            "경계 혼동": r.boundary_errors,
            "경계 비중": r.boundary_share,
        }
        for r in results
    ]
