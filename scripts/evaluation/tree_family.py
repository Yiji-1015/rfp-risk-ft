"""트리 계열을 SVD 없이 판정하고, 앙상블 재료로서의 값어치를 잰다.

가이드 02 §6-1은 `SVD100 + 구조·숫자 + XGBoost`(0.564)가 기준선(0.601)에 못 미친
것을 보고 "범인은 분류기가 아니라 SVD 100차원 압축"이라고 적었다. 그 스펙은 SVD와
구조·숫자 피처와 분류기가 **한꺼번에** 바뀐 것이라 트리 단독 효과가 분리되지 않는데,
SVD를 뺀 트리는 그 뒤로 돌린 적이 없다. 이 스크립트가 그 빈자리를 메운다.

점수만 재지 않는다. `finetune_ensemble.py`가 적었듯 앙상블은 "계열이 달라 틀리는
자리가 다를 때"만 올랐다. 그래서 오답 겹침과 오라클 정확도를 함께 내고, 실제로
섞었을 때 오르는지까지 확인한다. 다양성이 있어도 멤버가 약하면 평균이 내려간다는
것이 v5의 결론이다.
"""

from __future__ import annotations

import argparse
import statistics
from dataclasses import replace
from typing import Any, Sequence

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import f1_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from scripts.evaluation import baselines as B
from scripts.evaluation.folds import make_lodo_folds
from scripts.labeling.label_dataset import load_label_dataset

LABELS = ["통상수용", "견적반영", "계약·질의검토"]

# 기본값은 `_LabelEncodedXGBClassifier`와 같다. 그쪽은 100차원 SVD 입력을 전제로
# 굳어진 설정이라 10만 차원 희소 입력에서 과소적합일 수 있다는 가설을 확인하려면
# 용량을 올린 설정과 나란히 놓아야 한다.
DEFAULT_XGB = dict(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
)
LARGE_XGB = dict(
    n_estimators=600,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.3,
    min_child_weight=1,
)


class SparseXGB(ClassifierMixin, BaseEstimator):
    """희소 TF-IDF를 SVD 없이 그대로 받는 XGBoost.

    `baselines._LabelEncodedXGBClassifier`와 달리 설정을 주입받는다. 용량을 바꿔가며
    같은 입력에서 비교해야 "과소적합인가 계열이 안 맞는가"를 가를 수 있다.
    """

    def __init__(self, class_weight: str | dict[str, float] | None = "balanced", **params: Any):
        self.class_weight = class_weight
        self.params = params or dict(DEFAULT_XGB)

    def fit(self, X: Any, y: Sequence[str]) -> SparseXGB:
        self.encoder_ = LabelEncoder().fit(y)
        weight = compute_sample_weight(self.class_weight, y) if self.class_weight else None
        self.model_ = XGBClassifier(
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=B.RANDOM_STATE,
            n_jobs=-1,
            **self.params,
        )
        self.model_.fit(X, self.encoder_.transform(y), sample_weight=weight)
        self.classes_ = self.encoder_.classes_
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        return self.model_.predict_proba(X)

    def predict(self, X: Any) -> np.ndarray:
        return self.encoder_.inverse_transform(self.model_.predict(X).astype(int))


def _pipeline(spec: B.ModelSpec, labels: Sequence[str], xgb_params: dict | None) -> Pipeline:
    """스펙의 벡터라이저는 그대로 두고 분류기만 트리로 갈아끼운다.

    벡터라이저를 공유해야 "분류기만 다르다"가 성립한다. 등록된 스펙처럼 피처까지
    함께 바꾸면 어느 쪽 탓인지 다시 알 수 없게 된다.
    """
    resolved = replace(spec, class_weight=B._resolved_class_weight(spec, labels))
    proto = resolved.build()
    if xgb_params is None:
        return proto
    clf = SparseXGB(class_weight=resolved.class_weight, **xgb_params)
    return Pipeline(list(proto.steps[:-1]) + [("clf", clf)])


def collect_probabilities(
    rows: list[dict],
    members: dict[str, tuple[B.ModelSpec, dict | None]],
    verbose: bool = False,
) -> tuple[dict[str, str], dict[str, dict[str, np.ndarray]], dict[str, int]]:
    """멤버별 OOF 확률을 `LABELS` 순서로 맞춰 모은다."""
    folds = list(make_lodo_folds(rows))
    gold: dict[str, str] = {}
    fold_of: dict[str, int] = {}
    proba: dict[str, dict[str, np.ndarray]] = {name: {} for name in members}
    for index, fold in enumerate(folds):
        fit_rows, _, test_rows = fold.split(rows)
        labels = [row["primary_action"] for row in fit_rows]
        for row in test_rows:
            gold[row["requirement_uid"]] = row["primary_action"]
            fold_of[row["requirement_uid"]] = index
        for name, (spec, xgb_params) in members.items():
            pipe = _pipeline(spec, labels, xgb_params)
            pipe.fit(B._model_input(spec, fit_rows), labels)
            matrix = pipe.predict_proba(B._model_input(spec, test_rows))
            order = [list(pipe.classes_).index(label) for label in LABELS]
            for row, probabilities in zip(test_rows, matrix):
                proba[name][row["requirement_uid"]] = probabilities[order]
        if verbose:
            print(f"  fold {index + 1}/{len(folds)} 완료", flush=True)
    return gold, proba, fold_of


def score(
    gold: dict[str, str], pred: dict[str, str], fold_of: dict[str, int]
) -> dict[str, Any]:
    """fold 평균으로 낸다. 문서 크기가 달라 통합 평균과 값이 다르다."""
    per_fold, per_recall = [], []
    for index in sorted(set(fold_of.values())):
        uids = [uid for uid in gold if fold_of[uid] == index]
        truth = [gold[uid] for uid in uids]
        guess = [pred[uid] for uid in uids]
        per_fold.append(
            f1_score(truth, guess, labels=LABELS, average="macro", zero_division=0)
        )
        per_recall.append(
            recall_score(
                truth, guess, labels=["계약·질의검토"], average="macro", zero_division=0
            )
        )
    return {
        "macro_f1": statistics.fmean(per_fold),
        "review_recall": statistics.fmean(per_recall),
        "accuracy": sum(gold[uid] == pred[uid] for uid in gold) / len(gold),
        "per_fold": per_fold,
    }


def soft_vote(
    proba: dict[str, dict[str, np.ndarray]],
    names: Sequence[str],
    uids: Sequence[str],
    weights: Sequence[float] | None = None,
) -> dict[str, str]:
    resolved = list(weights) if weights else [1.0] * len(names)
    return {
        uid: LABELS[
            int(np.argmax(sum(w * proba[n][uid] for n, w in zip(names, resolved))))
        ]
        for uid in uids
    }


def overlap(
    gold: dict[str, str], left: dict[str, str], right: dict[str, str]
) -> dict[str, int | float]:
    """오답이 겹치는가. 앙상블 재료인지는 점수가 아니라 이 값이 정한다."""
    left_wrong = {uid for uid in gold if gold[uid] != left[uid]}
    right_wrong = {uid for uid in gold if gold[uid] != right[uid]}
    both = len(left_wrong & right_wrong)
    return {
        "left_errors": len(left_wrong),
        "right_errors": len(right_wrong),
        "both_wrong": both,
        "one_wrong": len(left_wrong ^ right_wrong),
        "oracle_accuracy": 1 - both / len(gold),
    }


TITLES = {
    "wc": "word+char + Logistic (기준선)",
    "ch": "char + Logistic",
    "xgb": "word+char + XGBoost (depth3, 200)",
    "xgbL": "word+char + XGBoost (depth6, 600)",
}

COMBOS = [
    ("wc + xgb", ["wc", "xgb"], None),
    ("wc + xgb (3:1)", ["wc", "xgb"], [3.0, 1.0]),
    ("wc + xgb (2:1)", ["wc", "xgb"], [2.0, 1.0]),
    ("wc + ch (기존 동일계열)", ["wc", "ch"], None),
    ("wc + ch + xgb", ["wc", "ch", "xgb"], None),
    ("wc + ch + xgb (2:1:1)", ["wc", "ch", "xgb"], [2.0, 1.0, 1.0]),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quiet", action="store_true", help="fold 진행 출력을 끈다")
    args = parser.parse_args()

    rows, meta = load_label_dataset()
    print(f"데이터셋 {meta['dataset_version']} / {meta['row_count']}건\n")

    members = {
        "wc": (B.WORD_CHAR_BALANCED, None),
        "ch": (B.CHAR_BALANCED, None),
        "xgb": (B.WORD_CHAR_BALANCED, dict(DEFAULT_XGB)),
        "xgbL": (B.WORD_CHAR_BALANCED, dict(LARGE_XGB)),
    }
    gold, proba, fold_of = collect_probabilities(rows, members, verbose=not args.quiet)
    uids = sorted(gold)
    single = {name: soft_vote(proba, [name], uids) for name in members}
    base = score(gold, single["wc"], fold_of)

    print(f"\n{'단독':<38}{'macroF1':>9}{'검토recall':>11}{'정확도':>9}{'우세':>7}")
    print("-" * 74)
    for name, title in TITLES.items():
        result = score(gold, single[name], fold_of)
        if name == "wc":
            win = "—"
        else:
            better = sum(
                1 for a, b in zip(result["per_fold"], base["per_fold"]) if a > b
            )
            win = f"{better}/{len(result['per_fold'])}"
        print(
            f"{title:<38}{result['macro_f1']:>9.4f}{result['review_recall']:>11.4f}"
            f"{result['accuracy']:>9.4f}{win:>7}"
        )

    print(f"\n{'오답 겹침 (기준선 대비)':<38}{'오답':>8}{'둘다':>7}{'한쪽만':>8}{'오라클':>9}")
    print("-" * 74)
    for name in ("ch", "xgb", "xgbL"):
        stats = overlap(gold, single["wc"], single[name])
        print(
            f"{TITLES[name]:<38}{stats['right_errors']:>8}{stats['both_wrong']:>7}"
            f"{stats['one_wrong']:>8}{stats['oracle_accuracy']:>9.3f}"
        )

    print(f"\n{'soft voting':<38}{'macroF1':>9}{'검토recall':>11}{'정확도':>9}{'우세':>7}")
    print("-" * 74)
    print(
        f"{'word+char 단독 (기준선)':<38}{base['macro_f1']:>9.4f}"
        f"{base['review_recall']:>11.4f}{base['accuracy']:>9.4f}{'—':>7}"
    )
    for title, names, weights in COMBOS:
        result = score(gold, soft_vote(proba, names, uids, weights), fold_of)
        better = sum(1 for a, b in zip(result["per_fold"], base["per_fold"]) if a > b)
        print(
            f"{title:<38}{result['macro_f1']:>9.4f}{result['review_recall']:>11.4f}"
            f"{result['accuracy']:>9.4f}{better:>4}/{len(result['per_fold'])}"
        )


if __name__ == "__main__":
    main()
