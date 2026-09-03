"""견적↔계약 경계를 겨냥한 두 가지 변형을 word+char 기준선 위에서 LODO로 잰다.

1. **불릿 max-pooling** — 2026-09-03의 KoBigBird 결과는 절단이 아니라 *희석*을 가리켰다.
   긴 원문에서 blocker 한 줄이 원가 어휘에 묻힌다. 학습은 그대로 두고 예측만
   `model_text`를 줄 단위로 나눠 각 줄의 계약 확률을 구한 뒤 최댓값을 취한다.
   "한 줄이라도 blocker면 계약"이라는 라벨 규칙(결정 21)과 구조가 같다.
2. **blocker 유형 타깃** — `계약·질의검토`는 5종 blocker의 합집합이고 종류마다 어휘가
   다르다. 5종 다중 라벨과 cost_basis 7종을 따로 배우고 `derive_primary_action()`의
   고정 규칙으로 주 라벨을 만든다. 학습 집합을 가르지 않으므로 2026-09-02에 계층 분해를
   접은 이유(LLM 경계를 아키텍처에 못 박음)가 적용되지 않는다.

명령: `$env:RFP_DATASET_VERSION='v4'; python -m scripts.evaluation.boundary_features`
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

from scripts.evaluation.baselines import (
    LABELS,
    RANDOM_STATE,
    REVIEW_LABEL,
    WORD_CHAR_BALANCED,
    _fit_pipeline,
)
from scripts.evaluation.folds import make_lodo_folds
from scripts.labeling.label_dataset import get_model_text, load_label_dataset
from scripts.labeling.label_schema import BLOCKER_TYPES

ROOT = Path(__file__).resolve().parents[2]
QUOTE_LABEL = "견적반영"
MIN_LINE_CHARS = 15  # ponytail: "[현황 분석]" 같은 제목 줄은 사전확률만 찍으므로 제외한다


def lines_of(row) -> list[str]:
    return [l for l in get_model_text(row).splitlines() if len(l.strip()) >= MIN_LINE_CHARS]


def logistic():
    return LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE)


def run_fold(fold, rows):
    fit_rows, _, test_rows = fold.split(rows)
    pipe = _fit_pipeline(WORD_CHAR_BALANCED, fit_rows)
    features = Pipeline(pipe.steps[:-1])
    clf = pipe.steps[-1][1]
    review_idx = list(clf.classes_).index(REVIEW_LABEL)

    texts = [get_model_text(r) for r in test_rows]
    whole = clf.predict_proba(features.transform(texts))
    out = {"baseline": [clf.classes_[i] for i in whole.argmax(1)]}

    # 1. 불릿 max-pooling. 각 건의 줄을 한 번에 변환해 계약 확률 최댓값을 얻는다.
    pooled = whole.copy()
    any_line_review = []
    for i, row in enumerate(test_rows):
        ls = lines_of(row)
        if not ls:
            any_line_review.append(False)
            continue
        p = clf.predict_proba(features.transform(ls))
        pooled[i, review_idx] = max(pooled[i, review_idx], p[:, review_idx].max())
        any_line_review.append(bool((p.argmax(1) == review_idx).any()))
    out["maxpool_soft"] = [clf.classes_[i] for i in pooled.argmax(1)]
    out["maxpool_any"] = [
        REVIEW_LABEL if a else b for a, b in zip(any_line_review, out["baseline"])
    ]

    # 2. blocker 유형 + cost_basis → 규칙으로 주 라벨.
    X_fit = features.transform([get_model_text(r) for r in fit_rows])
    X_test = features.transform(texts)
    has_blocker = np.zeros(len(test_rows), dtype=bool)
    for btype in BLOCKER_TYPES:
        y = [btype in r["blockers"] for r in fit_rows]
        if sum(y) < 2:
            continue
        has_blocker |= logistic().fit(X_fit, y).predict(X_test)
    cost = logistic().fit(X_fit, [r["cost_basis"] for r in fit_rows]).predict(X_test)
    out["blocker_types"] = [
        REVIEW_LABEL if b else (QUOTE_LABEL if c != "없음" else LABELS[0])
        for b, c in zip(has_blocker, cost)
    ]
    # 2b. 유형을 안 나누고 blocker 유무 + 원가 유무만 배운 대조군. 세분화의 효과를 분리한다.
    any_blocker = logistic().fit(X_fit, [bool(r["blockers"]) for r in fit_rows]).predict(X_test)
    any_cost = logistic().fit(X_fit, [r["cost_basis"] != "없음" for r in fit_rows]).predict(X_test)
    out["blocker_binary"] = [
        REVIEW_LABEL if b else (QUOTE_LABEL if c else LABELS[0])
        for b, c in zip(any_blocker, any_cost)
    ]
    gold = [r["primary_action"] for r in test_rows]
    return gold, out


def main() -> None:
    rows, meta = load_label_dataset()
    folds = make_lodo_folds(rows)
    per_fold: dict[str, list[float]] = {}
    pooled_gold, pooled_pred = [], {}
    for fold in folds:
        gold, out = run_fold(fold, rows)
        pooled_gold += gold
        for name, pred in out.items():
            per_fold.setdefault(name, []).append(
                f1_score(gold, pred, labels=LABELS, average="macro", zero_division=0)
            )
            pooled_pred.setdefault(name, []).extend(pred)
        print(f"fold {fold.index} {fold.test_document:30s} " + "  ".join(
            f"{n}={per_fold[n][-1]:.3f}" for n in out))

    base = per_fold["baseline"]
    lines = ["| 변형 | fold 평균 | 통합 OOF | 통상 | 견적 | 계약 | 오답 | 견적↔계약 혼동 | 우세 fold |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    summary = {}
    for name, scores in per_fold.items():
        pred = pooled_pred[name]
        macro = f1_score(pooled_gold, pred, labels=LABELS, average="macro", zero_division=0)
        per = f1_score(pooled_gold, pred, labels=LABELS, average=None, zero_division=0)
        wrong = sum(g != p for g, p in zip(pooled_gold, pred))
        boundary = sum(
            {g, p} == {QUOTE_LABEL, REVIEW_LABEL} for g, p in zip(pooled_gold, pred)
        )
        wins = sum(s > b for s, b in zip(scores, base))
        summary[name] = dict(fold_mean=float(np.mean(scores)), pooled=float(macro),
                             per_class=dict(zip(LABELS, map(float, per))), wrong=wrong,
                             boundary_confusion=boundary, wins_vs_baseline=wins,
                             fold_scores=[float(s) for s in scores])
        lines.append(
            f"| {name} | {np.mean(scores):.3f} | {macro:.3f} | {per[0]:.3f} | {per[1]:.3f} | "
            f"{per[2]:.3f} | {wrong} | {boundary} | {wins}/10 |")
    table = "\n".join(lines)
    print(table)
    out_dir = ROOT / "reports" / "current" / "v4"
    (out_dir / "boundary_features.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "boundary_features.md").write_text(
        "# 경계 겨냥 변형 — 불릿 max-pooling과 blocker 유형 타깃\n\n"
        f"- 데이터: {meta['dataset_version']}, 앵커 제외 924건, LODO 10-fold, 학습 8문서\n"
        "- 기반 모델: word+char TF-IDF balanced Logistic (기준선과 같은 학습)\n"
        "- 명령: `python -m scripts.evaluation.boundary_features`\n"
        "- 사전 판정 기준: 기준선 대비 10 fold 중 8 우세\n\n"
        "| 변형 | 뜻 |\n|---|---|\n"
        "| baseline | 기준선 그대로 (재현 확인용) |\n"
        f"| maxpool_soft | 줄({MIN_LINE_CHARS}자 이상)별 계약 확률의 최댓값으로 계약 확률을 올린 뒤 argmax |\n"
        "| maxpool_any | 어느 한 줄이라도 계약으로 예측되면 계약, 아니면 기준선 예측 |\n"
        "| blocker_types | blocker 5종 다중 라벨 + cost_basis 7종 → 결정 21 규칙 |\n"
        "| blocker_binary | blocker 유무 + 원가 유무 (2진) → 결정 21 규칙. 세분화 효과의 대조군 |\n\n"
        + table + "\n", encoding="utf-8")
    print(f"기록: {out_dir / 'boundary_features.md'}")


if __name__ == "__main__":
    main()
