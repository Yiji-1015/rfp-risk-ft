#!/usr/bin/env python3
"""`통상수용` 대 `검토필요` 2분류를 같은 LODO에서 정식으로 학습·평가한다.

3분류의 오답 294건 중 98건이 `견적반영`과 `계약·질의검토`의 상호 혼동이다. 두 소수
클래스를 하나로 합치면 그 경계를 피할 수 있는데, 문제는 3분류로 학습한 예측을 사후에
접은 값(macro F1 0.788)이 학습 결과가 아니라는 것이다. 이 스크립트는 접은 라벨로
**처음부터 학습해서** 그 값을 확정한다.

`검토필요`는 blocker가 있거나 원가 요인이 있는 건의 합집합이며, 학습 집합을 라벨로
가르지 않는다. LLM이 blocker라고 부른 경계에 구조를 기대지 않기 위해서다.

`baselines.py`는 세 라벨을 모듈 상수로 고정하고 있고 그 위에 기록된 결과가 많아
건드리지 않는다. 파이프라인 구성과 fold만 재사용하고 지표는 여기서 따로 센다.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
from pathlib import Path
from typing import Any, Sequence

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from scripts.evaluation.baselines import (
    CHAR_BALANCED,
    DUMMY,
    SVM_BALANCED,
    WORD_BALANCED,
    WORD_CHAR_BALANCED,
    ModelSpec,
    _fit_pipeline,
    _model_input,
)
from scripts.evaluation.folds import make_lodo_folds
from scripts.labeling.label_dataset import (
    DATASET_VERSION_ENV,
    DEFAULT_DATASET_KEY,
    load_label_dataset,
)

ROOT = Path(__file__).resolve().parents[2]
ACCEPT, REVIEW = "통상수용", "검토필요"
BINARY_LABELS = (ACCEPT, REVIEW)

SPECS = (DUMMY, WORD_BALANCED, CHAR_BALANCED, SVM_BALANCED, WORD_CHAR_BALANCED)


def collapse(label: str) -> str:
    """`견적반영`과 `계약·질의검토`를 하나로 합친다."""
    return ACCEPT if label == ACCEPT else REVIEW


def collapse_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """주 라벨만 접은 사본을 만든다. 원본 행은 건드리지 않는다."""
    return [
        {**row, "primary_action": collapse(row["primary_action"])} for row in rows
    ]


def score(gold: Sequence[str], pred: Sequence[str]) -> dict[str, float]:
    """2분류 지표. `labels`를 명시해 한쪽 클래스가 빠진 fold에서도 분모를 고정한다."""
    return {
        "macro_f1": float(
            f1_score(gold, pred, labels=BINARY_LABELS, average="macro", zero_division=0)
        ),
        "accuracy": float(accuracy_score(gold, pred)),
        "review_precision": float(
            precision_score(gold, pred, labels=[REVIEW], average="macro", zero_division=0)
        ),
        "review_recall": float(
            recall_score(gold, pred, labels=[REVIEW], average="macro", zero_division=0)
        ),
        "review_f1": float(
            f1_score(gold, pred, labels=[REVIEW], average="macro", zero_division=0)
        ),
    }


def run_binary_lodo(
    rows: Sequence[dict[str, Any]], spec: ModelSpec
) -> dict[str, Any]:
    """접은 라벨로 fold마다 학습하고, fold 평균과 통합 OOF를 함께 돌려준다."""
    collapsed = collapse_rows(rows)
    per_fold, pooled_gold, pooled_pred, predictions = [], [], [], []

    for fold in make_lodo_folds(collapsed):
        fit_rows, _, test_rows = fold.split(collapsed)
        pipeline = _fit_pipeline(spec, fit_rows)
        gold = [row["primary_action"] for row in test_rows]
        pred = list(pipeline.predict(_model_input(spec, test_rows)))

        per_fold.append({"test_document": fold.test_document, "size": len(test_rows), **score(gold, pred)})
        pooled_gold.extend(gold)
        pooled_pred.extend(pred)
        predictions.extend(
            {
                "requirement_uid": row["requirement_uid"],
                "test_document": fold.test_document,
                "gold": g,
                "pred": p,
            }
            for row, g, p in zip(test_rows, gold, pred)
        )

    average = {
        key: statistics.fmean(fold[key] for fold in per_fold)
        for key in ("macro_f1", "accuracy", "review_precision", "review_recall", "review_f1")
    }
    return {
        "name": spec.name,
        "fold_average": average,
        "pooled": score(pooled_gold, pooled_pred),
        "folds": per_fold,
        "predictions": predictions,
    }


def collapsed_three_class_reference(oof_path: Path, model: str) -> dict[str, float] | None:
    """3분류로 학습한 예측을 사후에 접은 값. 학습 결과가 아니라 비교용 참조다."""
    if not oof_path.exists():
        return None
    with oof_path.open(encoding="utf-8-sig", newline="") as handle:
        saved = list(csv.DictReader(handle))
    gold = [collapse(record["gold"]) for record in saved]
    pred = [collapse(record[f"{model}_pred"]) for record in saved]
    return score(gold, pred)


def render_markdown(
    results: Sequence[dict[str, Any]],
    reference: dict[str, float] | None,
    *,
    version: str,
    size: int,
) -> str:
    lines = [
        f"# {version} 통상수용 대 검토필요 2분류",
        "",
        f"- 데이터셋: `label_dataset_{version}`, 동결 앵커 100건을 제외한 {size}건",
        "- 평가: 학습 8 / 검증 1 / 평가 1 문서 LODO 10-fold, 3분류와 같은 분할",
        "- 라벨: `견적반영`과 `계약·질의검토`를 `검토필요`로 합친 뒤 **처음부터 학습**",
        f"- 명령: `$env:{DATASET_VERSION_ENV}='{version}'; python -m scripts.evaluation.binary_review`",
        "",
        "## fold 단순 평균",
        "",
        "| 설정 | macro F1 | 정확도 | 검토 precision | 검토 recall | 검토 F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        value = result["fold_average"]
        lines.append(
            f"| {result['name']} | {value['macro_f1']:.3f} | {value['accuracy']:.3f} | "
            f"{value['review_precision']:.3f} | {value['review_recall']:.3f} | {value['review_f1']:.3f} |"
        )

    lines += ["", "## 통합 OOF (fold를 나누지 않고 전체를 한 번에)", "",
              "| 설정 | macro F1 | 정확도 | 검토 precision | 검토 recall | 검토 F1 |",
              "|---|---:|---:|---:|---:|---:|"]
    for result in results:
        value = result["pooled"]
        lines.append(
            f"| {result['name']} | {value['macro_f1']:.3f} | {value['accuracy']:.3f} | "
            f"{value['review_precision']:.3f} | {value['review_recall']:.3f} | {value['review_f1']:.3f} |"
        )

    if reference:
        lines += [
            "",
            "## 참조 — 3분류 예측을 사후에 접은 값",
            "",
            "학습 결과가 아니다. 3분류로 학습한 word+char Logistic의 예측을 같은 규칙으로",
            "접어서 통합 OOF로 잰 값이며, 위 표의 통합 OOF와만 비교한다.",
            "",
            f"- macro F1 {reference['macro_f1']:.3f} / 정확도 {reference['accuracy']:.3f} / "
            f"검토 P {reference['review_precision']:.3f} R {reference['review_recall']:.3f} "
            f"F1 {reference['review_f1']:.3f}",
        ]
    return "\n".join(lines) + "\n"


def main() -> None:
    version = os.getenv(DATASET_VERSION_ENV, DEFAULT_DATASET_KEY)
    default_dir = ROOT / "reports" / "current" / version
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof", type=Path, default=default_dir / "model_candidate_oof.csv")
    parser.add_argument("--reference-model", default="word_char_logistic")
    parser.add_argument("--output", type=Path, default=default_dir / "binary_review_results.md")
    parser.add_argument("--json", type=Path, default=default_dir / "binary_review_results.json")
    args = parser.parse_args()

    rows, _ = load_label_dataset()
    results = [run_binary_lodo(rows, spec) for spec in SPECS]
    reference = collapsed_three_class_reference(args.oof, args.reference_model)
    size = len(results[0]["predictions"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        render_markdown(results, reference, version=version, size=size), encoding="utf-8"
    )
    args.json.write_text(
        json.dumps(
            {"dataset_version": version, "evaluated": size, "results": results,
             "collapsed_three_class_reference": reference},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"2분류 평가 {size}건")
    for result in results:
        print(
            f"  {result['name']:<34} fold평균 macro F1 {result['fold_average']['macro_f1']:.3f}"
            f"  통합 {result['pooled']['macro_f1']:.3f}"
        )
    print(f"저장: {args.output}")
    print(f"저장: {args.json}")


if __name__ == "__main__":
    main()
