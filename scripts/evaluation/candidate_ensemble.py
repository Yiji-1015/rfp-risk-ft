"""검증된 세 후보의 LODO 예측을 보존하고 단순 결합을 평가한다."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from scripts.evaluation.baselines import (
    CHAR_BALANCED,
    LABELS,
    REVIEW_LABEL,
    WORD_CHAR_BALANCED,
    _aligned_probabilities,
    _fit_pipeline,
    _fold_result_from_predictions,
    _model_input,
    summarize,
)
from scripts.evaluation.embeddings import TFIDF_E5, load_cached_embeddings, predict_hybrid_fold
from scripts.evaluation.duplication import DEFAULT_THRESHOLD
from scripts.evaluation.folds import _repeat_flags, make_lodo_folds

CANDIDATES = (
    ("char_logistic", "문자 3~4gram TF-IDF + balanced Logistic"),
    ("word_char_logistic", "단어 1~2gram + 문자 3~4gram TF-IDF + balanced Logistic"),
    ("tfidf_e5_hybrid", "문자 TF-IDF + E5 검증 선택 결합"),
)


def soft_vote(probabilities: Sequence[np.ndarray]) -> np.ndarray:
    """같은 클래스 순서의 확률을 동일 가중 평균한다."""
    if not probabilities:
        raise ValueError("후보 확률이 하나 이상 필요합니다")
    shape = probabilities[0].shape
    if any(values.shape != shape for values in probabilities):
        raise ValueError("후보 확률 배열의 크기가 모두 같아야 합니다")
    return np.mean(probabilities, axis=0)


def review_union_predictions(
    predictions: Sequence[Sequence[str]], fallback: Sequence[str]
) -> list[str]:
    """후보 하나라도 계약 검토라면 올리고, 아니면 결합 예측을 쓴다."""
    if any(len(values) != len(fallback) for values in predictions):
        raise ValueError("후보 예측 길이가 모두 같아야 합니다")
    return [
        REVIEW_LABEL if any(values[i] == REVIEW_LABEL for values in predictions) else label
        for i, label in enumerate(fallback)
    ]


def agreement_summary(
    gold: Sequence[str], predictions: Sequence[Sequence[str]]
) -> dict[str, float | int]:
    """세 후보가 같은 오류를 내는지 계산한다."""
    if not predictions or any(len(values) != len(gold) for values in predictions):
        raise ValueError("정답과 후보 예측 길이가 같아야 합니다")
    matrix = np.asarray(predictions, dtype=object)
    truth = np.asarray(gold, dtype=object)
    all_agree = np.all(matrix == matrix[0], axis=0)
    all_wrong = np.all(matrix != truth, axis=0)
    at_least_one_correct = np.any(matrix == truth, axis=0)
    total = len(gold)
    return {
        "row_count": total,
        "all_agree_count": int(all_agree.sum()),
        "all_agree_rate": float(all_agree.mean()),
        "all_wrong_count": int(all_wrong.sum()),
        "all_wrong_rate": float(all_wrong.mean()),
        "at_least_one_correct_count": int(at_least_one_correct.sum()),
        "at_least_one_correct_rate": float(at_least_one_correct.mean()),
    }


def _label_from_probabilities(probabilities: np.ndarray) -> list[str]:
    return [LABELS[i] for i in probabilities.argmax(axis=1)]


def run_candidate_ensemble(
    rows: Sequence[dict[str, Any]], embeddings: np.ndarray
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """세 후보와 두 고정 결합 규칙을 같은 LODO fold에서 평가한다."""
    fold_results: dict[str, list[Any]] = {
        key: [] for key, _ in CANDIDATES
    } | {"soft_vote": [], "review_union": []}
    records: list[dict[str, Any]] = []
    all_gold: list[str] = []
    all_predictions: list[list[str]] = [[], [], []]

    for fold in make_lodo_folds(rows):
        fit_rows, _, test_rows = fold.split(rows)
        probabilities = []
        predictions = []
        for spec in (CHAR_BALANCED, WORD_CHAR_BALANCED):
            pipeline = _fit_pipeline(spec, fit_rows)
            features = _model_input(spec, test_rows)
            current_probabilities = _aligned_probabilities(pipeline, features)
            probabilities.append(current_probabilities)
            predictions.append(_label_from_probabilities(current_probabilities))

        hybrid_pred, hybrid_probabilities, selected_weight = predict_hybrid_fold(
            rows, embeddings, fold, TFIDF_E5
        )
        probabilities.append(hybrid_probabilities)
        predictions.append(hybrid_pred)
        averaged = soft_vote(probabilities)
        voted = _label_from_probabilities(averaged)
        escalated = review_union_predictions(predictions, voted)
        flags = _repeat_flags(fold, rows, DEFAULT_THRESHOLD)

        for key, pred in zip(
            [key for key, _ in CANDIDATES] + ["soft_vote", "review_union"],
            predictions + [voted, escalated],
        ):
            fold_results[key].append(
                _fold_result_from_predictions(
                    fold,
                    rows,
                    test_rows,
                    pred,
                    train_size=len(fit_rows),
                    repeat_flags=flags,
                    embedding_weight=selected_weight if key == "tfidf_e5_hybrid" else 0.0,
                )
            )

        for model_index, pred in enumerate(predictions):
            all_predictions[model_index].extend(pred)
        gold = [row["primary_action"] for row in test_rows]
        all_gold.extend(gold)
        for i, row in enumerate(test_rows):
            record: dict[str, Any] = {
                "fold": fold.index,
                "test_document": fold.test_document,
                "requirement_uid": row["requirement_uid"],
                "gold": gold[i],
                "tfidf_e5_weight": selected_weight,
                "soft_vote_pred": voted[i],
                "review_union_pred": escalated[i],
                "all_agree": len({pred[i] for pred in predictions}) == 1,
                "all_wrong": all(pred[i] != gold[i] for pred in predictions),
                "at_least_one_correct": any(pred[i] == gold[i] for pred in predictions),
            }
            for (key, _), pred, values in zip(CANDIDATES, predictions, probabilities):
                record[f"{key}_pred"] = pred[i]
                for label, value in zip(LABELS, values[i]):
                    record[f"{key}_p_{label}"] = float(value)
            for label, value in zip(LABELS, averaged[i]):
                record[f"soft_vote_p_{label}"] = float(value)
            records.append(record)

    registry = {
        "protocol": "LODO: fit 8 documents / validation 1 / test 1",
        "probability_class_order": list(LABELS),
        "candidates": {
            key: {
                "name": name,
                "summary": summarize(fold_results[key]),
                "folds": [asdict(result) for result in fold_results[key]],
            }
            for key, name in CANDIDATES
        },
        "ensembles": {
            key: {
                "name": name,
                "summary": summarize(fold_results[key]),
                "folds": [asdict(result) for result in fold_results[key]],
            }
            for key, name in (
                ("soft_vote", "세 후보 동일 가중 soft voting"),
                ("review_union", "후보 하나라도 계약 검토면 검토"),
            )
        },
        "agreement": agreement_summary(all_gold, all_predictions),
    }
    return registry, records


def write_reports(
    registry: dict[str, Any],
    records: Sequence[dict[str, Any]],
    *,
    registry_path: Path,
    oof_path: Path,
) -> None:
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    oof_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with oof_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def _main() -> None:
    from scripts.labeling.label_dataset import load_label_dataset

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embedding-cache",
        type=Path,
        default=Path("data/processed/multilingual-e5-small.npz"),
    )
    parser.add_argument("--registry", type=Path, default=Path("reports/model_candidates.json"))
    parser.add_argument("--oof", type=Path, default=Path("reports/model_candidate_oof.csv"))
    args = parser.parse_args()

    rows, _ = load_label_dataset()
    embeddings = load_cached_embeddings(args.embedding_cache, rows)
    if embeddings is None:
        raise SystemExit(
            "현재 데이터와 맞는 임베딩 캐시가 없습니다. embeddings 모듈로 먼저 생성하세요."
        )
    registry, records = run_candidate_ensemble(rows, embeddings)
    write_reports(registry, records, registry_path=args.registry, oof_path=args.oof)
    for key, result in [
        *registry["candidates"].items(),
        *registry["ensembles"].items(),
    ]:
        summary = result["summary"]
        print(
            f"{result['name']}: macro F1 {summary['macro_f1']['fold_mean']:.3f}, "
            f"검토 recall {summary['review_recall']['fold_mean']:.3f}"
        )
    print(json.dumps(registry["agreement"], ensure_ascii=False))


if __name__ == "__main__":
    _main()
