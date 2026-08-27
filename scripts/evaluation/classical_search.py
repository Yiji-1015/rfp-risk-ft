"""설명 가능한 전통 텍스트 분류기를 같은 8/1/1 LODO에서 비교한다.

평가 문서는 파라미터 선택에 쓰지 않는다. 각 outer fold의 학습 8문서로 후보를
학습하고 회전 검증 1문서에서 macro F1을 고른 뒤, 평가 1문서를 한 번만 본다.
"""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.sparse import csr_matrix
from scipy.special import softmax
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, fbeta_score
from sklearn.pipeline import Pipeline

from scripts.evaluation.baselines import (
    CHAR_BALANCED,
    LABELS,
    REVIEW_LABEL,
    RANDOM_STATE,
    FoldResult,
    ModelSpec,
    _fit_pipeline,
    _fold_result_from_predictions,
    _model_input,
    run_lodo,
    summarize,
)
from scripts.evaluation.duplication import DEFAULT_THRESHOLD
from scripts.evaluation.folds import _repeat_flags, make_lodo_folds


LOGISTIC_CANDIDATES: tuple[ModelSpec, ...] = tuple(
    ModelSpec(
        name=f"char {lo}-{hi}gram min_df={min_df} C={C:g}",
        ngram_range=(lo, hi),
        min_df=min_df,
        C=C,
    )
    for lo, hi in ((2, 4), (3, 4), (3, 5))
    for min_df in (1, 2, 3)
    for C in (0.3, 1.0, 3.0)
)


@dataclass(frozen=True)
class NBSVMSpec:
    analyzer: str = "char_wb"
    ngram_range: tuple[int, int] = (3, 4)
    min_df: int = 2
    C: float = 1.0

    @property
    def key(self) -> str:
        lo, hi = self.ngram_range
        return f"{self.analyzer} {lo}-{hi}gram min_df={self.min_df} C={self.C:g}"


NBSVM_CANDIDATES: tuple[NBSVMSpec, ...] = tuple(
    NBSVMSpec(analyzer=analyzer, ngram_range=ngram_range, C=C)
    for analyzer, ngram_range in (("word", (1, 2)), ("char_wb", (3, 4)))
    for C in (0.3, 1.0, 3.0)
)


@dataclass(frozen=True)
class FastTextSpec:
    word_ngrams: int
    minn: int
    maxn: int
    epoch: int = 25
    lr: float = 0.1
    dim: int = 100

    @property
    def key(self) -> str:
        return (
            f"wordNgrams={self.word_ngrams} char={self.minn}-{self.maxn} "
            f"epoch={self.epoch} lr={self.lr:g}"
        )


FASTTEXT_CANDIDATES: tuple[FastTextSpec, ...] = (
    FastTextSpec(word_ngrams=1, minn=2, maxn=5),
    FastTextSpec(word_ngrams=2, minn=2, maxn=5),
    FastTextSpec(word_ngrams=2, minn=3, maxn=6),
)


@dataclass
class EvaluationRun:
    name: str
    fold_results: list[FoldResult]
    predictions: list[dict[str, Any]]
    selected_parameters: dict[str, str]


class NBSVMClassifier(ClassifierMixin, BaseEstimator):
    """클래스별 NB log-count ratio를 쓰는 one-vs-rest Logistic."""

    def __init__(self, C: float = 1.0):
        self.C = C

    def fit(self, X: Any, y: Sequence[str]) -> NBSVMClassifier:
        X = csr_matrix(X)
        y_array = np.asarray(y, dtype=object)
        if X.shape[0] != len(y_array):
            raise ValueError("특징 행 수와 라벨 행 수가 다릅니다")
        self.classes_ = np.array([label for label in LABELS if label in set(y_array)])
        if len(self.classes_) < 2:
            raise ValueError("NB-SVM 학습에는 두 클래스 이상이 필요합니다")
        self.ratios_: list[np.ndarray] = []
        self.models_: list[LogisticRegression] = []
        for label in self.classes_:
            positive = y_array == label
            p = np.asarray(X[positive].sum(axis=0)).ravel() + 1.0
            q = np.asarray(X[~positive].sum(axis=0)).ravel() + 1.0
            ratio = np.log(p / p.sum()) - np.log(q / q.sum())
            model = LogisticRegression(
                C=self.C,
                class_weight="balanced",
                max_iter=2000,
                random_state=RANDOM_STATE,
            )
            model.fit(X.multiply(ratio), positive.astype(int))
            self.ratios_.append(ratio)
            self.models_.append(model)
        return self

    def decision_function(self, X: Any) -> np.ndarray:
        X = csr_matrix(X)
        return np.column_stack(
            [
                model.decision_function(X.multiply(ratio))
                for model, ratio in zip(self.models_, self.ratios_)
            ]
        )

    def predict(self, X: Any) -> np.ndarray:
        return self.classes_[np.argmax(self.decision_function(X), axis=1)]

    def predict_proba(self, X: Any) -> np.ndarray:
        """비교·정렬용 softmax 점수다. 보정된 확률로 해석하지 않는다."""
        return softmax(self.decision_function(X), axis=1)

    def explain(
        self,
        X: Any,
        label: str,
        feature_names: Sequence[str],
        top_n: int = 5,
    ) -> list[tuple[str, float]]:
        class_index = list(self.classes_).index(label)
        row = csr_matrix(X)[0]
        weights = self.ratios_[class_index] * self.models_[class_index].coef_[0]
        scores = row.multiply(weights).toarray().ravel()
        indices = np.flatnonzero(scores)
        ranked = indices[np.argsort(scores[indices])[::-1]][:top_n]
        return [(str(feature_names[i]), float(scores[i])) for i in ranked]


def fit_nbsvm(
    spec: NBSVMSpec, texts: Sequence[str], labels: Sequence[str]
) -> Pipeline:
    pipeline = Pipeline(
        [
            (
                "vectorizer",
                CountVectorizer(
                    analyzer=spec.analyzer,
                    ngram_range=spec.ngram_range,
                    min_df=spec.min_df,
                    binary=True,
                ),
            ),
            ("clf", NBSVMClassifier(C=spec.C)),
        ]
    )
    pipeline.fit(texts, labels)
    return pipeline


def _selection_rank(
    gold: Sequence[str], pred: Sequence[str], index: int
) -> tuple[float, float, int]:
    macro = f1_score(gold, pred, labels=LABELS, average="macro", zero_division=0)
    review_f2 = fbeta_score(
        gold,
        pred,
        labels=[REVIEW_LABEL],
        average="macro",
        beta=2,
        zero_division=0,
    )
    return float(macro), float(review_f2), -index


def select_logistic(
    fit_rows: Sequence[dict[str, Any]],
    validation_rows: Sequence[dict[str, Any]],
    candidates: Sequence[ModelSpec] = LOGISTIC_CANDIDATES,
) -> ModelSpec:
    gold = [row["primary_action"] for row in validation_rows]
    ranks = []
    for index, candidate in enumerate(candidates):
        model = _fit_pipeline(candidate, fit_rows)
        pred = model.predict(_model_input(candidate, validation_rows))
        ranks.append(_selection_rank(gold, pred, index))
    return candidates[max(range(len(candidates)), key=ranks.__getitem__)]


def select_nbsvm(
    fit_rows: Sequence[dict[str, Any]],
    validation_rows: Sequence[dict[str, Any]],
    candidates: Sequence[NBSVMSpec] = NBSVM_CANDIDATES,
) -> NBSVMSpec:
    fit_texts = [row["raw_requirement_text"] for row in fit_rows]
    labels = [row["primary_action"] for row in fit_rows]
    validation_texts = [row["raw_requirement_text"] for row in validation_rows]
    gold = [row["primary_action"] for row in validation_rows]
    ranks = []
    for index, candidate in enumerate(candidates):
        pred = fit_nbsvm(candidate, fit_texts, labels).predict(validation_texts)
        ranks.append(_selection_rank(gold, pred, index))
    return candidates[max(range(len(candidates)), key=ranks.__getitem__)]


def _linear_explanations(model: Pipeline, texts: Sequence[str], pred: Sequence[str]) -> list[str]:
    vectorizer = model.named_steps["features"]
    classifier = model.named_steps["clf"]
    matrix = vectorizer.transform(texts)
    names = vectorizer.get_feature_names_out()
    class_index = {label: i for i, label in enumerate(classifier.classes_)}
    explanations = []
    for row_index, label in enumerate(pred):
        scores = matrix[row_index].multiply(classifier.coef_[class_index[label]]).toarray().ravel()
        indices = np.flatnonzero(scores)
        ranked = indices[np.argsort(scores[indices])[::-1]][:5]
        explanations.append(" | ".join(f"{names[i]}:{scores[i]:.4f}" for i in ranked))
    return explanations


def _nbsvm_explanations(model: Pipeline, texts: Sequence[str], pred: Sequence[str]) -> list[str]:
    vectorizer = model.named_steps["vectorizer"]
    classifier = model.named_steps["clf"]
    matrix = vectorizer.transform(texts)
    names = vectorizer.get_feature_names_out()
    explanations = []
    for i, label in enumerate(pred):
        ranked = classifier.explain(matrix[i], label, names)
        explanations.append(
            " | ".join(f"{feature}:{score:.4f}" for feature, score in ranked)
            or "활성 feature 없음"
        )
    return explanations


def _prediction_rows(
    model_name: str,
    fold_index: int,
    test_rows: Sequence[dict[str, Any]],
    pred: Sequence[str],
    scores: np.ndarray,
    explanations: Sequence[str],
    confidence_kind: str,
) -> list[dict[str, Any]]:
    order = np.sort(scores, axis=1)
    margins = order[:, -1] - order[:, -2]
    return [
        {
            "model": model_name,
            "fold": fold_index,
            "test_document": row["document_id"],
            "requirement_uid": row["requirement_uid"],
            "text": row["raw_requirement_text"],
            "gold": row["primary_action"],
            "pred": label,
            "correct": row["primary_action"] == label,
            "score_margin": float(margin),
            "confidence_kind": confidence_kind,
            "top_features": explanation,
        }
        for row, label, margin, explanation in zip(test_rows, pred, margins, explanations)
    ]


def run_tuned_logistic_lodo(
    rows: Sequence[dict[str, Any]],
    *,
    candidates: Sequence[ModelSpec] = LOGISTIC_CANDIDATES,
    repeat_threshold: float = DEFAULT_THRESHOLD,
) -> EvaluationRun:
    results, predictions, selected = [], [], {}
    for fold in make_lodo_folds(rows):
        fit_rows, validation_rows, test_rows = fold.split(rows)
        spec = select_logistic(fit_rows, validation_rows, candidates)
        model = _fit_pipeline(spec, fit_rows)
        texts = [row["raw_requirement_text"] for row in test_rows]
        pred = list(model.predict(texts))
        probabilities = model.predict_proba(texts)
        flags = _repeat_flags(fold, rows, repeat_threshold)
        results.append(
            _fold_result_from_predictions(
                fold,
                rows,
                test_rows,
                pred,
                train_size=len(fit_rows),
                repeat_flags=flags,
                repeat_threshold=repeat_threshold,
            )
        )
        predictions.extend(
            _prediction_rows(
                "Tuned char Logistic",
                fold.index,
                test_rows,
                pred,
                probabilities,
                _linear_explanations(model, texts, pred),
                "logistic_probability_margin",
            )
        )
        selected[fold.test_document] = spec.name
    return EvaluationRun("Tuned char Logistic", results, predictions, selected)


def run_nbsvm_lodo(
    rows: Sequence[dict[str, Any]],
    *,
    candidates: Sequence[NBSVMSpec] = NBSVM_CANDIDATES,
    repeat_threshold: float = DEFAULT_THRESHOLD,
) -> EvaluationRun:
    results, predictions, selected = [], [], {}
    for fold in make_lodo_folds(rows):
        fit_rows, validation_rows, test_rows = fold.split(rows)
        spec = select_nbsvm(fit_rows, validation_rows, candidates)
        fit_texts = [row["raw_requirement_text"] for row in fit_rows]
        labels = [row["primary_action"] for row in fit_rows]
        test_texts = [row["raw_requirement_text"] for row in test_rows]
        model = fit_nbsvm(spec, fit_texts, labels)
        pred = list(model.predict(test_texts))
        scores = model.named_steps["clf"].decision_function(
            model.named_steps["vectorizer"].transform(test_texts)
        )
        results.append(
            _fold_result_from_predictions(
                fold,
                rows,
                test_rows,
                pred,
                train_size=len(fit_rows),
                repeat_flags=_repeat_flags(fold, rows, repeat_threshold),
                repeat_threshold=repeat_threshold,
            )
        )
        predictions.extend(
            _prediction_rows(
                "NB-SVM",
                fold.index,
                test_rows,
                pred,
                scores,
                _nbsvm_explanations(model, test_texts, pred),
                "ovr_decision_margin",
            )
        )
        selected[fold.test_document] = spec.key
    return EvaluationRun("NB-SVM", results, predictions, selected)


def _fasttext_module():
    try:
        import fasttext
    except ImportError as exc:
        raise RuntimeError("fastText 실행에는 fasttext-wheel 패키지가 필요합니다") from exc
    return fasttext


def _balanced_fasttext_lines(rows: Sequence[dict[str, Any]]) -> list[str]:
    by_label = {label: [] for label in LABELS}
    for row in rows:
        by_label[row["primary_action"]].append(row["raw_requirement_text"])
    target = max(map(len, by_label.values()))
    lines = []
    for label_index, label in enumerate(LABELS):
        texts = by_label[label]
        for index in range(target):
            text = " ".join(texts[index % len(texts)].split())
            lines.append(f"__label__{label_index} {text}")
    return lines


def fit_fasttext(spec: FastTextSpec, rows: Sequence[dict[str, Any]]):
    fasttext = _fasttext_module()
    with tempfile.TemporaryDirectory(prefix="rfp-fasttext-") as temp_dir:
        path = Path(temp_dir) / "train.txt"
        path.write_text("\n".join(_balanced_fasttext_lines(rows)), encoding="utf-8")
        return fasttext.train_supervised(
            input=str(path),
            lr=spec.lr,
            epoch=spec.epoch,
            dim=spec.dim,
            wordNgrams=spec.word_ngrams,
            minn=spec.minn,
            maxn=spec.maxn,
            loss="softmax",
            thread=1,
            verbose=0,
        )


def _predict_fasttext(model: Any, rows: Sequence[dict[str, Any]]) -> tuple[list[str], np.ndarray]:
    predictions, scores = [], []
    for row in rows:
        text = " ".join(row["raw_requirement_text"].split())
        # fasttext-wheel 0.9.2의 단일 문자열 경로는 NumPy 2에서 copy=False로
        # 실패한다. 공개 batch 경로는 같은 예측을 호환되게 반환한다.
        label_batch, probability_batch = model.predict([text], k=len(LABELS))
        labels, probabilities = label_batch[0], probability_batch[0]
        by_label = {
            LABELS[int(label.removeprefix("__label__"))]: float(probability)
            for label, probability in zip(labels, probabilities)
        }
        predictions.append(max(by_label, key=by_label.get))
        scores.append([by_label.get(label, 0.0) for label in LABELS])
    return predictions, np.asarray(scores)


def select_fasttext(
    fit_rows: Sequence[dict[str, Any]],
    validation_rows: Sequence[dict[str, Any]],
    candidates: Sequence[FastTextSpec] = FASTTEXT_CANDIDATES,
) -> FastTextSpec:
    gold = [row["primary_action"] for row in validation_rows]
    ranks = []
    for index, candidate in enumerate(candidates):
        pred, _ = _predict_fasttext(fit_fasttext(candidate, fit_rows), validation_rows)
        ranks.append(_selection_rank(gold, pred, index))
    return candidates[max(range(len(candidates)), key=ranks.__getitem__)]


def run_fasttext_lodo(
    rows: Sequence[dict[str, Any]],
    *,
    candidates: Sequence[FastTextSpec] = FASTTEXT_CANDIDATES,
    repeat_threshold: float = DEFAULT_THRESHOLD,
) -> EvaluationRun:
    results, predictions, selected = [], [], {}
    for fold in make_lodo_folds(rows):
        fit_rows, validation_rows, test_rows = fold.split(rows)
        spec = select_fasttext(fit_rows, validation_rows, candidates)
        pred, scores = _predict_fasttext(fit_fasttext(spec, fit_rows), test_rows)
        results.append(
            _fold_result_from_predictions(
                fold,
                rows,
                test_rows,
                pred,
                train_size=len(fit_rows),
                repeat_flags=_repeat_flags(fold, rows, repeat_threshold),
                repeat_threshold=repeat_threshold,
            )
        )
        predictions.extend(
            _prediction_rows(
                "fastText",
                fold.index,
                test_rows,
                pred,
                scores,
                [""] * len(test_rows),
                "fasttext_probability_margin",
            )
        )
        selected[fold.test_document] = spec.key
    return EvaluationRun("fastText", results, predictions, selected)


def _write_outputs(runs: Sequence[EvaluationRun], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "classical_search_predictions.csv"
    fields = list(runs[0].predictions[0])
    with prediction_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            writer.writerows(run.predictions)

    summary = {
        run.name: {
            "metrics": summarize(run.fold_results),
            "selected_parameters": run.selected_parameters,
            "folds": [asdict(result) for result in run.fold_results],
        }
        for run in runs
    }
    (output_dir / "classical_search_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _main() -> None:
    from scripts.labeling.label_dataset import load_label_dataset

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()
    rows, _ = load_label_dataset()
    runs = [
        run_tuned_logistic_lodo(rows),
        run_nbsvm_lodo(rows),
        run_fasttext_lodo(rows),
    ]
    _write_outputs(runs, args.output_dir)
    baseline = summarize(run_lodo(rows, CHAR_BALANCED))["macro_f1"]["fold_mean"]
    print(f"기준선 char Logistic {baseline:.3f}")
    for run in runs:
        metrics = summarize(run.fold_results)
        print(
            f"{run.name:<24} macro F1 {metrics['macro_f1']['fold_mean']:.3f} / "
            f"계약 recall {metrics['review_recall']['fold_mean']:.3f}"
        )


if __name__ == "__main__":
    _main()
