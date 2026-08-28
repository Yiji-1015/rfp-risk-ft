"""동결 E5 문장 임베딩을 기존 LODO 선형 분류기와 비교한다.

인코더는 학습하지 않는다. 요구사항마다 독립적으로 384차원 벡터를 한 번 만든 뒤,
기존과 같은 학습 8 / 검증 1 / 평가 1 문서 분할에서 Logistic과 LinearSVC만 학습한다.
모델 파일은 실수로 C 드라이브에 받지 않도록 CLI에서 캐시 경로를 반드시 받는다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import transformers
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer

from scripts.evaluation.baselines import (
    RANDOM_STATE,
    LABELS,
    FoldResult,
    _aligned_probabilities,
    _select_number_features,
    _fold_result_from_predictions,
    summarize,
)
from scripts.evaluation.duplication import DEFAULT_THRESHOLD
from scripts.evaluation.folds import _repeat_flags, make_lodo_folds
from scripts.labeling.label_dataset import get_model_text

MODEL_ID = "intfloat/multilingual-e5-small"
MODEL_REVISION = "d2648e288f5fe1641aeab663a7fa6d1f0d1daff2"
INPUT_PREFIX = "query: "
MAX_LENGTH = 512
DEFAULT_BATCH_SIZE = 16
PIPELINE_VERSION = "attention-mask-mean+l2-v1"
EMBEDDING_WEIGHT_CANDIDATES: tuple[float, ...] = (0.0, 0.05, 0.1, 0.25, 0.5, 1.0)


@dataclass(frozen=True)
class EmbeddingSpec:
    name: str
    classifier: str
    C: float = 1.0
    class_weight: str | dict[str, float] | None = "balanced"

    def build(self):
        if self.classifier == "logistic":
            return LogisticRegression(
                C=self.C,
                class_weight=self.class_weight,
                max_iter=2000,
                random_state=RANDOM_STATE,
            )
        if self.classifier == "svm":
            return LinearSVC(
                C=self.C,
                class_weight=self.class_weight,
                random_state=RANDOM_STATE,
            )
        raise ValueError(f"알 수 없는 분류기: {self.classifier!r}")


E5_LOGISTIC = EmbeddingSpec("E5-small + balanced Logistic", "logistic")
E5_SVM = EmbeddingSpec("E5-small + balanced LinearSVC", "svm")


@dataclass(frozen=True)
class HybridSpec:
    name: str
    include_number_features: bool = False


TFIDF_E5 = HybridSpec("char TF-IDF + E5 검증 weight")
TFIDF_E5_NUMBERS = HybridSpec(
    "char TF-IDF + E5 검증 weight + 숫자 정보",
    include_number_features=True,
)


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """padding 토큰을 빼고 평균낸 뒤 코사인 비교에 맞게 L2 정규화한다."""
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    pooled = (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    return F.normalize(pooled, p=2, dim=1)


def _row_fingerprint(rows: Sequence[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(row["requirement_uid"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(get_model_text(row).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _metadata(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "input_prefix": INPUT_PREFIX,
        "max_length": MAX_LENGTH,
        "pipeline_version": PIPELINE_VERSION,
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "row_count": len(rows),
        "row_fingerprint": _row_fingerprint(rows),
    }


def load_cached_embeddings(
    path: Path, rows: Sequence[dict[str, Any]]
) -> np.ndarray | None:
    """현재 행·모델·풀링·라이브러리와 정확히 맞는 캐시만 반환한다."""
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as cached:
        if json.loads(str(cached["metadata"])) != _metadata(rows):
            return None
        embeddings = cached["embeddings"]
    return embeddings if embeddings.shape[0] == len(rows) else None


def load_or_create_embeddings(
    rows: Sequence[dict[str, Any]],
    *,
    model_cache: Path,
    embedding_cache: Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
    rebuild: bool = False,
) -> np.ndarray:
    """E5를 동결한 채 요구사항을 한 번만 임베딩하고 작은 npz로 캐시한다."""
    if not rebuild and (
        cached := load_cached_embeddings(embedding_cache, rows)
    ) is not None:
        print(f"임베딩 캐시 사용: {embedding_cache}")
        return cached

    model_cache.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, cache_dir=model_cache
    )
    model = AutoModel.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, cache_dir=model_cache
    )
    model.eval()

    batches = []
    texts = [INPUT_PREFIX + get_model_text(row) for row in rows]
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            encoded = tokenizer(
                texts[start : start + batch_size],
                max_length=MAX_LENGTH,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            output = model(**encoded)
            batches.append(mean_pool(output.last_hidden_state, encoded["attention_mask"]).numpy())
            print(f"임베딩 {min(start + batch_size, len(texts))}/{len(texts)}")

    embeddings = np.concatenate(batches).astype(np.float32, copy=False)
    embedding_cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        embedding_cache,
        embeddings=embeddings,
        metadata=json.dumps(_metadata(rows), ensure_ascii=False, sort_keys=True),
    )
    print(f"임베딩 저장: {embedding_cache}")
    return embeddings


def run_embedding_lodo(
    rows: Sequence[dict[str, Any]],
    embeddings: np.ndarray,
    spec: EmbeddingSpec,
    *,
    repeat_threshold: float = DEFAULT_THRESHOLD,
) -> list[FoldResult]:
    """같은 LODO 분할에서 텍스트 대신 미리 만든 임베딩만 사용한다."""
    if embeddings.ndim != 2 or embeddings.shape[0] != len(rows):
        raise ValueError("임베딩 행 수는 데이터 행 수와 같아야 합니다")

    positions = {row["requirement_uid"]: i for i, row in enumerate(rows)}
    results = []
    for fold in make_lodo_folds(rows):
        fit_rows, _, test_rows = fold.split(rows)
        fit_idx = [positions[row["requirement_uid"]] for row in fit_rows]
        test_idx = [positions[row["requirement_uid"]] for row in test_rows]
        classifier = spec.build()
        classifier.fit(
            embeddings[fit_idx],
            [row["primary_action"] for row in fit_rows],
        )
        pred = list(classifier.predict(embeddings[test_idx]))
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
    return results


def _hybrid_matrix(
    text_matrix: Any,
    embeddings: np.ndarray,
    embedding_weight: float,
    number_features: np.ndarray | None,
) -> Any:
    blocks = [text_matrix, csr_matrix(embeddings * embedding_weight)]
    if number_features is not None:
        blocks.append(csr_matrix(number_features))
    return hstack(blocks, format="csr")


def run_hybrid_lodo(
    rows: Sequence[dict[str, Any]],
    embeddings: np.ndarray,
    spec: HybridSpec,
    *,
    weight_candidates: Sequence[float] = EMBEDDING_WEIGHT_CANDIDATES,
    repeat_threshold: float = DEFAULT_THRESHOLD,
) -> list[FoldResult]:
    """검증 문서로 E5 블록 가중치를 고르는 TF-IDF 결합 LODO."""
    if embeddings.ndim != 2 or embeddings.shape[0] != len(rows):
        raise ValueError("임베딩 행 수는 데이터 행 수와 같아야 합니다")
    if not weight_candidates:
        raise ValueError("임베딩 가중치 후보가 하나 이상 필요합니다")

    results = []
    for fold in make_lodo_folds(rows):
        fit_rows, _, test_rows = fold.split(rows)
        pred, _, selected = predict_hybrid_fold(
            rows, embeddings, fold, spec, weight_candidates=weight_candidates
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
                embedding_weight=selected,
            )
        )
    return results


def predict_hybrid_fold(
    rows: Sequence[dict[str, Any]],
    embeddings: np.ndarray,
    fold: Any,
    spec: HybridSpec,
    *,
    weight_candidates: Sequence[float] = EMBEDDING_WEIGHT_CANDIDATES,
) -> tuple[list[str], np.ndarray, float]:
    """fold 하나의 검증 선택, 예측, 정렬 확률을 반환한다."""
    positions = {row["requirement_uid"]: i for i, row in enumerate(rows)}
    fit_rows, validation_rows, test_rows = fold.split(rows)
    fit_idx = [positions[row["requirement_uid"]] for row in fit_rows]
    validation_idx = [positions[row["requirement_uid"]] for row in validation_rows]
    test_idx = [positions[row["requirement_uid"]] for row in test_rows]

    vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 4), min_df=2, sublinear_tf=True
    )
    fit_text = vectorizer.fit_transform(get_model_text(row) for row in fit_rows)
    validation_text = vectorizer.transform(
        get_model_text(row) for row in validation_rows
    )
    test_text = vectorizer.transform(get_model_text(row) for row in test_rows)

    fit_numbers = validation_numbers = test_numbers = None
    if spec.include_number_features:
        scaler = StandardScaler()
        fit_numbers = scaler.fit_transform(_select_number_features(fit_rows))
        validation_numbers = scaler.transform(_select_number_features(validation_rows))
        test_numbers = scaler.transform(_select_number_features(test_rows))

    gold_fit = [row["primary_action"] for row in fit_rows]
    gold_validation = [row["primary_action"] for row in validation_rows]
    ranks = {}
    for weight in weight_candidates:
        classifier = E5_LOGISTIC.build()
        classifier.fit(
            _hybrid_matrix(fit_text, embeddings[fit_idx], weight, fit_numbers), gold_fit
        )
        pred = classifier.predict(
            _hybrid_matrix(
                validation_text, embeddings[validation_idx], weight, validation_numbers
            )
        )
        ranks[weight] = (
            float(
                f1_score(
                    gold_validation,
                    pred,
                    labels=LABELS,
                    average="macro",
                    zero_division=0,
                )
            ),
            -weight,
        )
    selected = max(weight_candidates, key=ranks.__getitem__)

    classifier = E5_LOGISTIC.build()
    fit_matrix = _hybrid_matrix(
        fit_text, embeddings[fit_idx], selected, fit_numbers
    )
    test_matrix = _hybrid_matrix(
        test_text, embeddings[test_idx], selected, test_numbers
    )
    classifier.fit(fit_matrix, gold_fit)
    probabilities = _aligned_probabilities(classifier, test_matrix)
    return [LABELS[i] for i in probabilities.argmax(axis=1)], probabilities, selected


def _main() -> None:
    from scripts.labeling.label_dataset import load_label_dataset

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-cache", required=True, type=Path)
    parser.add_argument(
        "--embedding-cache",
        type=Path,
        default=Path("data/processed/multilingual-e5-small.npz"),
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    rows, meta = load_label_dataset()
    print(f"데이터셋 {meta['dataset_version']} / {meta['row_count']}건")
    embeddings = load_or_create_embeddings(
        rows,
        model_cache=args.model_cache,
        embedding_cache=args.embedding_cache,
        batch_size=args.batch_size,
        rebuild=args.rebuild,
    )

    print(f"\n{'설정':<38}{'macroF1':>9}{'정확도':>9}{'계약prec':>10}{'계약recall':>11}{'계약F1':>9}")
    print("-" * 86)
    for spec in (E5_LOGISTIC, E5_SVM):
        summary = summarize(run_embedding_lodo(rows, embeddings, spec))
        print(
            f"{spec.name:<38}{summary['macro_f1']['fold_mean']:>9.3f}"
            f"{summary['accuracy']['fold_mean']:>9.3f}"
            f"{summary['review_precision']['fold_mean']:>10.3f}"
            f"{summary['review_recall']['fold_mean']:>11.3f}"
            f"{summary['review_f1']['fold_mean']:>9.3f}"
        )
    for spec in (TFIDF_E5, TFIDF_E5_NUMBERS):
        summary = summarize(run_hybrid_lodo(rows, embeddings, spec))
        print(
            f"{spec.name:<38}{summary['macro_f1']['fold_mean']:>9.3f}"
            f"{summary['accuracy']['fold_mean']:>9.3f}"
            f"{summary['review_precision']['fold_mean']:>10.3f}"
            f"{summary['review_recall']['fold_mean']:>11.3f}"
            f"{summary['review_f1']['fold_mean']:>9.3f}"
        )


if __name__ == "__main__":
    _main()
