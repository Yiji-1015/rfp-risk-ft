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
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer

from scripts.evaluation.baselines import (
    RANDOM_STATE,
    FoldResult,
    _fold_result_from_predictions,
    summarize,
)
from scripts.evaluation.duplication import DEFAULT_THRESHOLD
from scripts.evaluation.folds import _repeat_flags, make_lodo_folds

MODEL_ID = "intfloat/multilingual-e5-small"
MODEL_REVISION = "d2648e288f5fe1641aeab663a7fa6d1f0d1daff2"
INPUT_PREFIX = "query: "
MAX_LENGTH = 512
DEFAULT_BATCH_SIZE = 16
PIPELINE_VERSION = "attention-mask-mean+l2-v1"


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
        digest.update(row["raw_requirement_text"].encode("utf-8"))
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
    texts = [INPUT_PREFIX + row["raw_requirement_text"] for row in rows]
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


if __name__ == "__main__":
    _main()
