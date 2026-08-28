"""한국어 의존구문 요약 feature를 문자 TF-IDF 기준선에 한 번만 통제 결합한다."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from scripts.labeling.label_dataset import get_model_text


DEPENDENCY_GROUPS: tuple[tuple[str, frozenset[str]], ...] = (
    ("주어", frozenset({"nsubj", "csubj"})),
    ("목적어", frozenset({"obj", "iobj"})),
    ("수식어", frozenset({"amod", "advmod", "nmod", "obl"})),
    ("절", frozenset({"acl", "advcl", "ccomp", "xcomp"})),
    ("접속", frozenset({"conj", "cc"})),
)
DEPENDENCY_FEATURE_NAMES: tuple[str, ...] = tuple(
    name
    for group, _ in DEPENDENCY_GROUPS
    for name in (f"{group}_개수", f"{group}_비율")
) + ("서술어_개수", "서술어_비율")
STANZA_PIPELINE_OPTIONS = {"depparse_min_length_to_batch_separately": 150}


def dependency_features_from_doc(doc: Any) -> list[float]:
    """Stanza 문서에서 6개 해석 단위의 개수와 토큰 대비 비율을 만든다."""
    words = [word for sentence in doc.sentences for word in sentence.words]
    total = max(len(words), 1)
    features: list[float] = []
    for _, relations in DEPENDENCY_GROUPS:
        count = sum(word.deprel.split(":", 1)[0] in relations for word in words)
        features.extend((float(count), count / total))
    predicate_count = sum(word.upos in {"VERB", "AUX"} for word in words)
    features.extend((float(predicate_count), predicate_count / total))
    return features


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_cache(path: Path) -> dict[str, dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def attach_dependency_features(
    rows: Sequence[dict[str, Any]], pipeline: Any, cache_path: Path
) -> list[dict[str, Any]]:
    """변하지 않은 문장은 캐시하고, 나머지만 파싱한다."""
    cache = _load_cache(cache_path)
    pending = []
    for row in rows:
        uid, text = row["requirement_uid"], get_model_text(row)
        digest = _text_hash(text)
        cached = cache.get(uid)
        if not cached or cached.get("text_sha256") != digest:
            pending.append((uid, text, digest))
    for start in range(0, len(pending), 32):
        batch = pending[start : start + 32]
        docs = pipeline.bulk_process([text for _, text, _ in batch])
        for (uid, _, digest), doc in zip(batch, docs):
            cache[uid] = {
                "text_sha256": digest,
                "features": dependency_features_from_doc(doc),
            }
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    return [
        {**row, "dependency_features": cache[row["requirement_uid"]]["features"]}
        for row in rows
    ]


def _dependency_coefficients(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    import numpy as np

    from scripts.evaluation.baselines import CHAR_DEPENDENCY_BALANCED, _fit_pipeline
    from scripts.evaluation.folds import make_lodo_folds

    by_feature = {name: [] for name in DEPENDENCY_FEATURE_NAMES}
    by_class: dict[str, dict[str, list[float]]] = {}
    width = len(DEPENDENCY_FEATURE_NAMES)
    for fold in make_lodo_folds(rows):
        fit_rows, _, _ = fold.split(rows)
        fitted = _fit_pipeline(CHAR_DEPENDENCY_BALANCED, fit_rows)
        classifier = fitted.named_steps["clf"]
        block = classifier.coef_[:, -width:]
        for class_index, label in enumerate(classifier.classes_):
            target = by_class.setdefault(
                str(label), {name: [] for name in DEPENDENCY_FEATURE_NAMES}
            )
            for feature_index, name in enumerate(DEPENDENCY_FEATURE_NAMES):
                value = float(block[class_index, feature_index])
                target[name].append(value)
                by_feature[name].append(abs(value))
    return {
        "mean_absolute": {
            name: float(np.mean(values)) for name, values in by_feature.items()
        },
        "mean_signed_by_class": {
            label: {name: float(np.mean(values)) for name, values in features.items()}
            for label, features in by_class.items()
        },
    }


def dependency_group_eta_squared(
    rows: Sequence[dict[str, Any]], group_field: str
) -> dict[str, Any]:
    """각 feature 분산 중 문서나 라벨 집단 평균 차이가 설명하는 비율."""
    import numpy as np

    matrix = np.asarray([row["dependency_features"] for row in rows], dtype=float)
    groups = np.asarray([row[group_field] for row in rows], dtype=object)
    center = matrix.mean(axis=0)
    total = ((matrix - center) ** 2).sum(axis=0)
    between = np.zeros(matrix.shape[1])
    for group in set(groups):
        subset = matrix[groups == group]
        between += len(subset) * (subset.mean(axis=0) - center) ** 2
    values = np.divide(between, total, out=np.zeros_like(total), where=total > 0)
    return {
        "mean": float(values.mean()),
        "by_feature": dict(zip(DEPENDENCY_FEATURE_NAMES, map(float, values))),
    }


def evaluate_dependency_features(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    from scripts.evaluation.baselines import (
        CHAR_BALANCED,
        CHAR_DEPENDENCY_BALANCED,
        run_lodo,
        summarize,
    )

    baseline_results = run_lodo(rows, CHAR_BALANCED)
    variant_results = run_lodo(rows, CHAR_DEPENDENCY_BALANCED)
    deltas = [
        variant.macro_f1 - baseline.macro_f1
        for baseline, variant in zip(baseline_results, variant_results)
    ]
    return {
        "feature_names": DEPENDENCY_FEATURE_NAMES,
        "baseline": {
            "metrics": summarize(baseline_results),
            "folds": [asdict(result) for result in baseline_results],
        },
        "variant": {
            "metrics": summarize(variant_results),
            "folds": [asdict(result) for result in variant_results],
        },
        "comparison": {
            "mean_delta": sum(deltas) / len(deltas),
            "min_delta": min(deltas),
            "max_delta": max(deltas),
            "wins": sum(delta > 0 for delta in deltas),
            "per_fold": [
                (baseline.test_document, baseline.macro_f1, variant.macro_f1)
                for baseline, variant in zip(baseline_results, variant_results)
            ],
        },
        "coefficients": _dependency_coefficients(rows),
        "distribution_eta_squared": {
            "document": dependency_group_eta_squared(rows, "document_id"),
            "label": dependency_group_eta_squared(rows, "primary_action"),
        },
    }


def _stanza_pipeline(model_dir: Path) -> Any:
    try:
        import stanza
        from stanza.pipeline.core import DownloadMethod
    except ImportError as exc:
        raise RuntimeError("실행에는 stanza 패키지가 필요합니다") from exc
    return stanza.Pipeline(
        lang="ko",
        package="gsd",
        processors="tokenize,pos,lemma,depparse",
        dir=str(model_dir),
        download_method=DownloadMethod.NONE,
        use_gpu=False,
        verbose=False,
        **STANZA_PIPELINE_OPTIONS,
    )


def _main() -> None:
    from scripts.labeling.label_dataset import load_label_dataset

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument(
        "--cache", type=Path, default=Path("data/processed/dependency_features.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed/dependency_results.json")
    )
    args = parser.parse_args()
    rows, _ = load_label_dataset()
    pipeline = _stanza_pipeline(args.model_dir)
    enriched = attach_dependency_features(rows, pipeline, args.cache)
    del pipeline
    gc.collect()
    result = evaluate_dependency_features(enriched)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    baseline = result["baseline"]["metrics"]
    variant = result["variant"]["metrics"]
    delta = result["comparison"]
    print(f"기준선 macro F1 {baseline['macro_f1']['fold_mean']:.3f}")
    print(f"의존구문 macro F1 {variant['macro_f1']['fold_mean']:.3f}")
    print(
        f"차이 {delta['mean_delta']:+.3f} "
        f"({delta['min_delta']:+.3f}~{delta['max_delta']:+.3f}), "
        f"우세 {delta['wins']}/10"
    )


if __name__ == "__main__":
    _main()
