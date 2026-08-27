from unittest.mock import patch

import numpy as np

from scripts.evaluation import classical_search
from scripts.evaluation.baselines import LABELS
from scripts.evaluation.classical_search import (
    FASTTEXT_CANDIDATES,
    FastTextSpec,
    LOGISTIC_CANDIDATES,
    NBSVMClassifier,
    NBSVMSpec,
    fit_nbsvm,
    fit_fasttext,
    _predict_fasttext,
    run_tuned_logistic_lodo,
)
from scripts.evaluation.folds import make_lodo_folds
from scripts.labeling.label_dataset import load_label_dataset


def test_nbsvm_predicts_all_training_classes_and_explains_features():
    texts = [
        "무상 추가 수행 책임",
        "발주기관 요청 무상 조치",
        "API 연계 구축",
        "시스템 인터페이스 개발",
        "월간 보고 관리",
        "지침 준수 보고",
    ]
    labels = [LABELS[2], LABELS[2], LABELS[1], LABELS[1], LABELS[0], LABELS[0]]
    pipeline = fit_nbsvm(NBSVMSpec(analyzer="word", ngram_range=(1, 2), min_df=1), texts, labels)

    assert set(pipeline.predict(texts)) == set(LABELS)
    explanation = pipeline.named_steps["clf"].explain(
        pipeline.named_steps["vectorizer"].transform([texts[0]]),
        LABELS[2],
        pipeline.named_steps["vectorizer"].get_feature_names_out(),
    )
    assert explanation
    assert all(isinstance(feature, str) and np.isfinite(score) for feature, score in explanation)


def test_tuned_logistic_never_uses_the_test_document_for_selection():
    rows, _ = load_label_dataset()
    calls = []
    original = classical_search.select_logistic

    def spy(fit_rows, validation_rows, candidates=LOGISTIC_CANDIDATES):
        calls.append(
            (
                {row["requirement_uid"] for row in fit_rows},
                {row["requirement_uid"] for row in validation_rows},
            )
        )
        return original(fit_rows, validation_rows, candidates[:2])

    with patch.object(classical_search, "select_logistic", side_effect=spy):
        run = run_tuned_logistic_lodo(rows, candidates=LOGISTIC_CANDIDATES[:2])

    assert len(run.fold_results) == len(calls) == 10
    assert len(run.predictions) == 924
    for fold, (fit_uids, validation_uids) in zip(make_lodo_folds(rows), calls):
        expected_fit, expected_validation, test_rows = fold.split(rows)
        test_uids = {row["requirement_uid"] for row in test_rows}
        assert fit_uids == {row["requirement_uid"] for row in expected_fit}
        assert validation_uids == {
            row["requirement_uid"] for row in expected_validation
        }
        assert not test_uids.intersection(fit_uids | validation_uids)


def test_fasttext_candidate_grid_is_small_and_deterministic():
    assert 1 <= len(FASTTEXT_CANDIDATES) <= 4
    assert len({candidate.key for candidate in FASTTEXT_CANDIDATES}) == len(
        FASTTEXT_CANDIDATES
    )
    assert all(candidate.epoch > 0 for candidate in FASTTEXT_CANDIDATES)


def test_fasttext_smoke_returns_three_class_scores():
    rows = [
        {"raw_requirement_text": f"{label} 대표 표현 {i}", "primary_action": label}
        for label in LABELS
        for i in range(2)
    ]
    model = fit_fasttext(
        FastTextSpec(word_ngrams=1, minn=2, maxn=4, epoch=3, dim=10), rows
    )
    predictions, scores = _predict_fasttext(model, rows)

    assert len(predictions) == len(rows)
    assert set(predictions).issubset(LABELS)
    assert scores.shape == (len(rows), len(LABELS))


def test_nbsvm_rejects_row_mismatch():
    model = NBSVMClassifier()
    with np.testing.assert_raises_regex(ValueError, "행 수"):
        model.fit(np.ones((2, 2)), [LABELS[0]])
