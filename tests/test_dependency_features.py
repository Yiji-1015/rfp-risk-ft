from types import SimpleNamespace

from scripts.evaluation.dependency_features import (
    DEPENDENCY_FEATURE_NAMES,
    STANZA_PIPELINE_OPTIONS,
    dependency_group_eta_squared,
    dependency_features_from_doc,
)
from scripts.evaluation.baselines import CHAR_DEPENDENCY_BALANCED


def test_dependency_features_are_small_interpretable_counts_and_ratios():
    words = [
        SimpleNamespace(deprel="nsubj", upos="NOUN"),
        SimpleNamespace(deprel="obj", upos="NOUN"),
        SimpleNamespace(deprel="advmod", upos="ADV"),
        SimpleNamespace(deprel="conj", upos="VERB"),
        SimpleNamespace(deprel="root", upos="VERB"),
    ]
    doc = SimpleNamespace(sentences=[SimpleNamespace(words=words)])

    features = dependency_features_from_doc(doc)

    assert len(features) == len(DEPENDENCY_FEATURE_NAMES) == 12
    assert features[DEPENDENCY_FEATURE_NAMES.index("주어_개수")] == 1.0
    assert features[DEPENDENCY_FEATURE_NAMES.index("목적어_비율")] == 0.2
    assert features[DEPENDENCY_FEATURE_NAMES.index("서술어_개수")] == 2.0


def test_dependency_features_handle_empty_parse():
    doc = SimpleNamespace(sentences=[])

    assert dependency_features_from_doc(doc) == [0.0] * len(DEPENDENCY_FEATURE_NAMES)


def test_dependency_block_combines_with_tfidf():
    rows = [
        {"raw_requirement_text": text, "dependency_features": [float(i)] * 12}
        for i, text in enumerate(
            ["저장해야 한다", "저장해야 한다", "비용을 반영", "비용을 반영", "계약 검토", "계약 검토"]
        )
    ]
    labels = ["통상수용", "통상수용", "견적반영", "견적반영", "계약·질의검토", "계약·질의검토"]

    model = CHAR_DEPENDENCY_BALANCED.build().fit(rows, labels)

    assert len(model.predict(rows)) == len(rows)


def test_long_requirements_are_parsed_in_separate_batches():
    assert STANZA_PIPELINE_OPTIONS["depparse_min_length_to_batch_separately"] == 150


def test_group_eta_squared_is_one_for_perfectly_separated_groups():
    rows = [
        {"document_id": "a", "dependency_features": [0.0] * 12},
        {"document_id": "a", "dependency_features": [0.0] * 12},
        {"document_id": "b", "dependency_features": [1.0] * 12},
        {"document_id": "b", "dependency_features": [1.0] * 12},
    ]

    result = dependency_group_eta_squared(rows, "document_id")

    assert result["mean"] == 1.0
