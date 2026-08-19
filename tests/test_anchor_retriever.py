import pytest

from scripts.labeling.anchor_retriever import PureTfidfAnchorRetriever, retriever_config


def _anchor(uid, doc, name, text, action):
    return {
        "requirement_uid": uid,
        "document_id": doc,
        "requirement_name": name,
        "raw_requirement_text": text,
        "primary_action": action,
        "reasoning": f"{action} 판정 사유",
    }


POOL = [
    _anchor(
        "doc_a:R-1", "doc_a", "상주 인력",
        "사업기간 동안 상주 인력 2명을 투입한다.", "견적반영",
    ),
    _anchor(
        "doc_b:R-1", "doc_b", "무상 추가개발",
        "검수 이후 발주기관이 요구하는 기능은 무상으로 추가 개발한다.", "계약·질의검토",
    ),
    _anchor(
        "doc_c:R-1", "doc_c", "주간 보고",
        "수행사는 주간 진척 보고서를 제출한다.", "통상수용",
    ),
    _anchor(
        "doc_a:R-2", "doc_a", "무상 하자보수",
        "하자보수 기간 중 발생하는 결함은 무상으로 조치한다.", "계약·질의검토",
    ),
]


def _target(doc="doc_z", name="무상 추가개발", text="발주기관 요구 기능을 무상으로 추가 개발한다."):
    return {
        "requirement_uid": f"{doc}:T-1",
        "document_id": doc,
        "requirement_name": name,
        "raw_requirement_text": text,
    }


def test_char_ngrams_are_part_of_the_representation():
    # 결정 9와 PROJECT_DIRECTION 9.2: 어절만으로는 띄어쓰기 변형을 놓친다.
    config = retriever_config()
    assert config["char_ngram_range"] == [3, 4]
    assert config["word_ngram_range"] == [1, 2]

    retriever = PureTfidfAnchorRetriever(POOL)
    assert retriever.anchor_matrix.shape[1] == (
        retriever.anchor_word_matrix.shape[1] + retriever.anchor_char_matrix.shape[1]
    )


def test_spacing_variant_still_retrieves_the_right_anchor():
    """'무상으로' vs '무상 으로'처럼 띄어쓰기가 달라도 문자 n-gram이 잡아낸다."""
    retriever = PureTfidfAnchorRetriever(POOL)
    target = _target(text="발주기관이요구하는기능은무상으로추가개발한다.")

    top = retriever.get_top_k_anchors(target, top_k=1)

    assert top, "띄어쓰기가 붕괴된 입력에서 앵커를 하나도 찾지 못했다"
    assert top[0][0]["requirement_uid"] == "doc_b:R-1"


def test_same_document_anchors_are_masked():
    retriever = PureTfidfAnchorRetriever(POOL)
    target = _target(doc="doc_a", name="상주 인력", text="상주 인력 2명을 투입한다.")

    results = retriever.get_top_k_anchors(target, top_k=4)

    assert results
    assert all(anchor["document_id"] != "doc_a" for anchor, _ in results)


def test_stratified_retrieval_returns_one_anchor_per_label():
    retriever = PureTfidfAnchorRetriever(POOL)

    results = retriever.get_stratified_top_k_anchors(_target())

    labels = [anchor["primary_action"] for anchor, _ in results]
    assert sorted(labels) == sorted(["통상수용", "견적반영", "계약·질의검토"])


def test_stratified_retrieval_respects_document_masking():
    retriever = PureTfidfAnchorRetriever(POOL)
    target = _target(doc="doc_b")

    results = retriever.get_stratified_top_k_anchors(target)

    assert all(anchor["document_id"] != "doc_b" for anchor, _ in results)


def test_empty_pool_returns_no_anchors():
    retriever = PureTfidfAnchorRetriever([])

    assert retriever.get_top_k_anchors(_target()) == []
    assert retriever.get_stratified_top_k_anchors(_target()) == []


def test_retrieve_exposes_similarity_and_overlap_terms():
    retriever = PureTfidfAnchorRetriever(POOL)

    rendered = retriever.retrieve(_target(), strategy="stratified")

    assert rendered
    for item in rendered:
        assert 0.0 < item["similarity"] <= 1.0
        assert item["primary_action"] in {"통상수용", "견적반영", "계약·질의검토"}
    matched = next(
        item for item in rendered if item["requirement_uid"] == "doc_b:R-1"
    )
    # 어절 1~2gram이라 '무상으로 추가'처럼 결합형으로 나올 수 있다.
    assert any("무상" in term for term in matched["overlap_terms"])


def test_retrieve_rejects_unknown_strategy():
    retriever = PureTfidfAnchorRetriever(POOL)

    with pytest.raises(ValueError, match="지원하지 않는"):
        retriever.retrieve(_target(), strategy="mmr")
