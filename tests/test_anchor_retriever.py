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


def test_global_retrieval_renders_identically_for_different_targets(monkeypatch):
    """
    결정 29: 고정 앵커는 대상이 달라도 렌더링 결과가 바이트 단위로 같아야 한다.

    같지 않으면 앵커 블록을 system에 올려도 캐시 프리픽스가 매 건 깨져
    캐시가 붙지 않는다. 유사도·공통 어휘가 대상별로 계산되면 여기서 깨진다.
    """
    from scripts.labeling import anchor_retriever
    from scripts.labeling.claude_client import render_anchor_block

    monkeypatch.setattr(
        anchor_retriever,
        "GLOBAL_ANCHORS",
        {
            "통상수용": ["doc_c:R-1"],
            "견적반영": ["doc_a:R-1"],
            "계약·질의검토": ["doc_b:R-1"],
        },
    )
    retriever = PureTfidfAnchorRetriever(POOL)

    first = retriever.retrieve(_target(doc="doc_y"), strategy="global")
    second = retriever.retrieve(
        _target(doc="doc_z", name="상주 인력", text="상주 인력 3명을 배치한다."),
        strategy="global",
    )

    assert [a["requirement_uid"] for a in first] == [a["requirement_uid"] for a in second]
    # 공통 어휘를 계산하면 대상마다 달라지므로 고정 앵커에서는 비운다.
    assert all(a["overlap_terms"] == [] for a in first)
    assert render_anchor_block(
        first, show_retrieval_evidence=False
    ) == render_anchor_block(second, show_retrieval_evidence=False)


def test_global_retrieval_masks_same_document_anchors(monkeypatch):
    """동일 문서 앵커는 고정 전략에서도 차단한다. 같은 라벨의 다음 순위로 대체된다."""
    from scripts.labeling import anchor_retriever

    monkeypatch.setattr(
        anchor_retriever,
        "GLOBAL_ANCHORS",
        {"계약·질의검토": ["doc_b:R-1", "doc_a:R-2"]},
    )
    retriever = PureTfidfAnchorRetriever(POOL)

    rendered = retriever.retrieve(_target(doc="doc_b"), strategy="global")

    assert [a["requirement_uid"] for a in rendered] == ["doc_a:R-2"]

def test_retrieve_rejects_unknown_strategy():
    retriever = PureTfidfAnchorRetriever(POOL)

    with pytest.raises(ValueError, match="지원하지 않는"):
        retriever.retrieve(_target(), strategy="mmr")
