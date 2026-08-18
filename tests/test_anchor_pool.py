import json

import pytest

from scripts.labeling.anchor_pool import AnchorPoolError, load_anchor_pool


def _anchor(**overrides):
    base = {
        "requirement_uid": "doc_a:SFR-001",
        "document_id": "doc_a",
        "requirement_name": "상주 인력",
        "raw_requirement_text": "사업기간 동안 상주 인력 2명을 투입한다.",
        "primary_action": "견적반영",
        "reasoning": "인원과 기간이 명시되어 공수 산정이 가능하다.",
        "review_status": "검토완료",
        "pool_version": "anchor_pool_v1",
    }
    base.update(overrides)
    return base


def _write_pool(tmp_path, anchors, name="anchor_pool_v1.jsonl"):
    path = tmp_path / name
    path.write_text(
        "\n".join(json.dumps(anchor, ensure_ascii=False) for anchor in anchors) + "\n",
        encoding="utf-8",
    )
    return path


def test_load_returns_reviewed_anchors_and_metadata(tmp_path):
    path = _write_pool(
        tmp_path,
        [
            _anchor(),
            _anchor(
                requirement_uid="doc_b:SFR-002",
                document_id="doc_b",
                primary_action="계약·질의검토",
            ),
        ],
    )

    pool, metadata = load_anchor_pool(path)

    assert len(pool) == 2
    assert metadata["pool_version"] == "anchor_pool_v1"
    assert metadata["reviewed_count"] == 2
    assert metadata["document_count"] == 2
    assert metadata["label_counts"] == {"계약·질의검토": 1, "견적반영": 1}
    assert len(metadata["sha256"]) == 64


def test_unreviewed_anchors_are_excluded(tmp_path):
    path = _write_pool(
        tmp_path,
        [
            _anchor(),
            _anchor(
                requirement_uid="doc_b:SFR-002",
                document_id="doc_b",
                review_status="후보",
            ),
            _anchor(
                requirement_uid="doc_c:SFR-003",
                document_id="doc_c",
                review_status="제외",
            ),
        ],
    )

    pool, metadata = load_anchor_pool(path)

    assert [anchor["requirement_uid"] for anchor in pool] == ["doc_a:SFR-001"]
    assert metadata["total_rows"] == 3
    assert metadata["reviewed_count"] == 1


def test_pool_without_reviewed_anchors_is_rejected(tmp_path):
    path = _write_pool(tmp_path, [_anchor(review_status="후보")])

    with pytest.raises(AnchorPoolError, match="검토완료 앵커가 없습니다"):
        load_anchor_pool(path)


def test_mixed_pool_versions_are_rejected(tmp_path):
    path = _write_pool(
        tmp_path,
        [
            _anchor(),
            _anchor(
                requirement_uid="doc_b:SFR-002",
                document_id="doc_b",
                pool_version="anchor_pool_v2",
            ),
        ],
    )

    with pytest.raises(AnchorPoolError, match="pool_version"):
        load_anchor_pool(path)


def test_duplicate_uid_is_rejected(tmp_path):
    path = _write_pool(tmp_path, [_anchor(), _anchor(document_id="doc_b")])

    with pytest.raises(AnchorPoolError, match="중복 requirement_uid"):
        load_anchor_pool(path)


def test_invalid_primary_action_is_rejected(tmp_path):
    path = _write_pool(tmp_path, [_anchor(primary_action="보류")])

    with pytest.raises(AnchorPoolError, match="primary_action"):
        load_anchor_pool(path)


def test_missing_field_is_rejected(tmp_path):
    anchor = _anchor()
    del anchor["reasoning"]
    path = _write_pool(tmp_path, [anchor])

    with pytest.raises(AnchorPoolError, match="필수 필드 누락"):
        load_anchor_pool(path)


def test_missing_file_names_the_path(tmp_path):
    with pytest.raises(AnchorPoolError, match="앵커 풀 파일이 없습니다"):
        load_anchor_pool(tmp_path / "nope.jsonl")


def test_labels_without_anchor_is_reported(tmp_path):
    path = _write_pool(tmp_path, [_anchor()])

    _, metadata = load_anchor_pool(path)

    assert metadata["labels_without_anchor"] == ["계약·질의검토", "통상수용"]
