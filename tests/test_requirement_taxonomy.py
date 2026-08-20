import pytest

from scripts.labeling.label_dataset import load_label_dataset
from scripts.labeling.requirement_taxonomy import (
    CANONICAL_TYPES,
    PREFIX_TO_CANONICAL,
    TEXT_TO_CANONICAL,
    UNKNOWN,
    normalize_requirement_type,
    normalize_text,
)


@pytest.mark.parametrize(
    "raw",
    ["기능 요구사항", "기능", "기능 요구사항(SFR)", "기능요구사항"],
)
def test_fragmented_spellings_collapse_to_one_type(raw):
    """
    원본이 60종으로 쪼개진 이유가 이 네 표기다. 같은 유형이 표기만 달라 별개로
    집계되면서 통상수용 비율이 48.5/35.0/10.5%로 갈렸다(docs/issues/001).
    """
    canonical, source = normalize_requirement_type(raw, "SFR-001")

    assert canonical == "기능"
    assert source == "text"


def test_text_wins_over_prefix_when_they_disagree():
    """
    접두어는 문서를 넘어 충돌한다. kexim_ai_platform은 표준 접두어를 다른 뜻으로 쓴다.
    INR이 다섯 문서에서는 인터페이스지만 kexim에서는 인프라다. 문서가 스스로 밝힌
    표기가 접두어 관행보다 사실에 가깝다.
    """
    assert normalize_requirement_type("인프라 요구사항", "INR-001") == (
        "시스템장비구성",
        "text",
    )
    assert normalize_requirement_type("인터페이스 요구사항", "INR-001") == (
        "인터페이스",
        "text",
    )
    # 표기가 없을 때만 접두어로 떨어진다.
    assert normalize_requirement_type(None, "INR-001") == ("인터페이스", "prefix")


def test_missing_type_falls_back_to_prefix():
    """koen_ai_infrastructure 101건은 표기가 없다. 접두어 충돌이 없어 대체가 안전하다."""
    assert normalize_requirement_type(None, "FUN-001") == ("기능", "prefix")
    assert normalize_requirement_type(None, "TST-001") == ("테스트", "prefix")
    assert normalize_requirement_type("", "DAT-001") == ("데이터", "prefix")


def test_unmappable_input_is_marked_not_guessed():
    """모르는 것을 최빈값으로 채우면 조용히 왜곡된다. 기타로 두고 근거를 none으로 남긴다."""
    assert normalize_requirement_type("듣도보도못한 요구사항", "ZZZ-001") == (
        UNKNOWN,
        "none",
    )


def test_normalize_text_strips_code_space_and_suffix():
    assert normalize_text("프로젝트 관리 요구사항(PMR)") == "프로젝트관리"
    assert normalize_text("요구사항") is None
    assert normalize_text(None) is None


def test_every_mapping_target_is_canonical():
    """매핑 표가 정본 목록 밖의 값을 만들어내지 않는지 본다."""
    for table in (TEXT_TO_CANONICAL, PREFIX_TO_CANONICAL):
        unknown = set(table.values()) - set(CANONICAL_TYPES)
        assert not unknown, f"정본에 없는 대상: {unknown}"


def test_frozen_dataset_is_fully_mapped():
    """
    실제 데이터에 기타가 남으면 매핑 표에 구멍이 있다는 뜻이다. 새 문서가 들어와
    새 표기를 가져오면 여기서 걸린다.
    """
    rows, meta = load_label_dataset()

    assert meta["unmapped_type_count"] == 0
    assert set(meta["requirement_type_counts"]) <= set(CANONICAL_TYPES)
    # 표기가 없는 문서는 koen 하나뿐이고 그 건수가 101이다.
    assert meta["requirement_type_source_counts"] == {"text": 923, "prefix": 101}
    assert all(r["requirement_type"] or True for r in rows)  # 원본 표기는 보존한다
    assert "requirement_type" in rows[0], "원본 표기를 지우면 매핑을 재검증할 수 없다"
