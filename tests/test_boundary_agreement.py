from scripts.evaluation.boundary_agreement import profile, unanimous_errors

GOLD = {
    "a": "견적반영",
    "b": "계약·질의검토",
    "c": "견적반영",
    "d": "통상수용",
}
UIDS = ["a", "b", "c", "d"]


def test_only_unanimous_flips_on_the_boundary_are_collected():
    tables = {
        # a: 셋 다 반대쪽 → 수집. b: 하나가 정답을 맞혀 제외.
        "m1": {"a": "계약·질의검토", "b": "견적반영", "c": "견적반영", "d": "견적반영"},
        "m2": {"a": "계약·질의검토", "b": "계약·질의검토", "c": "계약·질의검토", "d": "견적반영"},
        "m3": {"a": "계약·질의검토", "b": "견적반영", "c": "통상수용", "d": "견적반영"},
    }

    result = unanimous_errors(tables, GOLD, UIDS)

    assert result == ["a"]
    # c는 한 모델이 통상수용이라 "한 방향으로 전원 오답"이 아니다.
    assert "c" not in result
    # d는 경계 라벨이 아니라 애초에 대상이 아니다.
    assert "d" not in result


def test_profile_counts_denials_costs_and_markers():
    rows = {
        "a": {
            "reasoning": "수치 목표가 없어 blocker는 아니다",
            "cost_basis": "복합",
            "build_difficulty": "높음",
            "model_text": "폐쇄망 내에서 협의 후 결정한다",
        },
        "b": {
            "reasoning": "조달 가능성을 사전 확인해야 한다",
            "cost_basis": "장비·인프라",
            "build_difficulty": "보통",
            "model_text": "서버 3대를 도입한다",
        },
    }

    result = profile(rows, ["a", "b"])

    assert result["count"] == 2
    assert result["blocker_denied"] == 0.5  # a만 blocker를 부정한다
    assert result["cost_complex"] == 0.5
    assert result["build_high"] == 0.5
    assert result["markers"]["폐쇄망"] == 0.5
    assert result["markers"]["협의"] == 0.5
    assert result["markers"]["특정되지·명시되지"] == 0.0


def test_profile_of_an_empty_set_is_empty():
    assert profile({}, []) == {}
