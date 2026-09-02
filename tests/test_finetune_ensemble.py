from scripts.evaluation.finetune_ensemble import describe, nested_selection, overlap, vote

MEMBERS = {
    "a": {"u1": "통상수용", "u2": "견적반영", "u3": "계약·질의검토"},
    "b": {"u1": "통상수용", "u2": "계약·질의검토", "u3": "계약·질의검토"},
    "c": {"u1": "견적반영", "u2": "계약·질의검토", "u3": "통상수용"},
}
UIDS = ["u1", "u2", "u3"]
GOLD = {"u1": "통상수용", "u2": "견적반영", "u3": "계약·질의검토"}


def test_majority_wins_and_a_three_way_split_falls_back_to_the_first_member():
    # u1: a·b가 통상수용 → 다수결. u2: b·c가 계약 → 다수결(정답과 다름).
    # u3: a·b가 계약 → 다수결.
    assert vote(MEMBERS, ["a", "b", "c"], UIDS) == ["통상수용", "계약·질의검토", "계약·질의검토"]

    # 셋이 전부 갈리면 첫 멤버를 따른다. 순서가 곧 우선순위다.
    split = {
        "a": {"u1": "통상수용"},
        "b": {"u1": "견적반영"},
        "c": {"u1": "계약·질의검토"},
    }
    assert vote(split, ["b", "a", "c"], ["u1"]) == ["견적반영"]


def test_describe_separates_boundary_errors_from_the_rest():
    gold = ["견적반영", "계약·질의검토", "통상수용", "통상수용"]
    pred = ["계약·질의검토", "견적반영", "견적반영", "통상수용"]

    result = describe(gold, pred, ["d1", "d1", "d2", "d2"])

    assert result["errors"] == 3
    # 앞의 두 건만 견적↔계약 상호 혼동이다. 세 번째는 통상수용이 섞여 제외된다.
    assert result["boundary_errors"] == 2
    assert set(result["per_label_f1"]) == {"통상수용", "견적반영", "계약·질의검토"}


def test_overlap_counts_what_only_one_member_gets_wrong():
    result = overlap(MEMBERS, GOLD, UIDS, "a", "b")

    assert result["left_errors"] == 0  # a는 전부 맞다
    assert result["right_errors"] == 1  # b는 u2를 틀린다
    assert result["both_wrong"] == 0
    assert result["one_wrong"] == 1
    assert result["oracle_accuracy"] == 1.0  # 한쪽이 맞으면 건질 수 있다


def test_nested_selection_reports_which_combination_each_round_picked():
    documents = {"u1": "d1", "u2": "d2", "u3": "d2"}
    combos = [("a",), ("b",), ("a", "b", "c")]

    result = nested_selection(MEMBERS, GOLD, documents, UIDS, combos)

    assert 0.0 <= result["macro_f1"] <= 1.0
    # 문서가 둘이므로 선택도 두 번 일어난다.
    assert sum(result["selected"].values()) == 2
    assert set(result["selected"]) <= {"a", "b", "a+b+c"}
