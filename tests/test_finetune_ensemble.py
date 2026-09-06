import json

import pytest

from scripts.evaluation.finetune_ensemble import (
    TFIDF_MEMBERS,
    describe,
    load_members,
    member_tag,
    nested_selection,
    overlap,
    vote,
)

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


class TestMemberTag:
    """멤버 태그는 실행을 유일하게 가리켜야 한다.

    예전 규칙은 크기와 seed를 뭉뚱그려 `ftL`·`ft42`·`ftM` 하나에 여러 실행이 들어왔고,
    딕셔너리에 나중 것이 덮어써져 large seed 7·13과 마스킹 seed 3개가 조용히 사라졌다.
    """

    def test_roberta_sizes_get_short_codes(self):
        assert member_tag({"model": "klue/roberta-large", "seed": 42}) == "ftL42"
        assert member_tag({"model": "klue/roberta-base", "seed": 7}) == "ftB7"
        assert member_tag({"model": "klue/roberta-small", "seed": 13}) == "ftS13"

    def test_seeds_do_not_collide(self):
        tags = {member_tag({"model": "klue/roberta-large", "seed": s}) for s in (42, 7, 13)}
        assert len(tags) == 3

    def test_masking_is_a_separate_member(self):
        plain = member_tag({"model": "klue/roberta-base", "seed": 42})
        masked = member_tag({"model": "klue/roberta-base", "seed": 42, "mask": "subject+ending+josa"})
        assert plain != masked
        assert masked.endswith("M")

    def test_other_families_do_not_borrow_the_roberta_size_code(self):
        # `kobigbird-bert-base`는 이름이 base로 끝나지만 roberta-base가 아니다.
        big = member_tag({"model": "monologg/kobigbird-bert-base", "seed": 42})
        assert big != member_tag({"model": "klue/roberta-base", "seed": 42})
        assert member_tag({"model": "monologg/koelectra-base-v3-discriminator", "seed": 42}) != big


class TestLoadMembersGuards:
    def _write(self, tmp_path, configs):
        runs = tmp_path / "runs.jsonl"
        with runs.open("w", encoding="utf-8") as handle:
            for config in configs:
                handle.write(json.dumps({
                    "config": {"fold": -1, "mask": None, **config},
                    "results": [{"predictions": [{"requirement_uid": "u1", "pred": "통상수용"}]}],
                }, ensure_ascii=False) + "\n")
        oof = tmp_path / "oof.csv"
        oof.write_text(
            "requirement_uid,gold,test_document,"
            + ",".join(TFIDF_MEMBERS.values())
            + "\nu1,통상수용,d1," + ",".join(["통상수용"] * len(TFIDF_MEMBERS)) + "\n",
            encoding="utf-8",
        )
        return runs, oof

    def test_binary_runs_are_skipped(self, tmp_path):
        # 2분류 실행은 `검토필요`를 예측하므로 3분류 투표에 들어가면 안 된다.
        runs, oof = self._write(tmp_path, [
            {"model": "klue/roberta-base", "seed": 42},
            {"model": "klue/roberta-base", "seed": 42, "binary": True},
        ])
        members, _, _ = load_members(runs, oof)
        assert "ftB42" in members
        assert len([t for t in members if t.startswith("ftB42")]) == 1

    def test_duplicate_runs_raise_instead_of_overwriting(self, tmp_path):
        runs, oof = self._write(tmp_path, [
            {"model": "klue/roberta-large", "seed": 42},
            {"model": "klue/roberta-large", "seed": 42},
        ])
        with pytest.raises(ValueError, match="겹칩니다"):
            load_members(runs, oof)
