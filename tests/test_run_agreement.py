import json
import math
from pathlib import Path

import pytest

from scripts.evaluation.run_agreement import (
    Agreement,
    anchor_uids,
    agreement_rows,
    compare_runs,
    confusion_rows,
    jaccard,
    load_run_results,
    repeat_baseline_agreement,
    same_document_anchor_count,
    stratify_by_anchor_overlap,
    wilson_interval,
)


def make_record(uid, primary, *, anchors=(), blockers=(), cost="없음", domain="보통", build="보통"):
    return {
        "requirement_uid": uid,
        "status": "ok",
        "label": {
            "requirement_uid": uid,
            "primary_action": primary,
            "blockers": list(blockers),
            "cost_basis": cost,
            "domain_dependency": domain,
            "build_difficulty": build,
            "reasoning": "테스트",
        },
        "anchors_used": [{"requirement_uid": a} for a in anchors],
    }


def index(records):
    return {r["requirement_uid"]: r for r in records}


class TestWilsonInterval:
    def test_zero_total_means_no_information(self):
        assert wilson_interval(0, 0) == (0.0, 1.0)

    def test_interval_contains_the_point_estimate(self):
        low, high = wilson_interval(74, 86)
        assert low < 74 / 86 < high

    def test_interval_stays_inside_zero_and_one(self):
        for successes, total in [(0, 5), (5, 5), (1, 100), (99, 100)]:
            low, high = wilson_interval(successes, total)
            assert 0.0 <= low <= high <= 1.0

    def test_more_data_narrows_the_interval(self):
        narrow = wilson_interval(90, 100)
        wide = wilson_interval(9, 10)
        assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])

    def test_known_value(self):
        # 결정 34가 인용한 37/40의 구간 (0.801–0.974).
        low, high = wilson_interval(37, 40)
        assert math.isclose(low, 0.801, abs_tol=0.002)
        assert math.isclose(high, 0.974, abs_tol=0.002)

    def test_rejects_impossible_counts(self):
        with pytest.raises(ValueError):
            wilson_interval(5, 3)
        with pytest.raises(ValueError):
            wilson_interval(-1, 3)


class TestAgreement:
    def test_rate_and_empty_total(self):
        assert Agreement("x", 3, 4).rate == 0.75
        assert Agreement("x", 0, 0).rate == 0.0

    def test_overlaps_is_symmetric(self):
        a = Agreement("a", 74, 86)
        b = repeat_baseline_agreement()
        assert a.overlaps(b) is b.overlaps(a)

    def test_clearly_separated_intervals_do_not_overlap(self):
        assert not Agreement("low", 10, 100).overlaps(Agreement("high", 95, 100))

    def test_baseline_matches_decision_23(self):
        assert (repeat_baseline_agreement().agreed, repeat_baseline_agreement().total) == (37, 40)


class TestJaccard:
    def test_identical_sets(self):
        assert jaccard(frozenset({"a", "b"}), frozenset({"a", "b"})) == 1.0

    def test_disjoint_sets(self):
        assert jaccard(frozenset({"a"}), frozenset({"b"})) == 0.0

    def test_both_empty_counts_as_identical(self):
        assert jaccard(frozenset(), frozenset()) == 1.0

    def test_partial_overlap(self):
        assert jaccard(frozenset({"a", "b"}), frozenset({"b", "c"})) == pytest.approx(1 / 3)


class TestLoadRunResults:
    def test_skips_error_rows_rather_than_counting_them(self, tmp_path):
        path = tmp_path / "results.jsonl"
        path.write_text(
            json.dumps(make_record("d:A", "통상수용"), ensure_ascii=False)
            + "\n"
            + json.dumps({"requirement_uid": "d:B", "status": "error", "error": "x"}, ensure_ascii=False)
            + "\n\n",
            encoding="utf-8",
        )
        loaded = load_run_results(path)
        assert set(loaded) == {"d:A"}


class TestSameDocumentAnchors:
    def test_counts_anchors_from_the_targets_own_document(self):
        results = index([make_record("doc1:A", "통상수용", anchors=("doc1:Z", "doc2:Y"))])
        assert same_document_anchor_count(results) == 1

    def test_zero_when_decision_10_is_respected(self):
        results = index([make_record("doc1:A", "통상수용", anchors=("doc2:Y", "doc3:Z"))])
        assert same_document_anchor_count(results) == 0


class TestCompareRuns:
    def test_uses_the_intersection_as_the_denominator(self):
        left = index([make_record("d:A", "통상수용"), make_record("d:B", "견적반영")])
        right = index([make_record("d:A", "통상수용")])
        comparison = compare_runs(left, right)
        assert comparison.uids == ("d:A",)
        assert comparison.primary.total == 1

    def test_raises_when_nothing_is_shared(self):
        with pytest.raises(ValueError):
            compare_runs(index([make_record("d:A", "통상수용")]), index([make_record("d:B", "통상수용")]))

    def test_counts_field_agreement(self):
        left = index([make_record("d:A", "통상수용", cost="없음"), make_record("d:B", "견적반영", cost="라이선스")])
        right = index([make_record("d:A", "통상수용", cost="없음"), make_record("d:B", "계약·질의검토", cost="복합")])
        comparison = compare_runs(left, right)
        assert comparison.primary.agreed == 1
        assert comparison.agreements["cost_basis"].agreed == 1

    def test_blockers_compare_as_sets_not_sequences(self):
        left = index([make_record("d:A", "계약·질의검토", blockers=("범위·책임", "기술실현성"))])
        right = index([make_record("d:A", "계약·질의검토", blockers=("기술실현성", "범위·책임"))])
        assert compare_runs(left, right).agreements["blockers"].agreed == 1

    def test_records_disagreement_direction_and_anchor_overlap(self):
        left = index([make_record("d:A", "견적반영", anchors=("x:1", "x:2"))])
        right = index([make_record("d:A", "계약·질의검토", anchors=("x:2", "x:3"))])
        (disagreement,) = compare_runs(left, right).disagreements
        assert (disagreement.left, disagreement.right) == ("견적반영", "계약·질의검토")
        assert disagreement.anchor_jaccard == pytest.approx(1 / 3)

    def test_confusion_counts_every_pair(self):
        left = index([make_record("d:A", "통상수용"), make_record("d:B", "통상수용")])
        right = index([make_record("d:A", "통상수용"), make_record("d:B", "견적반영")])
        comparison = compare_runs(left, right)
        assert comparison.confusion[("통상수용", "통상수용")] == 1
        assert comparison.confusion[("통상수용", "견적반영")] == 1
        assert sum(comparison.confusion.values()) == len(comparison.uids)

    def test_identical_anchor_uids_selects_the_unchanged_stratum(self):
        left = index([make_record("d:A", "통상수용", anchors=("x:1",)), make_record("d:B", "통상수용", anchors=("x:1",))])
        right = index([make_record("d:A", "통상수용", anchors=("x:1",)), make_record("d:B", "통상수용", anchors=("x:2",))])
        assert compare_runs(left, right).identical_anchor_uids == ("d:A",)


class TestStratifyByAnchorOverlap:
    def test_splits_into_three_strata_that_sum_to_the_whole(self):
        left = index([
            make_record("d:A", "통상수용", anchors=("x:1",)),
            make_record("d:B", "통상수용", anchors=("x:1", "x:2")),
            make_record("d:C", "통상수용", anchors=("x:1",)),
        ])
        right = index([
            make_record("d:A", "통상수용", anchors=("x:1",)),
            make_record("d:B", "견적반영", anchors=("x:2", "x:3")),
            make_record("d:C", "견적반영", anchors=("x:9",)),
        ])
        comparison = compare_runs(left, right)
        strata = stratify_by_anchor_overlap(comparison, left, right)
        assert [s.total for s in strata] == [1, 1, 1]
        assert sum(s.total for s in strata) == len(comparison.uids)
        assert [s.agreed for s in strata] == [1, 0, 0]

    def test_empty_stratum_reports_no_information_rather_than_zero_percent(self):
        left = index([make_record("d:A", "통상수용", anchors=("x:1",))])
        right = index([make_record("d:A", "통상수용", anchors=("x:1",))])
        strata = stratify_by_anchor_overlap(compare_runs(left, right), left, right)
        empty = [s for s in strata if s.total == 0]
        assert empty and all(s.interval == (0.0, 1.0) for s in empty)


class TestTableHelpers:
    def test_confusion_rows_is_square_over_the_label_set(self):
        left = index([make_record("d:A", "통상수용")])
        right = index([make_record("d:A", "견적반영")])
        rows = confusion_rows(compare_runs(left, right))
        assert len(rows) == 3
        assert all(len(r) == 4 for r in rows)

    def test_agreement_rows_always_carry_the_interval(self):
        (row,) = agreement_rows([Agreement("x", 9, 10)])
        assert row["일치"] == "9/10"
        assert row["Wilson 95% 하한"] < row["일치율"] < row["Wilson 95% 상한"]


class TestRealRunsIfPresent:
    """실제 실행 기록이 있으면 형식 계약을 확인한다. 없으면 건너뛴다."""

    ROOT = Path(__file__).resolve().parents[1]
    CHUNK2 = ROOT / "reports/current/claude_batches/batch_chunk2/results.jsonl"
    FULL = ROOT / "reports/current/claude_runs/batch_full_101_1024/results.jsonl"

    def test_the_two_batches_share_requirements_and_respect_decision_10(self):
        if not (self.CHUNK2.exists() and self.FULL.exists()):
            pytest.skip("실행 기록이 없습니다.")
        left = load_run_results(self.CHUNK2)
        right = load_run_results(self.FULL)
        comparison = compare_runs(left, right, left_name="chunk2", right_name="full")
        assert comparison.uids
        assert same_document_anchor_count(left) == 0
        assert same_document_anchor_count(right) == 0
        assert sum(comparison.confusion.values()) == len(comparison.uids)
        assert len(comparison.disagreements) == comparison.primary.total - comparison.primary.agreed
