import pytest

from scripts.evaluation.duplication import cross_document_similarity
from scripts.evaluation.folds import (
    _repeat_flags,
    aggregate,
    diagnose_all,
    evaluation_excluded_uids,
    make_lodo_folds,
)
from scripts.labeling.label_dataset import load_label_dataset


def _rows(spec):
    """(문서, 건수) 목록으로 최소 행을 만든다."""
    rows = []
    for document_id, count in spec:
        for i in range(count):
            rows.append(
                {
                    "requirement_uid": f"{document_id}:R-{i:03d}",
                    "document_id": document_id,
                    "raw_requirement_text": f"{document_id} 요구사항 {i}",
                    "primary_action": "통상수용",
                    "cost_basis": "없음",
                }
            )
    return rows


def test_every_document_is_the_unseen_one_exactly_once():
    """
    LODO를 쓰는 이유가 이것이다. 8:2 한 번이면 2개 문서만 평가에 오르고, fold
    기준선이 34~79%로 벌어져 있어 어느 2개가 걸리느냐로 결과가 뒤집힌다.
    """
    rows = _rows([("a", 3), ("b", 3), ("c", 3), ("d", 3)])
    folds = make_lodo_folds(rows)

    assert len(folds) == 4
    assert sorted(f.test_document for f in folds) == ["a", "b", "c", "d"]


def test_validation_rotates_so_no_document_is_always_the_yardstick():
    """
    고정된 한 문서를 계속 검증에 쓰면 그 문서의 난이도가 모든 fold의 조기 종료
    시점에 같은 방향으로 스며든다. 회전시키면 편향이 fold 사이에서 상쇄된다.
    """
    rows = _rows([("a", 3), ("b", 3), ("c", 3), ("d", 3)])
    folds = make_lodo_folds(rows)

    assert sorted(f.validation_document for f in folds) == ["a", "b", "c", "d"]
    for fold in folds:
        assert fold.validation_document != fold.test_document


def test_the_three_splits_never_share_a_document():
    """
    평가 문서를 보고 멈출 시점을 고르면 그 문서는 더 이상 처음 보는 문서가 아니다.
    학습·검증·평가가 문서 단위로 완전히 갈라져 있어야 한다.
    """
    rows = _rows([("a", 3), ("b", 4), ("c", 5), ("d", 6)])

    for fold in make_lodo_folds(rows):
        fit, validation, test = fold.split(rows)

        assert len(fit) + len(validation) + len(test) == len(rows)
        uids = [r["requirement_uid"] for r in fit + validation + test]
        assert len(set(uids)) == len(rows)

        documents = [{r["document_id"] for r in part} for part in (fit, validation, test)]
        assert documents[0] & documents[1] == set()
        assert documents[0] & documents[2] == set()
        assert documents[1] & documents[2] == set()


def test_folds_do_not_depend_on_input_row_order():
    """분할이 입력 순서에 흔들리면 같은 실험을 두 번 돌릴 수 없다."""
    rows = _rows([("c", 2), ("a", 2), ("b", 2)])

    assert make_lodo_folds(rows) == make_lodo_folds(list(reversed(rows)))


def test_frozen_anchors_are_train_only_and_never_scored():
    """결정 25: 이미 라벨 예시로 쓰인 100건은 처음 보는 평가 사례가 아니다."""
    rows, _ = load_label_dataset()
    excluded = evaluation_excluded_uids()
    evaluated = set()

    assert len(excluded) == 100
    for fold in make_lodo_folds(rows):
        fit, validation, test = fold.split(rows)
        assert not excluded.intersection(r["requirement_uid"] for r in validation)
        assert not excluded.intersection(r["requirement_uid"] for r in test)
        assert excluded.intersection(r["requirement_uid"] for r in fit)
        evaluated.update(r["requirement_uid"] for r in test)

    assert len(evaluated) == len(rows) - len(excluded) == 924


def test_repeat_exposure_uses_fit_documents_not_the_validation_document():
    rows = _rows([("a", 1), ("b", 1), ("c", 1), ("d", 1)])
    rows[0]["raw_requirement_text"] = "검증 문서에만 똑같이 존재하는 특별 문구"
    rows[1]["raw_requirement_text"] = "검증 문서에만 똑같이 존재하는 특별 문구"
    rows[2]["raw_requirement_text"] = "완전히 다른 학습 요구사항 하나"
    rows[3]["raw_requirement_text"] = "완전히 다른 학습 요구사항 둘"
    fold = make_lodo_folds(rows)[0]  # a=평가, b=검증, c·d=학습

    assert cross_document_similarity(rows).is_repeat[0]
    assert not _repeat_flags(fold, rows, 0.6)["a:R-000"]


def test_fit_excludes_the_validation_document_but_train_keeps_it():
    """
    §9.3의 통제 비교는 분할을 고정하고 모델만 바꾸는 것이다. early stopping이
    필요 없는 모델도 기본은 8문서(`fit_documents`)를 쓴다. 9문서를 쓰고 싶을 때를
    위해 `train_documents`를 남겨두되 둘을 섞지 않는다.
    """
    rows = _rows([("a", 2), ("b", 2), ("c", 2), ("d", 2)])
    fold = make_lodo_folds(rows)[0]

    assert fold.validation_document in fold.train_documents
    assert fold.validation_document not in fold.fit_documents
    assert len(fold.fit_documents) == len(fold.train_documents) - 1


def test_too_few_documents_fails_loudly():
    """3문서 미만이면 학습·검증·평가로 가를 수 없다. 조용히 빈 분할을 내지 않는다."""
    with pytest.raises(ValueError, match="최소 3개"):
        make_lodo_folds(_rows([("a", 2), ("b", 2)]))


def test_the_two_baselines_are_not_the_same_number():
    """
    배포 가능한 Dummy(학습 최빈을 찍음)와 oracle(평가 문서 안의 최빈)은 다른 값이다.
    앵커를 제외한 평가 모집단에서는 ccrs·genai의 최빈이 `견적반영`이라 학습 최빈
    `통상수용`과 갈린다. 두 값을 하나로 합쳐 보고하면 모델이 기준선을 넘었는지
    판단이 흐려진다.
    """
    rows, _ = load_label_dataset()
    diagnostics = diagnose_all(rows)

    assert all(d.trained_majority_label == "통상수용" for d in diagnostics)

    diverging = {
        d.fold.test_document
        for d in diagnostics
        if d.trained_majority_accuracy < d.oracle_majority_accuracy
    }
    assert diverging == {
        "ccrs_ai_platform",
        "genai_incident_response",
    }

    # oracle은 정의상 그 문서 안의 최빈이므로 어떤 예측기도 이보다 잘 찍을 수 없다.
    for d in diagnostics:
        assert d.trained_majority_accuracy <= d.oracle_majority_accuracy


def test_rare_cost_basis_cannot_be_evaluated_in_most_folds():
    """
    `외부인증` 3건은 defense 2 + mfds 1에 있다. 나머지 8개 fold는 평가 집합에
    한 건도 없어 그 값의 성능을 잴 수 없다(docs/issues/005). 학습 fold가 0건인
    것이 아니라 **평가가 불가능**한 것이 문제다.
    """
    rows, _ = load_label_dataset()
    diagnostics = diagnose_all(rows)

    missing = [
        d.fold.test_document
        for d in diagnostics
        if d.rare_value_coverage.get("cost_basis", {}).get("외부인증") == 0
    ]
    assert len(missing) == 8
    assert "defense_intelligent_platform" not in missing
    assert "mfds_drug_ai_review" not in missing


def test_repeat_exposure_reproduces_the_recorded_measurement():
    """
    결정 34가 기록한 실측값을 고정한다. 임계값 0.6에서 12.0%, 중앙값 0.209다.
    계산이 조용히 달라지면 issues/006의 보고 방식 전체가 근거를 잃는다.
    """
    rows, _ = load_label_dataset()
    result = cross_document_similarity(rows, threshold=0.6)

    assert result.repeat_rate == pytest.approx(0.120, abs=0.001)
    assert sorted(result.nearest_similarity)[len(rows) // 2] == pytest.approx(
        0.209, abs=0.001
    )


def test_easy_folds_are_also_the_ones_with_the_most_repeated_phrasing():
    """
    issues/003(분포 편차)과 006(문구 반복)을 서로 독립된 보고 문제로 적어뒀지만,
    실제로는 같은 방향으로 정렬돼 있다. 기준선이 높은 문서가 반복 노출도 높다.
    두 교란이 상쇄되지 않고 겹치므로 fold 간 점수 비교는 각각을 따로 볼 때보다
    더 조심해야 한다.
    """
    rows, _ = load_label_dataset()
    diagnostics = diagnose_all(rows)

    easiest = max(diagnostics, key=lambda d: d.oracle_majority_accuracy)
    hardest = min(diagnostics, key=lambda d: d.oracle_majority_accuracy)

    assert easiest.fold.test_document == "koen_ai_infrastructure"
    assert hardest.fold.test_document == "defense_intelligent_platform"
    assert easiest.repeat_exposure_rate > hardest.repeat_exposure_rate


def test_aggregate_reports_both_views_and_they_differ():
    """
    fold 단순 평균과 건수 가중은 다른 질문에 답한다. 하나만 쓰면 어느 쪽이든
    오해를 만든다(issues/003). 작은 fold가 낮은 점수일 때 둘이 갈린다.
    """
    result = aggregate([0.2, 0.8], [10, 90])

    assert result["fold_mean"] == pytest.approx(0.5)
    assert result["count_weighted"] == pytest.approx(0.74)


def test_aggregate_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        aggregate([0.1, 0.2], [10])
