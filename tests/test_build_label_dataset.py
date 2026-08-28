import json

import pytest

from scripts.labeling.build_label_dataset import REQUIREMENT_FIELDS, build_rows


def _requirement(uid, doc="doc_a"):
    return {
        "requirement_uid": uid,
        "document_id": doc,
        "agency": "기관",
        "domain": "도메인",
        "requirement_id": uid.split(":")[1],
        "requirement_type": "기능 요구사항",
        "requirement_name": "이름",
        "raw_requirement_text": "본문",
    }


def _label(uid, action, blockers=(), cost="없음"):
    return {
        "requirement_uid": uid,
        "primary_action": action,
        "blockers": list(blockers),
        "cost_basis": cost,
        "domain_dependency": "보통",
        "build_difficulty": "보통",
        "reasoning": "사유",
    }


def _write_run(tmp_path, name, rows):
    d = tmp_path / name
    d.mkdir(parents=True)
    (d / "results.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )
    return d


@pytest.fixture
def runs_dir(tmp_path, monkeypatch):
    from scripts.labeling import build_label_dataset

    monkeypatch.setattr(build_label_dataset, "RUNS_DIR", tmp_path)
    return tmp_path


def test_rule_violation_is_corrected_and_original_is_kept(runs_dir):
    """
    보조 축이 주 라벨보다 구체적인 근거다. 어긋나면 규칙을 적용하되
    모델 원본을 버리지 않는다. 버리면 보정 여부를 나중에 감사할 수 없다.
    """
    _write_run(runs_dir, "run_a", [
        # cost_basis가 있는데 통상수용이라 답한 행. 규칙상 견적반영이다.
        {"status": "ok", "requirement_uid": "doc_a:R-1",
         "label": _label("doc_a:R-1", "통상수용", cost="라이선스")},
    ])
    reqs = {"doc_a:R-1": _requirement("doc_a:R-1")}

    rows, corrections = build_rows(reqs, runs=(("run_a", "배치"),))

    assert rows[0]["primary_action"] == "견적반영"
    assert rows[0]["primary_action_model"] == "통상수용"
    assert rows[0]["rule_corrected"] is True
    assert len(corrections) == 1
    assert corrections[0]["requirement_uid"] == "doc_a:R-1"


def test_execution_path_and_source_run_are_explicit(runs_dir):
    """기존 병합 파일은 'input' 필드 유무로 경로를 추측해야 했다. 명시 필드로 남긴다."""
    _write_run(runs_dir, "sync_run", [
        {"status": "ok", "requirement_uid": "doc_a:R-1",
         "label": _label("doc_a:R-1", "통상수용")},
    ])
    _write_run(runs_dir, "batch_run", [
        {"status": "ok", "requirement_uid": "doc_a:R-2",
         "label": _label("doc_a:R-2", "통상수용")},
    ])
    reqs = {u: _requirement(u) for u in ("doc_a:R-1", "doc_a:R-2")}

    rows, _ = build_rows(reqs, runs=(("sync_run", "동기"), ("batch_run", "배치")))

    by_uid = {r["requirement_uid"]: r for r in rows}
    assert by_uid["doc_a:R-1"]["execution_path"] == "동기"
    assert by_uid["doc_a:R-1"]["source_run"] == "sync_run"
    assert by_uid["doc_a:R-2"]["execution_path"] == "배치"
    # 모든 행이 같은 키 집합을 갖는다. 균일 스키마가 이 데이터셋의 존재 이유다.
    assert len({tuple(sorted(r)) for r in rows}) == 1
    for field in REQUIREMENT_FIELDS:
        assert all(field in r for r in rows)


def test_model_text_uses_normalized_list_and_requirement_name(runs_dir):
    _write_run(runs_dir, "run_a", [
        {"status": "ok", "requirement_uid": "doc_a:R-1",
         "label": _label("doc_a:R-1", "통상수용")},
    ])
    requirement = _requirement("doc_a:R-1")
    requirement["requirement_name"] = "AI 서비스"
    requirement["raw_requirement_text"] = "◦ 모델 개발\n■ 서비스 구현"

    rows, _ = build_rows(
        {"doc_a:R-1": requirement}, runs=(("run_a", "배치"),)
    )

    assert rows[0]["raw_requirement_text"] == "◦ 모델 개발\n■ 서비스 구현"
    assert rows[0]["normalized_requirement_text"] == "- 모델 개발\n- 서비스 구현"
    assert rows[0]["model_text"] == "AI 서비스\n- 모델 개발\n- 서비스 구현"


def test_failed_rows_are_excluded(runs_dir):
    """생성 반복으로 실패한 행은 label이 없다. 데이터셋에 들어가면 안 된다."""
    _write_run(runs_dir, "run_a", [
        {"status": "ok", "requirement_uid": "doc_a:R-1",
         "label": _label("doc_a:R-1", "통상수용")},
        {"status": "error", "requirement_uid": "doc_a:R-2", "error": "json_invalid"},
    ])
    reqs = {u: _requirement(u) for u in ("doc_a:R-1", "doc_a:R-2")}

    rows, _ = build_rows(reqs, runs=(("run_a", "배치"),))

    assert [r["requirement_uid"] for r in rows] == ["doc_a:R-1"]


def test_overlapping_runs_are_rejected(runs_dir):
    """재시도 실행이 원본과 겹치면 어느 라벨이 최종인지 정해지지 않는다."""
    _write_run(runs_dir, "run_a", [
        {"status": "ok", "requirement_uid": "doc_a:R-1",
         "label": _label("doc_a:R-1", "통상수용")},
    ])
    _write_run(runs_dir, "run_b", [
        {"status": "ok", "requirement_uid": "doc_a:R-1",
         "label": _label("doc_a:R-1", "견적반영", cost="라이선스")},
    ])
    reqs = {"doc_a:R-1": _requirement("doc_a:R-1")}

    with pytest.raises(ValueError, match="양쪽에 있습니다"):
        build_rows(reqs, runs=(("run_a", "배치"), ("run_b", "배치")))


def test_unknown_requirement_uid_is_rejected(runs_dir):
    """라벨은 있는데 요구사항이 없으면 조인이 조용히 비어버린다. 즉시 실패시킨다."""
    _write_run(runs_dir, "run_a", [
        {"status": "ok", "requirement_uid": "doc_a:R-9",
         "label": _label("doc_a:R-9", "통상수용")},
    ])

    with pytest.raises(KeyError, match="doc_a:R-9"):
        build_rows({}, runs=(("run_a", "배치"),))
