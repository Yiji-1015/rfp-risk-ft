import json

import pytest

from scripts.labeling.claude_client import PROMPT_VERSION
from scripts.labeling.label_schema import SCHEMA_VERSION
from scripts.labeling.run_claude_labeling import (
    _write_or_validate_manifest,
    build_parser,
    load_samples,
    main,
    make_manifest,
)


def test_cli_is_dry_run_by_default():
    args = build_parser().parse_args([])
    assert args.execute is False
    assert args.effort == "medium"
    assert args.cache_ttl == "5m"
    assert args.strategy == "zero-shot"
    assert args.thinking == "adaptive"


def test_load_samples_and_manifest(tmp_path):
    source = tmp_path / "sample.jsonl"
    source.write_text(
        json.dumps(
            {
                "requirement_uid": "doc:SFR-001",
                "requirement_name": "검수",
                "raw_requirement_text": "세부 기준은 추후 협의한다.",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    samples = load_samples(source)
    manifest = make_manifest(samples, build_parser().parse_args([]))

    assert manifest["execute"] is False
    assert manifest["sample_count"] == 1
    assert manifest["parameters"]["cache_ttl"] == "5m"
    assert "temperature" not in manifest["parameters"]
    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["prompt_version"] == PROMPT_VERSION
    assert manifest["strategy"] == "zero-shot"
    assert "anchoring" not in manifest
    assert manifest["parameters"]["thinking"] == "adaptive"


def test_manifest_omits_thinking_and_effort_for_haiku():
    args = build_parser().parse_args(["--model", "claude-haiku-4-5-20251001"])
    manifest = make_manifest([], args)

    assert manifest["parameters"]["thinking"] is None
    assert manifest["parameters"]["effort"] is None


def test_fewshot_input_requires_document_id(tmp_path):
    source = tmp_path / "sample.jsonl"
    source.write_text(
        json.dumps(
            {
                "requirement_uid": "doc:SFR-001",
                "requirement_name": "검수",
                "raw_requirement_text": "세부 기준은 추후 협의한다.",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    # zero-shot은 통과하지만, 앵커 전략은 동일 문서 차단이 불가능하므로 거부해야 한다.
    assert load_samples(source)
    with pytest.raises(ValueError, match="document_id"):
        load_samples(source, require_document_id=True)


def test_resume_rejects_changed_parameters(tmp_path):
    path = tmp_path / "manifest.json"
    first = {
        "input": "sample.jsonl",
        "sample_count": 1,
        "parameters": {"model": "claude-sonnet-5", "cache_ttl": "5m"},
    }
    changed = {
        **first,
        "parameters": {"model": "claude-sonnet-5", "cache_ttl": "1h"},
    }
    _write_or_validate_manifest(path, first)

    with pytest.raises(RuntimeError, match="실행 조건이 다릅니다"):
        _write_or_validate_manifest(path, changed)


def test_resume_allows_growing_sample_count_and_refreshes_manifest(tmp_path):
    """--limit로 3건 확인 후 전체를 이어 돌리는 것은 정상 재개다."""
    path = tmp_path / "manifest.json"
    first = {
        "input": "sample.jsonl",
        "sample_count": 3,
        "strategy": "zero-shot",
        "parameters": {"model": "claude-sonnet-5"},
    }
    grown = {**first, "sample_count": 40}
    _write_or_validate_manifest(path, first)

    _write_or_validate_manifest(path, grown)

    assert json.loads(path.read_text(encoding="utf-8"))["sample_count"] == 40


def test_resume_rejects_changed_strategy(tmp_path):
    """같은 results.jsonl에 zero-shot과 few-shot 결과가 섞이면 비교가 무너진다."""
    path = tmp_path / "manifest.json"
    first = {
        "input": "sample.jsonl",
        "sample_count": 1,
        "strategy": "zero-shot",
        "parameters": {"model": "claude-sonnet-5"},
    }
    changed = {**first, "strategy": "fewshot-stratified"}
    _write_or_validate_manifest(path, first)

    with pytest.raises(RuntimeError, match="실행 조건이 다릅니다"):
        _write_or_validate_manifest(path, changed)


def _write_sample(tmp_path):
    source = tmp_path / "sample.jsonl"
    source.write_text(
        json.dumps(
            {
                "requirement_uid": "doc_z:SFR-001",
                "document_id": "doc_z",
                "requirement_name": "무상 추가개발",
                "raw_requirement_text": "발주기관이 요구하는 기능은 무상으로 추가 개발한다.",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return source


def _write_anchor_pool(tmp_path, labels=("통상수용", "견적반영", "계약·질의검토")):
    path = tmp_path / "anchor_pool_v1.jsonl"
    rows = [
        {
            "requirement_uid": f"doc_{index}:R-1",
            "document_id": f"doc_{index}",
            "requirement_name": f"사례 {index}",
            "raw_requirement_text": f"무상 추가 개발과 관련한 사례 {index} 본문이다.",
            "primary_action": label,
            "reasoning": f"{label} 판정 사유",
            "review_status": "검토완료",
            "pool_version": "anchor_pool_v1",
        }
        for index, label in enumerate(labels)
    ]
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    return path


def test_fewshot_dry_run_records_anchoring_and_previews(tmp_path, capsys):
    source = _write_sample(tmp_path)
    pool = _write_anchor_pool(tmp_path)

    exit_code = main(
        [
            "--input", str(source),
            "--anchor-pool", str(pool),
            "--strategy", "fewshot-stratified",
        ]
    )
    out = capsys.readouterr().out

    assert exit_code == 0
    manifest = json.loads(out[: out.index("\n[앵커 인출 미리보기]")])
    assert manifest["strategy"] == "fewshot-stratified"
    assert manifest["anchoring"]["retrieval"] == "stratified"
    assert manifest["anchoring"]["anchor_pool"]["pool_version"] == "anchor_pool_v1"
    assert manifest["anchoring"]["retriever"]["char_ngram_range"] == [3, 4]
    assert "주입 앵커 라벨 분포" in out
    assert "dry-run" in out


def test_fewshot_dry_run_fails_without_anchor_pool(tmp_path, capsys):
    source = _write_sample(tmp_path)

    exit_code = main(
        [
            "--input", str(source),
            "--anchor-pool", str(tmp_path / "missing.jsonl"),
            "--strategy", "fewshot-similarity",
        ]
    )

    assert exit_code == 2
    assert "앵커 풀 파일이 없습니다" in capsys.readouterr().err


def test_stratified_strategy_requires_every_label(tmp_path, capsys):
    source = _write_sample(tmp_path)
    pool = _write_anchor_pool(tmp_path, labels=("견적반영", "통상수용"))

    exit_code = main(
        [
            "--input", str(source),
            "--anchor-pool", str(pool),
            "--strategy", "fewshot-stratified",
        ]
    )

    assert exit_code == 2
    assert "계약·질의검토" in capsys.readouterr().err
