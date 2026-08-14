import json

import pytest

from scripts.labeling.run_claude_labeling import (
    _write_or_validate_manifest,
    build_parser,
    load_samples,
    make_manifest,
)


def test_cli_is_dry_run_by_default():
    args = build_parser().parse_args([])
    assert args.execute is False
    assert args.effort == "medium"
    assert args.cache_ttl == "5m"


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
