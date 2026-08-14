"""Run cached Claude RFP labeling. Paid calls require --execute."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from scripts.labeling.claude_client import (
    DEFAULT_MODEL,
    HAIKU_MODEL,
    ClaudeLabelingClient,
    ClaudeSettings,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data" / "samples" / "labeling_pilot_sample_v0.1.0.jsonl"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model", choices=[DEFAULT_MODEL, HAIKU_MODEL], default=DEFAULT_MODEL)
    parser.add_argument("--effort", choices=["low", "medium", "high"], default="medium")
    parser.add_argument("--cache-ttl", choices=["5m", "1h"], default="5m")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="실제 유료 API 호출 실행. 생략하면 manifest만 출력합니다.",
    )
    return parser


def load_samples(path: Path) -> list[dict[str, Any]]:
    required = {"requirement_uid", "requirement_name", "raw_requirement_text"}
    samples = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            missing = required - item.keys()
            if missing:
                raise ValueError(
                    f"{path}:{line_number} 필수 필드 누락: {sorted(missing)}"
                )
            samples.append(item)
    return samples


def make_manifest(samples: Sequence[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    effort = args.effort if args.model == DEFAULT_MODEL else None
    return {
        "execute": args.execute,
        "input": str(args.input.resolve()),
        "sample_count": len(samples),
        "parameters": {
            "model": args.model,
            "effort": effort,
            "max_tokens": args.max_tokens,
            "cache_ttl": args.cache_ttl,
            "timeout_seconds": 120.0,
            "max_retries": 2,
            "structured_output": True,
        },
    }


def _default_output_dir() -> Path:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return ROOT / "reports" / "current" / "claude_runs" / run_id


def _write_or_validate_manifest(path: Path, manifest: dict[str, Any]) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        comparable_keys = ("input", "sample_count", "parameters")
        if any(existing.get(key) != manifest.get(key) for key in comparable_keys):
            raise RuntimeError(
                "기존 output-dir의 manifest와 실행 조건이 다릅니다. "
                "새 output-dir를 사용하세요."
            )
        return
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _completed_uids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                if item.get("status") == "ok":
                    completed.add(item["requirement_uid"])
    return completed


def execute_run(
    samples: Sequence[dict[str, Any]],
    args: argparse.Namespace,
    output_dir: Path,
) -> int:
    try:
        from dotenv import load_dotenv
    except ImportError as exc:
        raise RuntimeError("python-dotenv 패키지가 필요합니다.") from exc
    load_dotenv(ROOT / ".env")
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY가 설정되지 않았습니다.")

    settings = ClaudeSettings(
        model=args.model,
        effort=args.effort,
        max_tokens=args.max_tokens,
        cache_ttl=args.cache_ttl,
    )
    client = ClaudeLabelingClient(settings=settings)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"
    manifest_path = output_dir / "manifest.json"
    _write_or_validate_manifest(manifest_path, make_manifest(samples, args))

    completed = _completed_uids(results_path)
    failures = 0
    with results_path.open("a", encoding="utf-8") as handle:
        for index, sample in enumerate(samples, 1):
            uid = sample["requirement_uid"]
            if uid in completed:
                print(f"[{index}/{len(samples)}] skip {uid}")
                continue
            try:
                result = client.label_requirement(
                    requirement_uid=uid,
                    requirement_name=sample["requirement_name"],
                    requirement_text=sample["raw_requirement_text"],
                )
                record = {
                    "status": "ok",
                    "requirement_uid": uid,
                    "input": sample,
                    "label": result.label.model_dump(),
                    "metadata": result.metadata,
                }
                print(
                    f"[{index}/{len(samples)}] ok {uid} "
                    f"cache_read={result.metadata['cache_read_input_tokens']}"
                )
            except Exception as exc:  # item failures must not discard prior work
                failures += 1
                record = {
                    "status": "error",
                    "requirement_uid": uid,
                    "input": sample,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                print(f"[{index}/{len(samples)}] error {uid}: {exc}", file=sys.stderr)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    return 1 if failures else 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.input.exists():
        print(f"입력 파일 없음: {args.input}", file=sys.stderr)
        return 2
    try:
        samples = load_samples(args.input)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"입력 오류: {exc}", file=sys.stderr)
        return 2
    if args.limit is not None:
        if args.limit < 1:
            print("--limit은 1 이상이어야 합니다.", file=sys.stderr)
            return 2
        samples = samples[: args.limit]

    manifest = make_manifest(samples, args)
    if not args.execute:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        print("dry-run: API 키와 네트워크를 사용하지 않았습니다.")
        return 0

    output_dir = args.output_dir or _default_output_dir()
    try:
        return execute_run(samples, args, output_dir)
    except RuntimeError as exc:
        print(f"실행 오류: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
