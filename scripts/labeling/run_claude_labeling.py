"""Run cached Claude RFP labeling. Paid calls require --execute."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from scripts.labeling.anchor_pool import AnchorPoolError, load_anchor_pool
from scripts.labeling.claude_client import (
    ANCHOR_BLOCK_VERSION,
    CONSTANT_ANCHOR_BLOCK_VERSION,
    DEFAULT_MODEL,
    HAIKU_MODEL,
    PROMPT_VERSION,
    PROMPT_VERSIONS,
    ClaudeLabelingClient,
    ClaudeSettings,
)
from scripts.labeling.label_schema import SCHEMA_VERSION

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data" / "samples" / "labeling_pilot_sample_v0.1.0.jsonl"
DEFAULT_ANCHOR_POOL = ROOT / "data" / "anchors" / "anchor_pool_v1.jsonl"

ZERO_SHOT = "zero-shot"
FEWSHOT_SIMILARITY = "fewshot-similarity"
FEWSHOT_STRATIFIED = "fewshot-stratified"
FEWSHOT_GLOBAL = "fewshot-global"
STRATEGIES = (ZERO_SHOT, FEWSHOT_SIMILARITY, FEWSHOT_STRATIFIED, FEWSHOT_GLOBAL)
RETRIEVAL_BY_STRATEGY = {
    FEWSHOT_SIMILARITY: "similarity",
    FEWSHOT_STRATIFIED: "stratified",
    FEWSHOT_GLOBAL: "global",
}

# 앵커가 입력과 무관하게 고정된 전략만 앵커 블록을 캐시되는 system 블록에 싣는다.
# 동적 인출은 매 건 앵커가 달라져서 캐시 프리픽스가 깨지므로 넣으면 손해다(결정 29).
CACHEABLE_ANCHOR_STRATEGIES = frozenset({FEWSHOT_GLOBAL})


def _load_retriever_module():
    """sklearn·scipy는 무겁다. zero-shot 실행과 dry-run이 끌고 오지 않도록 지연 임포트한다."""
    from scripts.labeling import anchor_retriever

    return anchor_retriever


def retriever_config() -> dict[str, Any]:
    return _load_retriever_module().retriever_config()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--strategy",
        choices=STRATEGIES,
        default=ZERO_SHOT,
        help=(
            "라벨링 전략. fewshot-similarity는 유사도 Top-k(결정 13), "
            "fewshot-stratified는 라벨별 1:1:1 층화 인출(결정 14), "
            "fewshot-global은 모든 입력에 같은 고정 앵커(결정 28)."
        ),
    )
    parser.add_argument("--anchor-pool", type=Path, default=DEFAULT_ANCHOR_POOL)
    parser.add_argument(
        "--anchor-top-k",
        type=int,
        default=3,
        help="fewshot-similarity 전용. 층화 전략은 라벨당 1개로 고정된다.",
    )
    parser.add_argument("--model", choices=[DEFAULT_MODEL, HAIKU_MODEL], default=DEFAULT_MODEL)
    parser.add_argument(
        "--effort",
        choices=["low", "medium", "high", "xhigh", "max"],
        default="medium",
    )
    parser.add_argument(
        "--thinking",
        choices=["adaptive", "disabled"],
        default="adaptive",
        help=(
            "Sonnet 5는 생략 시 adaptive로 켜지므로 항상 명시한다. "
            "사고 토큰은 출력 토큰으로 과금된다."
        ),
    )
    parser.add_argument("--cache-ttl", choices=["5m", "1h"], default="5m")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=16000,
        help="사고와 응답을 합친 상한. 상한일 뿐 소비량이 아니다.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--prompt-version",
        choices=sorted(PROMPT_VERSIONS),
        default="v5",
        help="v6는 기준 수행사 범위·범위책임 조건·근거 인용 규칙을 더한 프롬프트다",
    )
    parser.add_argument(
        "--hints",
        action="store_true",
        help="대상 요구사항 뒤에 [드문 표현]·[주목 줄] 블록을 붙인다 (label_hints.py). v6와 함께 쓴다",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="실제 유료 API 호출 실행. 생략하면 manifest만 출력합니다.",
    )
    return parser


def load_samples(path: Path, *, require_document_id: bool = False) -> list[dict[str, Any]]:
    required = {"requirement_uid", "requirement_name", "raw_requirement_text"}
    if require_document_id:
        # 동일 문서 앵커 차단(결정 10)이 document_id 없이는 동작하지 않는다.
        required = required | {"document_id"}
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


def make_manifest(
    samples: Sequence[dict[str, Any]],
    args: argparse.Namespace,
    anchor_pool_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    supports = args.model == DEFAULT_MODEL
    effort = args.effort if supports else None
    thinking = args.thinking if supports else None
    manifest: dict[str, Any] = {
        "execute": args.execute,
        "input": str(args.input.resolve()),
        "sample_count": len(samples),
        "strategy": args.strategy,
        "schema_version": SCHEMA_VERSION,
        "prompt_version": PROMPT_VERSIONS[getattr(args, "prompt_version", "v5")][0],
        "prompt_sha256": hashlib.sha256(
            PROMPT_VERSIONS[getattr(args, "prompt_version", "v5")][1].encode("utf-8")
        ).hexdigest()[:12],
        "hints": bool(getattr(args, "hints", False)),
        "parameters": {
            "model": args.model,
            "effort": effort,
            "thinking": thinking,
            "max_tokens": args.max_tokens,
            "cache_ttl": args.cache_ttl,
            "timeout_seconds": 120.0,
            "max_retries": 2,
            "structured_output": True,
        },
    }
    if args.strategy != ZERO_SHOT:
        manifest["anchoring"] = {
            "anchor_block_version": (
                CONSTANT_ANCHOR_BLOCK_VERSION
                if args.strategy in CACHEABLE_ANCHOR_STRATEGIES
                else ANCHOR_BLOCK_VERSION
            ),
            "retrieval": RETRIEVAL_BY_STRATEGY[args.strategy],
            "anchors_cached_in_system": args.strategy in CACHEABLE_ANCHOR_STRATEGIES,
            "top_k": args.anchor_top_k if args.strategy == FEWSHOT_SIMILARITY else 1,
            "global_anchors": (
                _load_retriever_module().GLOBAL_ANCHORS
                if args.strategy == FEWSHOT_GLOBAL
                else None
            ),
            "retriever": retriever_config(),
            "anchor_pool": anchor_pool_metadata,
        }
    return manifest


def _anchor_preview(
    samples: Sequence[dict[str, Any]],
    retriever: Any,
    args: argparse.Namespace,
) -> str:
    """
    dry-run에서 실제로 주입될 앵커를 미리 보여준다.

    돈을 쓰기 전에 검색 품질과 앵커 라벨 편향(결정 13에서 관측된 견적반영 쏠림)을
    눈으로 확인할 수 있어야 한다.
    """
    retrieval = RETRIEVAL_BY_STRATEGY[args.strategy]
    anchor_counts: Counter[int] = Counter()
    label_counts: Counter[str] = Counter()
    empty_uids: list[str] = []

    for sample in samples:
        anchors = retriever.retrieve(
            sample, strategy=retrieval, top_k=args.anchor_top_k
        )
        anchor_counts[len(anchors)] += 1
        for anchor in anchors:
            label_counts[anchor["primary_action"]] += 1
        if not anchors:
            empty_uids.append(sample["requirement_uid"])

    lines = ["", "[앵커 인출 미리보기]"]
    lines.append(
        "주입 앵커 수 분포: "
        + ", ".join(f"{count}개={n}건" for count, n in sorted(anchor_counts.items()))
    )
    total_anchors = sum(label_counts.values())
    if total_anchors:
        lines.append(
            "주입 앵커 라벨 분포: "
            + ", ".join(
                f"{label} {n}건({n / total_anchors:.1%})"
                for label, n in sorted(label_counts.items())
            )
        )
    if empty_uids:
        lines.append(
            f"앵커 0개로 zero-shot과 동일해지는 요구사항 {len(empty_uids)}건: "
            + ", ".join(empty_uids[:5])
            + (" ..." if len(empty_uids) > 5 else "")
        )
    return "\n".join(lines)


def _default_output_dir() -> Path:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return ROOT / "reports" / "current" / "claude_runs" / run_id


def _write_or_validate_manifest(path: Path, manifest: dict[str, Any]) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        # sample_count는 실험 조건이 아니라 같은 조건의 슬라이스다. --limit로 소규모
        # 확인 후 전체를 이어 돌리는 것이 재개 기능의 정상 사용이므로 비교에서 뺀다.
        comparable_keys = (
            "input",
            "strategy",
            "schema_version",
            "prompt_version",
            "parameters",
            "anchoring",
        )
        if any(existing.get(key) != manifest.get(key) for key in comparable_keys):
            raise RuntimeError(
                "기존 output-dir의 manifest와 실행 조건이 다릅니다. "
                "새 output-dir를 사용하세요."
            )
        if existing.get("sample_count") != manifest.get("sample_count"):
            # 건수가 늘었으면 manifest가 results.jsonl을 더 이상 설명하지 못한다.
            path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        return
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _anchor_trace(anchors: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """앵커 perturbation 분석(§11.12)을 위해 어떤 앵커가 쓰였는지만 남긴다. 본문은 중복이라 뺀다."""
    return [
        {
            "requirement_uid": anchor["requirement_uid"],
            "document_id": anchor.get("document_id"),
            "primary_action": anchor.get("primary_action"),
            "similarity": anchor.get("similarity"),
            "overlap_terms": anchor.get("overlap_terms"),
        }
        for anchor in anchors
    ]


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


def build_retriever(args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    """검토완료 앵커만 담은 검색기와 manifest용 풀 메타데이터를 만든다."""
    pool, metadata = load_anchor_pool(args.anchor_pool)
    if args.strategy in (FEWSHOT_STRATIFIED, FEWSHOT_GLOBAL) and metadata["labels_without_anchor"]:
        # 층화 인출이 성립하지 않으면 결정 14가 막으려던 다수 라벨 편향으로 되돌아간다.
        raise AnchorPoolError(
            "층화 인출에 필요한 라벨의 앵커가 없습니다: "
            f"{metadata['labels_without_anchor']}"
        )
    retriever = _load_retriever_module().PureTfidfAnchorRetriever(pool)
    return retriever, metadata


def execute_run(
    samples: Sequence[dict[str, Any]],
    args: argparse.Namespace,
    output_dir: Path,
    retriever: Any = None,
    anchor_pool_metadata: dict[str, Any] | None = None,
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
        thinking=args.thinking,
        max_tokens=args.max_tokens,
        cache_ttl=args.cache_ttl,
    )
    client = ClaudeLabelingClient(settings=settings, prompt_version=args.prompt_version)
    hint_builder = None
    if args.hints:
        from scripts.labeling.label_hints import HintBuilder

        hint_builder = HintBuilder()
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"
    manifest_path = output_dir / "manifest.json"
    _write_or_validate_manifest(
        manifest_path, make_manifest(samples, args, anchor_pool_metadata)
    )

    completed = _completed_uids(results_path)
    failures = 0
    with results_path.open("a", encoding="utf-8") as handle:
        for index, sample in enumerate(samples, 1):
            uid = sample["requirement_uid"]
            if uid in completed:
                print(f"[{index}/{len(samples)}] skip {uid}")
                continue
            anchors = (
                retriever.retrieve(
                    sample,
                    strategy=RETRIEVAL_BY_STRATEGY[args.strategy],
                    top_k=args.anchor_top_k,
                )
                if retriever is not None
                else []
            )
            hints = hint_builder.for_uid(uid) if hint_builder else None
            try:
                result = client.label_requirement(
                    requirement_uid=uid,
                    requirement_name=sample["requirement_name"],
                    requirement_text=sample["raw_requirement_text"],
                    anchors=anchors,
                    cache_anchors=args.strategy in CACHEABLE_ANCHOR_STRATEGIES,
                    hints=hints,
                )
                record = {
                    "status": "ok",
                    "requirement_uid": uid,
                    "strategy": args.strategy,
                    "input": sample,
                    "hints": hints,
                    "anchors_used": _anchor_trace(anchors),
                    "label": result.label.model_dump(),
                    "metadata": result.metadata,
                }
                # output_tokens에는 사고 토큰이 포함된다. 라벨 자체는 100 토큰 안팎이므로
                # 이 값이 그보다 크게 나오면 그 차이가 사고 비용이다.
                # cache_write가 계속 0이면 프리픽스가 캐시 최소 길이에 미달한 것이다.
                print(
                    f"[{index}/{len(samples)}] ok {uid} "
                    f"anchors={len(anchors)} "
                    f"out={result.metadata['output_tokens']} "
                    f"cache_write={result.metadata['cache_creation_input_tokens']} "
                    f"cache_read={result.metadata['cache_read_input_tokens']}"
                )
            except Exception as exc:  # item failures must not discard prior work
                failures += 1
                record = {
                    "status": "error",
                    "requirement_uid": uid,
                    "strategy": args.strategy,
                    "input": sample,
                    "anchors_used": _anchor_trace(anchors),
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
    uses_anchors = args.strategy != ZERO_SHOT
    if not args.input.exists():
        print(f"입력 파일 없음: {args.input}", file=sys.stderr)
        return 2
    if args.anchor_top_k < 1:
        print("--anchor-top-k는 1 이상이어야 합니다.", file=sys.stderr)
        return 2
    try:
        samples = load_samples(args.input, require_document_id=uses_anchors)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"입력 오류: {exc}", file=sys.stderr)
        return 2
    if args.limit is not None:
        if args.limit < 1:
            print("--limit은 1 이상이어야 합니다.", file=sys.stderr)
            return 2
        samples = samples[: args.limit]

    retriever = None
    anchor_pool_metadata = None
    if uses_anchors:
        # dry-run에서도 앵커 풀을 검증한다. 유료 실행 직전에 풀 문제를 발견하는 것이 가장 비싸다.
        try:
            retriever, anchor_pool_metadata = build_retriever(args)
        except AnchorPoolError as exc:
            print(f"앵커 풀 오류: {exc}", file=sys.stderr)
            return 2

    manifest = make_manifest(samples, args, anchor_pool_metadata)
    if not args.execute:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        if uses_anchors:
            print(_anchor_preview(samples, retriever, args))
        print("dry-run: API 키와 네트워크를 사용하지 않았습니다.")
        return 0

    output_dir = args.output_dir or _default_output_dir()
    try:
        return execute_run(samples, args, output_dir, retriever, anchor_pool_metadata)
    except RuntimeError as exc:
        print(f"실행 오류: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
