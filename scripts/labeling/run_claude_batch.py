"""Anthropic Message Batches API를 사용한 비동기 배치 라벨링 CLI (비용 50% 절감).

사용법:
  # 1. 배치 생성 및 제출
  python -m scripts.labeling.run_claude_batch --submit --input data/processed/requirements_v0.3.0.jsonl --start 101 --limit 100 --output-dir reports/current/claude_runs/batch_chunk2

  # 2. 배치 진행 상태 확인
  python -m scripts.labeling.run_claude_batch --status --batch-dir reports/current/claude_runs/batch_chunk2

  # 3. 완료 시 결과 다운로드 및 파싱
  python -m scripts.labeling.run_claude_batch --download --batch-dir reports/current/claude_runs/batch_chunk2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.labeling.anchor_pool import load_anchor_pool
from scripts.labeling.anchor_retriever import PureTfidfAnchorRetriever
from scripts.labeling.claude_client import (
    DEFAULT_MODEL,
    PROMPT_VERSION,
    SCHEMA_VERSION,
    SYSTEM_PROMPT,
    ANCHOR_BLOCK_VERSION,
    CONSTANT_ANCHOR_BLOCK_VERSION,
    render_anchor_block,
)
from pydantic import ValidationError

from scripts.labeling.label_schema import LabelResult

DEFAULT_ANCHOR_POOL = ROOT / "data" / "anchors" / "anchor_pool_v3.jsonl"
DEFAULT_INPUT = ROOT / "data" / "processed" / "requirements_v0.3.0.jsonl"


def get_anthropic_client() -> Any:
    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY가 .env 파일에 없습니다.")
    from anthropic import Anthropic

    return Anthropic(api_key=api_key)


# 앵커가 입력과 무관하게 고정된 인출 방식만 앵커 블록을 캐시되는 system 블록에 싣는다.
# 동적 인출은 건마다 앵커가 달라져 캐시 프리픽스가 매번 깨지므로 넣으면 손해다(결정 29).
CACHEABLE_RETRIEVALS = frozenset({"global"})


def build_batch_requests(
    samples: list[dict[str, Any]],
    retriever: PureTfidfAnchorRetriever,
    retrieval: str = "stratified",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    requests = []
    cache_anchors = retrieval in CACHEABLE_RETRIEVALS
    traces = {}
    label_schema = LabelResult.model_json_schema()

    for r in samples:
        uid = r["requirement_uid"]
        name = r["requirement_name"]
        text = r["raw_requirement_text"]

        anchors = retriever.retrieve(r, strategy=retrieval)
        traces[uid] = [
            {
                "requirement_uid": a["requirement_uid"],
                "primary_action": a["primary_action"],
                "similarity": a.get("similarity"),
                "overlap_terms": a.get("overlap_terms"),
            }
            for a in anchors
        ]

        target_block = f"[요구사항 ID]: {uid}\n[요구사항명]: {name}\n[요구사항 내용]:\n{text}"
        # 브레이크포인트를 둘로 나눈다. 기본 프롬프트 블록은 인출 방식과 무관하게 동일해서
        # 다른 전략의 실행과 캐시를 공유하고, 앵커 블록은 그 뒤에서 따로 캐시된다.
        system_blocks: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral", "ttl": "5m"},
            }
        ]
        if cache_anchors:
            user_content = f"[대상 요구사항]\n{target_block}"
            system_blocks.append(
                {
                    "type": "text",
                    "text": render_anchor_block(
                        anchors, show_retrieval_evidence=False
                    ),
                    "cache_control": {"type": "ephemeral", "ttl": "5m"},
                }
            )
        else:
            user_content = (
                f"{render_anchor_block(anchors)}\n\n[대상 요구사항]\n{target_block}"
            )

        params = {
            "model": DEFAULT_MODEL,
            "max_tokens": 16000,
            "thinking": {"type": "adaptive"},
            "output_config": {
                "effort": "medium",
                # 동기 실행(messages.parse)이 내부적으로 만드는 것과 같은 필드다.
                # 배치에는 SDK 헬퍼가 없으므로 요청 본문에 직접 넣는다.
                "format": {"type": "json_schema", "schema": label_schema},
            },
            "system": system_blocks,
            "messages": [{"role": "user", "content": user_content}],
        }

        # Anthropic custom_id pattern: ^[a-zA-Z0-9_-]{1,64}$
        custom_id = uid.replace(":", "__")
        requests.append({"custom_id": custom_id, "params": params})

    return requests, traces


def cmd_submit(args: argparse.Namespace) -> None:
    client = get_anthropic_client()
    pool_rows, pool_meta = load_anchor_pool(args.anchor_pool)
    retriever = PureTfidfAnchorRetriever(pool_rows)

    # Read samples with slicing
    all_samples = []
    with args.input.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                all_samples.append(json.loads(line))

    start_idx = args.start - 1 if args.start and args.start > 0 else 0
    end_idx = start_idx + args.limit if args.limit else len(all_samples)
    samples = all_samples[start_idx:end_idx]

    print(f"총 추출 표본: {len(samples)}건 ({start_idx + 1}번 ~ {min(end_idx, len(all_samples))}번)")
    requests, traces = build_batch_requests(samples, retriever, retrieval=args.retrieval)

    if not args.execute:
        print(f"\n[dry-run] {len(requests)}건의 배치 요청 생성 완료 (네트워크 호출 생략)")
        print(f"실제 제출하려면 --execute 플래그를 추가하세요.")
        return

    output_dir = args.output_dir or ROOT / "reports" / "current" / "claude_batches" / f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Anthropic Message Batches API로 {len(requests)}건 제출 중...")
    batch = client.messages.batches.create(requests=requests)

    batch_info = {
        "batch_id": batch.id,
        "status": batch.processing_status,
        "created_at": batch.created_at.isoformat() if hasattr(batch.created_at, "isoformat") else str(batch.created_at),
        "request_count": len(requests),
        "retrieval": args.retrieval,
        "anchor_block_version": (
            CONSTANT_ANCHOR_BLOCK_VERSION
            if args.retrieval in CACHEABLE_RETRIEVALS
            else ANCHOR_BLOCK_VERSION
        ),
        "anchors_cached_in_system": args.retrieval in CACHEABLE_RETRIEVALS,
        "input_path": str(args.input.resolve()),
        "anchor_pool_meta": pool_meta,
        "traces": traces,
        "sample_uids": [r["requirement_uid"] for r in samples],
    }

    info_path = output_dir / "batch_info.json"
    info_path.write_text(json.dumps(batch_info, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"✅ 배치 제출 완료!")
    print(f"  - Batch ID: {batch.id}")
    print(f"  - 상태: {batch.processing_status}")
    print(f"  - 정보 저장: {info_path}")
    print(f"\n상태 확인 명령어:")
    print(f"  py -3.13 -m scripts.labeling.run_claude_batch --status --batch-dir {output_dir}")


def cmd_status(args: argparse.Namespace) -> None:
    client = get_anthropic_client()
    batch_dir = args.batch_dir
    info_path = batch_dir / "batch_info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"batch_info.json이 없습니다: {info_path}")

    info = json.loads(info_path.read_text(encoding="utf-8"))
    batch_id = info["batch_id"]

    batch = client.messages.batches.retrieve(batch_id)
    counts = getattr(batch, "request_counts", None)

    print(f"=== Batch 상태 ({batch_id}) ===")
    print(f"상태 (Status): {batch.processing_status}")
    if counts:
        print(
            f"진행 상황: 처리중={getattr(counts, 'processing', 0)}, "
            f"성공={getattr(counts, 'succeeded', 0)}, "
            f"오류={getattr(counts, 'errored', 0)}, "
            f"취소={getattr(counts, 'canceled', 0)}, "
            f"만료={getattr(counts, 'expired', 0)}"
        )

    if batch.processing_status == "ended":
        print(f"\n🎉 배치가 완료되었습니다! 결과 다운로드:")
        print(f"  py -3.13 -m scripts.labeling.run_claude_batch --download --batch-dir {batch_dir}")


def cmd_download(args: argparse.Namespace) -> None:
    client = get_anthropic_client()
    batch_dir = args.batch_dir
    info_path = batch_dir / "batch_info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"batch_info.json이 없습니다: {info_path}")

    info = json.loads(info_path.read_text(encoding="utf-8"))
    batch_id = info["batch_id"]
    traces = info.get("traces", {})

    print(f"Batch 결과 스트리밍 다운로드 중 ({batch_id})...")
    results_path = batch_dir / "results.jsonl"
    success_count = 0
    error_count = 0

    with open(results_path, "w", encoding="utf-8") as out_f:
        for result in client.messages.batches.results(batch_id):
            uid = result.custom_id.replace("__", ":")
            res_type = result.result.type

            if res_type == "succeeded":
                message = result.result.message
                # output_config.format을 쓰면 응답은 text 블록에 순수 JSON으로 온다.
                text_blocks = [
                    c.text for c in message.content if getattr(c, "type", "") == "text"
                ]
                # 한 건의 스키마 위반이 전체 다운로드를 중단시키면 안 된다.
                # API 호출은 이미 과금됐으므로 실패 건도 원문과 함께 기록해
                # 재시도와 원인 분석이 가능하게 한다.
                failure = None
                label_obj = None
                if not text_blocks:
                    failure = ("EmptyOutput", "구조화 출력 text 블록 없음", "")
                else:
                    try:
                        label_obj = LabelResult.model_validate_json(text_blocks[0])
                    except ValidationError as exc:
                        failure = (
                            "ValidationError",
                            " ".join(str(exc).split())[:400],
                            text_blocks[0][:2000],
                        )

                if failure is None:
                    record = {
                        "requirement_uid": uid,
                        "status": "ok",
                        "label": label_obj.model_dump(),
                        "anchors_used": traces.get(uid, []),
                        "usage": {
                            "input_tokens": message.usage.input_tokens,
                            "output_tokens": message.usage.output_tokens,
                            "cache_creation_input_tokens": getattr(message.usage, "cache_creation_input_tokens", 0),
                            "cache_read_input_tokens": getattr(message.usage, "cache_read_input_tokens", 0),
                        },
                    }
                    success_count += 1
                else:
                    error_type, error_msg, raw = failure
                    record = {
                        "requirement_uid": uid,
                        "status": "error",
                        "error_type": error_type,
                        "error": error_msg,
                        "raw_output": raw,
                    }
                    error_count += 1
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            else:
                err_msg = str(getattr(result.result, "error", "Batch item failed"))
                out_f.write(json.dumps({"requirement_uid": uid, "status": "error", "error": err_msg}, ensure_ascii=False) + "\n")
                error_count += 1

    print(f"\n결과 저장 완료: {results_path}")
    print(f"  - 성공: {success_count}건")
    print(f"  - 실패: {error_count}건")


def main():
    parser = argparse.ArgumentParser(description="Anthropic Message Batches API CLI")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--submit", action="store_true", help="배치 생성 및 제출")
    group.add_argument("--status", action="store_true", help="배치 상태 조회")
    group.add_argument("--download", action="store_true", help="배치 결과 다운로드")

    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--anchor-pool", type=Path, default=DEFAULT_ANCHOR_POOL)
    parser.add_argument("--start", type=int, default=1, help="시작 행 번호 (1-based)")
    parser.add_argument("--limit", type=int, help="처리할 건수")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--batch-dir", type=Path)
    parser.add_argument("--execute", action="store_true", help="실제 API 호출")
    parser.add_argument(
        "--retrieval",
        choices=["stratified", "similarity", "global"],
        default="stratified",
        help="앵커 인출 방식. global만 앵커 블록이 system에 실려 캐시된다.",
    )

    args = parser.parse_args()

    if args.submit:
        cmd_submit(args)
    elif args.status:
        if not args.batch_dir:
            parser.error("--status에는 --batch-dir가 필요합니다.")
        cmd_status(args)
    elif args.download:
        if not args.batch_dir:
            parser.error("--download에는 --batch-dir가 필요합니다.")
        cmd_download(args)


if __name__ == "__main__":
    main()
