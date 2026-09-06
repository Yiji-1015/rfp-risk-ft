"""Upstage Solar로 요구사항을 라벨링한다 (OpenAI 호환 API).

Anthropic 경로(`run_claude_batch.py`)와 **같은 프롬프트·같은 스키마**를 쓰되 모델만
바꾼다. 그래야 나온 라벨의 차이를 모델 탓으로 읽을 수 있다.

Solar에는 배치 API가 없다. 동기 호출을 스레드로 병렬화하고, 중단되면 이어서 돌릴 수
있게 결과를 건별로 append 한다. 이미 성공한 uid는 건너뛴다.

## 캐시

Anthropic은 프리픽스를 자동으로 맞추지만 Solar는 `prompt_cache_key`로 **명시한다.**
시스템 프롬프트가 전 건에 공통이므로 키 하나를 모든 호출에 쓴다. 사전 점검에서
2회차 1,767토큰 중 1,760(99.6%)이 캐시에서 읽혔다.

## 모델 이름

별칭(`solar-pro4`) 대신 **날짜 고정 버전**(`solar-pro4-260806`)을 기본값으로 둔다.
별칭은 나중에 다른 모델을 가리킬 수 있어 재현이 깨진다. 데이터셋을 해시로 동결하는
것과 같은 이유다.

사용법:
  python -m scripts.labeling.run_solar_labeling --input data/processed/requirements_v0.4.0.jsonl
  python -m scripts.labeling.run_solar_labeling --input ... --hints-from <batch_info.json>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import ValidationError

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.labeling.label_schema import LabelResult  # noqa: E402

BASE_URL = "https://api.upstage.ai/v1"
DEFAULT_MODEL = "solar-pro4-260806"
DEFAULT_PROMPT = ROOT / "notebooks" / "prompts" / "system_prompt_v6.txt"
CACHE_KEY = "rfp-risk-ft-system-prompt"

# 동시 호출 수. Upstage 기본 계정은 rate limit이 낮다 — 8로 돌렸을 때 1,445건 중
# 972건이 429로 떨어졌다. 낮게 시작해 필요하면 올린다.
DEFAULT_CONCURRENCY = 2

# 429는 "천천히 하라"는 뜻이다. 곧바로 다시 때리면 상황이 나빠질 뿐이므로
# 지수 백오프로 기다린다. 서버가 `retry-after`를 주면 그 값을 우선한다.
MAX_ATTEMPTS = 6
BASE_BACKOFF_SECONDS = 2.0
MAX_BACKOFF_SECONDS = 60.0


def get_client():
    load_dotenv(ROOT / ".env")
    key = os.getenv("UPSTAGE_API_KEY")
    if not key:
        raise RuntimeError("UPSTAGE_API_KEY가 .env에 없습니다.")
    from openai import OpenAI

    return OpenAI(api_key=key, base_url=BASE_URL)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_user_content(row: dict[str, Any], hint: str | None) -> str:
    """zero-shot 입력. 앵커가 없으므로 대상 요구사항만 싣고 힌트는 뒤에 붙인다."""
    block = (
        f"[요구사항 ID]: {row['requirement_uid']}\n"
        f"[요구사항명]: {row['requirement_name']}\n"
        f"[요구사항 내용]:\n{row['raw_requirement_text']}"
    )
    return f"{block}\n\n{hint}" if hint else block


def label_one(client, model: str, system_prompt: str, row: dict, hint: str | None) -> dict:
    """한 건을 라벨링한다. 실패해도 예외를 올리지 않고 기록으로 남긴다."""
    uid = row["requirement_uid"]
    schema = LabelResult.model_json_schema()
    last_error = ""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": build_user_content(row, hint)},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "label_result", "schema": schema, "strict": True},
                },
                prompt_cache_key=CACHE_KEY,
            )
            text = response.choices[0].message.content
            usage = response.usage
            details = getattr(usage, "prompt_tokens_details", None)
            return {
                "requirement_uid": uid,
                "status": "ok",
                "label": LabelResult.model_validate_json(text).model_dump(),
                "hints": hint,
                "prompt_version": "claude-rfp-risk-v6",
                "model": model,
                "anchors_used": [],
                "usage": {
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "cached_tokens": getattr(details, "cached_tokens", 0) or 0,
                },
                "attempts": attempt,
            }
        except ValidationError as exc:
            # 스키마 위반은 재시도해도 같을 가능성이 높지만, 출력 흔들림이 있으므로 몇 번 더 준다.
            last_error = f"ValidationError: {' '.join(str(exc).split())[:300]}"
            _sleep_backoff(attempt)
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {str(exc)[:300]}"
            _sleep_backoff(attempt, exc)
    return {"requirement_uid": uid, "status": "error", "error": last_error, "model": model}


def _sleep_backoff(attempt: int, exc: Exception | None = None) -> None:
    """다음 시도까지 기다린다. 서버가 알려준 `retry-after`가 있으면 그것을 따른다."""
    wait = min(BASE_BACKOFF_SECONDS * (2 ** (attempt - 1)), MAX_BACKOFF_SECONDS)
    response = getattr(exc, "response", None)
    header = getattr(response, "headers", {}) or {}
    retry_after = header.get("retry-after") or header.get("Retry-After")
    if retry_after:
        try:
            wait = max(wait, float(retry_after))
        except ValueError:
            pass
    # 같은 순간에 풀린 스레드가 한꺼번에 다시 때리지 않도록 흩는다.
    time.sleep(wait + random.uniform(0, wait * 0.25))


def main() -> int:
    parser = argparse.ArgumentParser(description="Upstage Solar 라벨링")
    parser.add_argument("--input", type=Path, required=True, help="요구사항 jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument(
        "--hints-from", type=Path,
        help="힌트를 담은 batch_info.json. 없으면 힌트 없이 돌린다.",
    )
    parser.add_argument("--limit", type=int, help="앞에서 N건만")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--execute", action="store_true", help="실제 API 호출")
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    if args.limit:
        rows = rows[: args.limit]
    system_prompt = args.prompt.read_text(encoding="utf-8")
    prompt_sha256 = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()

    hints: dict[str, str] = {}
    if args.hints_from:
        hints = json.loads(args.hints_from.read_text(encoding="utf-8")).get("hints", {})

    results_path = args.output_dir / "results.jsonl"
    done: set[str] = set()
    if results_path.exists():
        done = {r["requirement_uid"] for r in read_jsonl(results_path) if r.get("status") == "ok"}
    pending = [r for r in rows if r["requirement_uid"] not in done]

    print(f"대상 {len(rows)}건 · 완료 {len(done)}건 · 남은 {len(pending)}건")
    print(f"모델 {args.model} · 프롬프트 {args.prompt.name} (sha256 {prompt_sha256[:12]}…)")
    print(f"힌트 {len(hints)}건" + (" (전건 적용 아님)" if hints and len(hints) < len(rows) else ""))

    if not args.execute:
        print("\n[dry-run] 네트워크 호출 없음. 실제 실행은 --execute 를 붙이세요.")
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "run_info.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "prompt_file": str(args.prompt),
                "prompt_sha256": prompt_sha256,
                "prompt_version": "claude-rfp-risk-v6",
                "retrieval": "none",
                "hints_from": str(args.hints_from) if args.hints_from else None,
                "hint_count": len(hints),
                "input_path": str(args.input.resolve()),
                "request_count": len(rows),
                "started_at": datetime.now(timezone.utc).isoformat(),
                "base_url": BASE_URL,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    client = get_client()
    lock = threading.Lock()
    counts = {"ok": 0, "error": 0}
    with results_path.open("a", encoding="utf-8") as handle:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {
                pool.submit(label_one, client, args.model, system_prompt, row, hints.get(row["requirement_uid"])): row
                for row in pending
            }
            for index, future in enumerate(as_completed(futures), start=1):
                record = future.result()
                with lock:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()
                    counts["ok" if record["status"] == "ok" else "error"] += 1
                if index % 50 == 0 or index == len(pending):
                    print(f"  {index}/{len(pending)} · 성공 {counts['ok']} · 실패 {counts['error']}")

    print(f"\n결과: {results_path}")
    print(f"  성공 {counts['ok']}건 · 실패 {counts['error']}건")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
