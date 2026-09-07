"""문서 맥락을 주면 경계 이견이 풀리는가 — Claude 재판정, 맥락 있음 대 없음.

2026-09-07 19:20 진단: 경계 혼동은 "뜻이 같은 조항이 다른 문서에서 다른 라벨을 받은 자리"이고
그 차이는 문서 맥락(망 환경·예산·계약 구조)에서 온다. Solar 파일럿(18:08)은 원문만 다시
읽히면 16건 고치고 17건 망친다는 대조군을 남겼다. 여기서는 같은 100건을 Claude로
두 번 판정한다.

- plain   : v5 시스템 프롬프트 + 원문. zero-shot. (Solar 파일럿과 같은 입력, 모델만 Claude)
- context : plain + `data/context/document_context_v1.jsonl`의 문서 맥락 카드 한 장.

두 팔의 차이는 맥락 블록 하나다. 실패 건은 기존 앙상블 예측을 유지한다(파일럿과 같은 규칙).

    python -m scripts.evaluation.context_rejudge                 # 준비·미리보기 (호출 없음)
    python -m scripts.evaluation.context_rejudge --execute       # 두 팔 모두 호출
    python -m scripts.evaluation.context_rejudge --report        # 결과 집계만
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.evaluation.finetune_ensemble import describe, load_members, vote
from scripts.labeling.claude_client import (
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    ClaudeLabelingClient,
    ClaudeResponseError,
    ClaudeSettings,
)
from scripts.labeling.label_schema import LabelResult, derive_primary_action

ROOT = Path(__file__).resolve().parents[2]
PILOT_DIR = ROOT / "reports/current/solar_runs/disagreement_v5_100_s42"
OUT_DIR = ROOT / "reports/current/v5/context_rejudge_100_s42"
CONTEXT_PATH = ROOT / "data/context/document_context_v1.jsonl"
MEMBERS = ("wc", "ftB7", "ftL42")
ARMS = ("plain", "context")

CONTEXT_SUPPLEMENT = """
[문서 맥락 보충]
아래 [문서 맥락]은 이 요구사항이 실린 제안요청서 전체에서 확인한 사실이다. 판단 규칙 1의
"문서에 없는 정보를 추측하지 않는다"는 그대로 두되, 이 블록에 적힌 사실은 추측이 아니라
확인된 정보이므로 blocker·원가 판단의 근거로 써도 된다. 맥락에 없는 것은 여전히 추측하지 않는다.
"""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def render_context(card: dict[str, Any]) -> str:
    budget = card["budget_note"] if card["budget_krw"] is None else f"{card['budget_krw']:,}원 ({card['budget_note']})"
    lines = [
        "[문서 맥락]",
        f"- 발주기관: {card['agency']} ({card['agency_type']})",
        f"- 사업: {card['project_type']}",
        f"- 사업비: {budget}",
        f"- 사업기간: {card['period']}",
        f"- 망 환경: {card['network']}",
        f"- 인프라·도입: {card['infra']}",
        f"- 조달·계약 조건: {card['procurement']}",
        f"- 업무 도메인: {card['domain']}",
        f"- 과업 요약: {card['scope_summary']}",
    ]
    return "\n".join(lines)


def build_request(arm: str, row: dict[str, Any], card: dict[str, Any], settings: ClaudeSettings) -> dict[str, Any]:
    target = (
        f"[요구사항 ID]: {row['requirement_uid']}\n"
        f"[요구사항명]: {row['requirement_name']}\n"
        f"[요구사항 내용]:\n{row['raw_requirement_text']}"
    )
    cache = {"type": "ephemeral", "ttl": settings.cache_ttl}
    system = [{"type": "text", "text": SYSTEM_PROMPT, "cache_control": cache}]
    if arm == "context":
        system.append({"type": "text", "text": CONTEXT_SUPPLEMENT, "cache_control": cache})
        user = f"{render_context(card)}\n\n[대상 요구사항]\n{target}"
    else:
        user = target
    request: dict[str, Any] = {
        "model": settings.model, "max_tokens": settings.max_tokens, "system": system,
        "messages": [{"role": "user", "content": user}], "output_format": LabelResult,
    }
    if settings.supports_thinking_and_effort:
        request["output_config"] = {"effort": settings.effort}
        request["thinking"] = {"type": settings.thinking}
    return request


def call_one(client: ClaudeLabelingClient, arm: str, row: dict, card: dict, settings: ClaudeSettings) -> dict[str, Any]:
    record: dict[str, Any] = {"arm": arm, "requirement_uid": row["requirement_uid"],
                              "started_at": datetime.now(timezone.utc).isoformat()}
    started = time.perf_counter()
    try:
        response = client._get_client().messages.parse(**build_request(arm, row, card, settings))
        usage = getattr(response, "usage", None)
        record["usage"] = {k: getattr(usage, k, None) for k in
                           ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")} if usage else {}
        record["stop_reason"] = response.stop_reason
        if response.stop_reason != "end_turn":
            raise ClaudeResponseError(f"stop_reason={response.stop_reason}")
        parsed = response.parsed_output
        label = parsed if isinstance(parsed, LabelResult) else LabelResult.model_validate(parsed)
        if label.requirement_uid != row["requirement_uid"]:
            raise ClaudeResponseError("UID 불일치")
        record.update(status="ok", label=label.model_dump(), derived=derive_primary_action(label))
    except Exception as exc:  # 실패도 기록한다. 재개 시 다시 호출한다.
        record.update(status="error", error=f"{type(exc).__name__}: {exc}")
    record["latency_seconds"] = round(time.perf_counter() - started, 2)
    return record


def load_eval() -> tuple[list[dict], dict, dict, dict]:
    members, gold, documents = load_members(ROOT / "reports/current/v5/finetune_runs.jsonl",
                                            ROOT / "reports/current/v5/model_candidate_oof.csv")
    uids = sorted(gold)
    baseline = dict(zip(uids, vote(members, MEMBERS, uids)))
    requests = read_jsonl(PILOT_DIR / "requests.jsonl")
    return requests, gold, documents, baseline


def summarize(arm: str, results: dict[str, dict], requests, gold, documents, baseline) -> dict[str, Any]:
    uids = sorted(gold)
    sample = [r["requirement_uid"] for r in requests]
    replaced = dict(baseline)
    ok = fixed = broken = 0
    for u in sample:
        r = results.get(u)
        if r and r["status"] == "ok":
            ok += 1
            new = r["derived"]
            if new != baseline[u]:
                if new == gold[u]:
                    fixed += 1
                elif baseline[u] == gold[u]:
                    broken += 1
            replaced[u] = new
    g = [gold[u] for u in uids]
    d = [documents[u] for u in uids]
    before = describe(g, [baseline[u] for u in uids], d)
    after = describe(g, [replaced[u] for u in uids], d)
    gs = [gold[u] for u in sample]
    sample_before = describe(gs, [baseline[u] for u in sample], [documents[u] for u in sample])
    sample_after = describe(gs, [replaced[u] for u in sample], [documents[u] for u in sample])
    cited = sum(1 for u in sample if (r := results.get(u)) and r["status"] == "ok"
                and any(q in next(x["raw_requirement_text"] for x in requests if x["requirement_uid"] == u)
                        for q in _quotes(r["label"]["reasoning"])))
    return {
        "arm": arm, "called": len(sample), "ok": ok, "failed": len(sample) - ok,
        "fixed": fixed, "broken": broken, "net": fixed - broken,
        "sample_pooled_macro_f1": [sample_before["pooled_macro_f1"], sample_after["pooled_macro_f1"]],
        "full_fold_mean_macro_f1": [before["fold_mean_macro_f1"], after["fold_mean_macro_f1"]],
        "full_pooled_macro_f1": [before["pooled_macro_f1"], after["pooled_macro_f1"]],
        "full_errors": [before["errors"], after["errors"]],
        "full_boundary_errors": [before["boundary_errors"], after["boundary_errors"]],
        "quotes_found_in_text": cited,
        "tokens": {k: sum((r.get("usage") or {}).get(k) or 0 for r in results.values()) for k in
                   ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")},
    }


def _quotes(text: str) -> list[str]:
    import re
    return [q for q in re.findall(r"「([^」]{4,})」", text)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="실제 API 호출")
    parser.add_argument("--report", action="store_true", help="집계만")
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--effort", default="medium")
    args = parser.parse_args()
    arms = [a for a in args.arms.split(",") if a in ARMS]

    cards = {c["document_id"]: c for c in read_jsonl(CONTEXT_PATH)}
    requests, gold, documents, baseline = load_eval()
    assert all(documents[r["requirement_uid"]] in cards for r in requests), "맥락 카드가 없는 문서가 있다"
    settings = ClaudeSettings(effort=args.effort)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    protocol = {
        "experiment": "context-rejudge-v5-100-s42", "sample_from": str(PILOT_DIR.relative_to(ROOT)),
        "sample_size": len(requests), "arms": list(ARMS), "members": MEMBERS,
        "settings": {"model": settings.model, "effort": settings.effort, "thinking": settings.thinking,
                     "max_tokens": settings.max_tokens, "prompt_version": PROMPT_VERSION},
        "context_sha256": hashlib.sha256(CONTEXT_PATH.read_bytes()).hexdigest(),
        "supplement_sha256": hashlib.sha256(CONTEXT_SUPPLEMENT.encode()).hexdigest(),
        "adoption": "ok -> derive_primary_action; error -> original ensemble",
        "caveat": "v5 라벨은 문서 맥락 없이(판단 규칙 1) 생성됐다. 맥락 팔이 v5와 멀어지는 것은 오류일 수도, 더 나은 판정일 수도 있다.",
    }
    (OUT_DIR / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT_DIR / "context_cards.txt").write_text("\n\n".join(render_context(cards[d]) for d in sorted(cards)), encoding="utf-8")

    if not args.execute and not args.report:
        sample_doc = documents[requests[0]["requirement_uid"]]
        print(f"표본 {len(requests)}건 / 팔 {arms} / 모델 {settings.model} effort={settings.effort}")
        print(f"맥락 카드 {len(cards)}장 → {OUT_DIR / 'context_cards.txt'}")
        print("\n--- context 팔 user 메시지 예시 (첫 건) ---")
        print(build_request("context", requests[0], cards[sample_doc], settings)["messages"][0]["content"][:1500])
        print("\n호출하지 않았습니다. --execute 로 실행합니다.")
        return

    if args.execute:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise SystemExit("ANTHROPIC_API_KEY가 없습니다. .env를 두거나 환경변수로 넘기세요. 호출하지 않았습니다.")

    results: dict[str, dict[str, dict]] = {}
    for arm in arms:
        path = OUT_DIR / f"results_{arm}.jsonl"
        results[arm] = {r["requirement_uid"]: r for r in (read_jsonl(path) if path.exists() else []) if r["status"] == "ok"}
        if args.execute:
            todo = [r for r in requests if r["requirement_uid"] not in results[arm]]
            print(f"[{arm}] 기존 성공 {len(results[arm])}건, 호출 {len(todo)}건")
            client = ClaudeLabelingClient(settings)
            with path.open("a", encoding="utf-8") as handle, ThreadPoolExecutor(args.workers) as pool:
                futures = [pool.submit(call_one, client, arm, row, cards[documents[row["requirement_uid"]]], settings) for row in todo]
                for n, fut in enumerate(as_completed(futures), 1):
                    rec = fut.result()
                    handle.write(json.dumps(rec, ensure_ascii=False) + "\n"); handle.flush()
                    if rec["status"] == "ok":
                        results[arm][rec["requirement_uid"]] = rec
                    print(f"  {n}/{len(todo)} {rec['requirement_uid']} {rec['status']} {rec.get('derived', rec.get('error', ''))[:60]}", flush=True)

    summary = {arm: summarize(arm, results[arm], requests, gold, documents, baseline) for arm in arms}
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n{'팔':<8}{'성공':>5}{'고침':>5}{'망침':>5}{'순증':>5}{'표본 통합':>16}{'전체 fold평균':>18}{'오답':>12}{'경계':>12}{'인용 일치':>8}")
    for arm, s in summary.items():
        print(f"{arm:<8}{s['ok']:>5}{s['fixed']:>5}{s['broken']:>5}{s['net']:>+5}"
              f"{s['sample_pooled_macro_f1'][0]:>8.3f}→{s['sample_pooled_macro_f1'][1]:<7.3f}"
              f"{s['full_fold_mean_macro_f1'][0]:>9.4f}→{s['full_fold_mean_macro_f1'][1]:<8.4f}"
              f"{s['full_errors'][0]:>5}→{s['full_errors'][1]:<6}{s['full_boundary_errors'][0]:>5}→{s['full_boundary_errors'][1]:<6}{s['quotes_found_in_text']:>6}")
    print("Solar 파일럿(원문만, 18:08): 성공 82, 고침 16, 망침 17, 표본 0.588→0.565, 전체 0.6714→0.6731")
    print(f"저장: {OUT_DIR}")


if __name__ == "__main__":
    main()
