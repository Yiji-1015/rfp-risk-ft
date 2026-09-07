"""Frozen v5 ensemble disagreements: prepare 100 blind requests, run Solar, compare.

Default is offline preparation. --execute calls Upstage; --report only summarizes.
This is an exploratory hybrid inference experiment, not a new label dataset.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
import time

from scripts.evaluation.finetune_ensemble import describe, load_members, vote
from scripts.labeling.claude_client import SYSTEM_PROMPT
from scripts.labeling.label_schema import LabelResult, derive_primary_action
from scripts.labeling.run_solar_labeling import BASE_URL, DEFAULT_MODEL, build_user_content, get_client, read_jsonl

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = ROOT / 'reports/current/solar_runs/disagreement_v5_100_s42'
MEMBERS = ('wc', 'ftB7', 'ftL42')
INPUT_FIELDS = ('requirement_uid', 'requirement_name', 'raw_requirement_text')
PROMPT = SYSTEM_PROMPT + '\n[출력 형식 보충]\nreasoning에 판단 근거인 원문 일부를 「」로 직접 인용한다. 인용을 포함해 300자 이내로 쓴다.\n'
SETTINGS = {'model': DEFAULT_MODEL, 'temperature': 0, 'reasoning_effort': 'low', 'max_tokens': 4096}


def dumps(value):
    return json.dumps(value, ensure_ascii=False, indent=2) + '\n'


def digest(content):
    return hashlib.sha256(content).hexdigest()


def frozen_write(path, text):
    data = text.encode('utf-8')
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f'고정된 실행 조건과 다릅니다: {path.name}. 별도 출력 폴더를 사용하세요.')
    else:
        path.write_bytes(data)


def select_sample(members, documents, size=100, seed=42):
    """Document x vote-type proportional allocation; no gold argument exists."""
    groups = defaultdict(list)
    for uid in sorted(documents):
        count = len({members[name][uid] for name in MEMBERS})
        if count > 1:
            groups[(documents[uid], count)].append(uid)
    total = sum(map(len, groups.values()))
    if not 1 <= size <= total:
        raise ValueError('표본 크기는 1 이상, 전체 이견 건수 이하여야 합니다.')
    quotas = {key: size * len(items) // total for key, items in groups.items()}
    ranked = sorted(groups, key=lambda key: (-(size * len(groups[key]) % total), key))
    for key in ranked[:size - sum(quotas.values())]:
        quotas[key] += 1
    rng = random.Random(seed)
    selected = []
    strata = []
    for key in sorted(groups):
        selected.extend(rng.sample(groups[key], quotas[key]))
        strata.append({'document': key[0], 'distinct_votes': key[1],
                       'eligible': len(groups[key]), 'sample': quotas[key]})
    return sorted(selected), strata


def prepare(directory):
    sources = {
        'labels': ROOT / 'data/labels/label_dataset_v5.jsonl',
        'runs': ROOT / 'reports/current/v5/finetune_runs.jsonl',
        'oof': ROOT / 'reports/current/v5/model_candidate_oof.csv',
    }
    members, gold, documents = load_members(sources['runs'], sources['oof'])
    rows = {r['requirement_uid']: r for r in read_jsonl(sources['labels'])}
    if len(gold) != 1345 or any(rows[u]['primary_action'] != g for u, g in gold.items()):
        raise ValueError('동결 v5 평가 대상 또는 정답이 기존 OOF와 다릅니다.')
    selected, strata = select_sample(members, documents)
    requests = [{key: rows[uid][key] for key in INPUT_FIELDS} for uid in selected]
    uids = sorted(gold)
    baseline = dict(zip(uids, vote(members, MEMBERS, uids)))
    evaluation = [{'requirement_uid': uid, 'document': documents[uid], 'gold': gold[uid],
                   'baseline': baseline[uid], 'votes': [members[n][uid] for n in MEMBERS]}
                  for uid in uids]
    directory.mkdir(parents=True, exist_ok=True)
    files = {'requests.jsonl': ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in requests),
             'evaluation.json': dumps(evaluation), 'system_prompt.txt': PROMPT,
             'schema.json': dumps(LabelResult.model_json_schema())}
    protocol = {
        'experiment': 'solar-disagreement-v5-100-s42-v1', 'dataset': 'v5', 'seed': 42,
        'members': MEMBERS, 'settings': SETTINGS, 'base_url': BASE_URL,
        'sampling': 'document x distinct vote count; proportional largest remainder; without gold',
        'sample_size': len(selected), 'eligible': sum(s['eligible'] for s in strata), 'strata': strata,
        'retrieval': 'none', 'hints': 'none', 'prompt_basis': 'claude-rfp-risk-v5 + quotation format only',
        'adoption': 'valid Solar result -> derive_primary_action; failed/unattempted -> original ensemble',
        'interpretation': 'Exploratory v5 silver-label agreement. v5 used few-shot; Solar is zero-shot. Not a provider-only comparison or independent real-world accuracy.',
        'sources_sha256': {key: digest(path.read_bytes()) for key, path in sources.items()},
        'files_sha256': {key: digest(value.encode('utf-8')) for key, value in files.items()},
    }
    # Refuse to mix changed inputs/settings with a resumable paid run.
    frozen_write(directory / 'protocol.json', dumps(protocol))
    for name, content in files.items():
        frozen_write(directory / name, content)
    print(f'준비 완료: {len(requests)}건 / 이견 {protocol["eligible"]}건 / {len(set(documents.values()))}문서')
    return protocol


def load_bundle(directory):
    protocol = json.loads((directory / 'protocol.json').read_text(encoding='utf-8'))
    for name, expected in protocol['files_sha256'].items():
        if digest((directory / name).read_bytes()) != expected:
            raise ValueError(f'고정 파일이 변경됐습니다: {name}')
    return protocol, read_jsonl(directory / 'requests.jsonl')


def call_one(client, row, prompt, schema, settings):
    record = {'requirement_uid': row['requirement_uid'], 'started_at': datetime.now(timezone.utc).isoformat()}
    start = time.monotonic()
    try:
        response = client.chat.completions.create(
            **settings,
            messages=[{'role': 'system', 'content': prompt},
                      {'role': 'user', 'content': build_user_content({k: row[k] for k in INPUT_FIELDS}, None)}],
            response_format={'type': 'json_schema', 'json_schema': {
                'name': 'label_result', 'schema': schema, 'strict': True}},
        )
        record.update(response_id=response.id, model=response.model,
                      usage=response.usage.model_dump() if response.usage else {},
                      finish_reason=response.choices[0].finish_reason)
        record['content'] = response.choices[0].message.content
        if record['finish_reason'] != 'stop':
            raise ValueError('완료되지 않은 응답')
        label = LabelResult.model_validate_json(record['content'] or '')
        if label.requirement_uid != row['requirement_uid']:
            raise ValueError('응답 UID 불일치')
        derived = derive_primary_action(label)
        quotes = re.findall('「([^」]+)」', label.reasoning)
        record.update(status='ok', label=label.model_dump(), prediction=derived,
                      rule_corrected=derived != label.primary_action,
                      quote_present=bool(quotes),
                      quote_valid=bool(quotes) and all(q in row['raw_requirement_text'] for q in quotes))
    except Exception as exc:
        # Do not put SDK error bodies or credentials in logs.
        record.update(status='error', error_type=type(exc).__name__, http_status=getattr(exc, 'status_code', None))
        causes = []
        cause = exc
        while cause is not None:
            causes.append(type(cause).__name__)
            cause = cause.__cause__
        record['error_causes'] = causes
    record['seconds'] = round(time.monotonic() - start, 3)
    return record


def execute(directory):
    protocol, requests = load_bundle(directory)
    path = directory / 'results.jsonl'
    records = read_jsonl(path) if path.exists() else []
    allowed = {r['requirement_uid'] for r in requests}
    if len({r['requirement_uid'] for r in records}) != len(records) or any(r['requirement_uid'] not in allowed for r in records):
        raise ValueError('기존 결과 UID 중복 또는 표본 밖 결과가 있습니다.')
    # Failed requests also stay recorded; no silent paid retry on restart.
    done = {r['requirement_uid'] for r in records}
    pending = [r for r in requests if r['requirement_uid'] not in done]
    if not pending:
        return
    client = get_client().with_options(timeout=120, max_retries=2)
    prompt = (directory / 'system_prompt.txt').read_text(encoding='utf-8')
    schema = json.loads((directory / 'schema.json').read_text(encoding='utf-8'))
    # At most two in flight, matching the existing Solar runner's rate-limit policy.
    with path.open('a', encoding='utf-8') as handle, ThreadPoolExecutor(max_workers=2) as pool:
        remaining = iter(pending)
        futures = set()
        stop = False
        while futures or not stop:
            while not stop and len(futures) < 2:
                row = next(remaining, None)
                if row is None:
                    break
                futures.add(pool.submit(call_one, client, row, prompt, schema, protocol['settings']))
            if not futures:
                break
            finished, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in finished:
                futures.remove(future)
                record = future.result()
                handle.write(json.dumps(record, ensure_ascii=False) + '\n')
                handle.flush()
                done.add(record['requirement_uid'])
                print(f'{len(done)}/{len(requests)} {record["status"]}', flush=True)
                stop |= record.get('http_status') in (400, 401, 403, 404, 429) or record.get('error_type') in ('APIConnectionError', 'APITimeoutError')
        if stop:
            print('API 오류로 중단. 미호출 건은 기존 예측을 유지합니다.', flush=True)


def summarize(directory):
    protocol, requests = load_bundle(directory)
    evaluation = json.loads((directory / 'evaluation.json').read_text(encoding='utf-8'))
    selected = {r['requirement_uid'] for r in requests}
    path = directory / 'results.jsonl'
    records = read_jsonl(path) if path.exists() else []
    if len({r['requirement_uid'] for r in records}) != len(records) or any(r['requirement_uid'] not in selected for r in records):
        raise ValueError('결과 UID 중복 또는 표본 밖 결과가 있습니다.')
    success = {r['requirement_uid']: r for r in records if r['status'] == 'ok'}
    cases = []
    counts = Counter()
    for row in evaluation:
        uid = row['requirement_uid']
        if uid not in selected:
            continue
        result = success.get(uid)
        after = result['prediction'] if result else row['baseline']
        before_ok, after_ok = row['baseline'] == row['gold'], after == row['gold']
        change = ('kept_correct' if after_ok else 'harmed') if before_ok else ('fixed' if after_ok else 'still_wrong')
        counts[change] += 1
        cases.append({**row, 'after': after, 'change': change, 'api_ok': bool(result),
                      'reasoning': result['label']['reasoning'] if result else None})
    def metrics(rows, use_solar):
        return describe([r['gold'] for r in rows],
                        [success[r['requirement_uid']]['prediction'] if use_solar and r['requirement_uid'] in success else r['baseline'] for r in rows],
                        [r['document'] for r in rows])
    report = {'requested': len(selected), 'attempted': len(records), 'ok': len(success),
              'failed': len(records) - len(success), 'not_attempted': len(selected) - len(records),
              'changes': {k: counts[k] for k in ('fixed', 'harmed', 'kept_correct', 'still_wrong')},
              'net_correct': counts['fixed'] - counts['harmed'],
              'sample_before': metrics(cases, False), 'sample_after': metrics(cases, True),
              'full_oof_before': metrics(evaluation, False),
              'full_oof_after_only_sample_replaced': metrics(evaluation, True),
              'rule_corrections': sum(r['rule_corrected'] for r in success.values()),
              'valid_quotes': sum(r['quote_valid'] for r in success.values()),
              'usage': {k: sum((r.get('usage') or {}).get(k, 0) or 0 for r in records) for k in ('prompt_tokens', 'completion_tokens')},
              'cases': cases, 'interpretation': protocol['interpretation']}
    pricing_path = directory / 'pricing_reference.json'
    if pricing_path.exists():
        pricing = json.loads(pricing_path.read_text(encoding='utf-8'))
        if all(r['started_at'] < pricing['valid_until_utc'] for r in records):
            cost = 0.0
            for record in records:
                usage = record.get('usage') or {}
                cached = (usage.get('prompt_tokens_details') or {}).get('cached_tokens', 0) or 0
                cost += ((usage.get('prompt_tokens', 0) - cached) * pricing['input_per_million']
                         + cached * pricing['cached_input_per_million']
                         + usage.get('completion_tokens', 0) * pricing['output_per_million']) / 1_000_000
            report['recorded_usage_cost_usd_ex_vat'] = round(cost, 6)
            report['billing_note'] = pricing['billing_note']
    (directory / 'comparison.json').write_text(dumps(report), encoding='utf-8')
    state = 'API 호출 전' if not records else ('완료' if len(records) == len(selected) else '일부 실행')
    lines = ['# Solar 이견 100건 파일럿', '', f'상태: **{state}**. 성공 {len(success)}건, 실패 {report["failed"]}건, 미호출 {report["not_attempted"]}건.', '',
             f'고친 답 {counts["fixed"]}건 / 망친 답 {counts["harmed"]}건 / 순증 {report["net_correct"]:+d}건.', '',
             '| 평가 범위 | 기존 macro F1 | 재판정 후 macro F1 |', '|---|---:|---:|']
    for name, before, after in [('표본 100건 통합', report['sample_before']['pooled_macro_f1'], report['sample_after']['pooled_macro_f1']),
                                ('전체 1,345건 fold 평균 — 표본만 교체', report['full_oof_before']['fold_mean_macro_f1'], report['full_oof_after_only_sample_replaced']['fold_mean_macro_f1'])]:
        lines.append(f'| {name} | {before:.4f} | {after:.4f} |')
    lines += ['', '실패·미호출은 기존 예측을 유지한다. 미호출 이견 335건의 개선을 추정하지 않는다.',
              '기존 LLM 생성 v5 라벨의 재현도를 보는 탐색 실험이다. 소형 모델 단독 성능이나 실무 정답률이 아니다.',
              'v5 생성은 few-shot, 이번 Solar 추론은 zero-shot이다. 모델만 바꾼 통제 비교가 아니다.',
              'API 호출 전의 재판정 후 수치는 원래 예측 그대로이며, 실험 결과가 아니다.', '']
    if 'recorded_usage_cost_usd_ex_vat' in report:
        lines += [f'기록된 토큰 기준 추정 비용: **${report["recorded_usage_cost_usd_ex_vat"]:.4f}** (세금 별도).',
                  '중단 요청·SDK 재시도 등 사용량이 남지 않은 호출은 제외했으므로 실제 청구액과 다를 수 있다.',
                  '단가 출처: [Upstage 공식 가격](https://www.upstage.ai/pricing/api), 실행 시점 할인 단가.', '']
    request_by_uid = {r['requirement_uid']: r for r in requests}
    examples = [r for r in cases if r['change'] == 'fixed'][:5] + [r for r in cases if r['change'] == 'harmed'][:5]
    if examples:
        lines += ['## 바뀐 판단 예시', '', 'UID 순으로 고침·악화 각 최대 5건. 실무 정답 여부는 원문과 함께 별도 검토한다.', '']
    for case in examples:
        request = request_by_uid[case['requirement_uid']]
        lines += [f'### {case["requirement_uid"]} — {request["requirement_name"]}', '',
                  f'기존 앙상블: **{case["baseline"]}** / Solar: **{case["after"]}** / 동결 v5: **{case["gold"]}**', '',
                  f'Solar 근거: {case["reasoning"]}', '', '원문:', '',
                  '> ' + request['raw_requirement_text'].replace('\n', '\n> '), '']
    (directory / 'comparison.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'{state}: 성공 {len(success)}건, 고침 {counts["fixed"]}건, 악화 {counts["harmed"]}건')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_DIR)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--report', action='store_true')
    args = parser.parse_args()
    if not args.report:
        prepare(args.output_dir)
    if args.execute:
        execute(args.output_dir)
    summarize(args.output_dir)


if __name__ == '__main__':
    main()
