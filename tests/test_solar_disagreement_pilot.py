import json
from types import SimpleNamespace

import pytest

from scripts.evaluation import solar_disagreement_pilot as pilot
from scripts.evaluation.solar_disagreement_pilot import (
    MEMBERS, SETTINGS, call_one, frozen_write, prepare, select_sample, summarize,
)
from scripts.labeling.label_schema import LabelResult


def test_blind_request_schema_uid_and_rule_correction():
    row = dict(requirement_uid='d:1', requirement_name='교육', raw_requirement_text='교육을 제공한다.',
               gold='DO_NOT_SEND_GOLD', reasoning='DO_NOT_SEND_REASONING', votes='DO_NOT_SEND_VOTES')
    label = dict(requirement_uid='d:1', primary_action='통상수용', blockers=[], cost_basis='고급·전문인력',
                 domain_dependency='낮음', build_difficulty='보통', reasoning='「교육을 제공한다.」는 전문인력이 필요하다.')
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(id='fake', model=SETTINGS['model'], usage=None,
                               choices=[SimpleNamespace(finish_reason='stop', message=SimpleNamespace(content=json.dumps(label)))])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    result = call_one(client, row, 'fixed rubric', LabelResult.model_json_schema(), SETTINGS)
    payload = json.dumps(calls[0], ensure_ascii=False)
    assert not any(value in payload for value in ('DO_NOT_SEND_GOLD', 'DO_NOT_SEND_REASONING', 'DO_NOT_SEND_VOTES'))
    assert result['prediction'] == '견적반영' and result['rule_corrected'] and result['quote_valid']
    label['requirement_uid'] = 'wrong'
    assert call_one(client, row, 'fixed rubric', LabelResult.model_json_schema(), SETTINGS)['status'] == 'error'


def test_sample_reproducibility_and_frozen_run(tmp_path):
    documents = {f'd{i}:u{j}': f'd{i}' for i in range(4) for j in range(12)}
    members = {name: {uid: ('통상수용' if i == 0 else '견적반영') for uid in documents}
               for i, name in enumerate(MEMBERS)}
    selected, strata = select_sample(members, documents, size=20)
    assert (selected, strata) == select_sample(members, dict(reversed(list(documents.items()))), size=20)
    assert len(set(selected)) == 20
    assert {documents[u] for u in selected} == set(documents.values())
    assert sum(s['sample'] for s in strata) == 20
    path = tmp_path / 'frozen.json'
    frozen_write(path, 'one')
    frozen_write(path, 'one')
    with pytest.raises(ValueError, match='고정'):
        frozen_write(path, 'two')


def test_real_sample_evaluation_counts_harm_and_keeps_uncalled_predictions(tmp_path):
    prepare(tmp_path)
    requests = [json.loads(line) for line in (tmp_path / 'requests.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(requests) == 100 and all(set(r) == {'requirement_uid', 'requirement_name', 'raw_requirement_text'} for r in requests)
    selected = {r['requirement_uid'] for r in requests}
    rows = json.loads((tmp_path / 'evaluation.json').read_text(encoding='utf-8'))
    wrong = next(r for r in rows if r['requirement_uid'] in selected and r['gold'] != r['baseline'])
    right = next(r for r in rows if r['requirement_uid'] in selected and r['gold'] == r['baseline'])
    records = []
    for row, after in [(wrong, wrong['gold']), (right, next(label for label in ('통상수용', '견적반영', '계약·질의검토') if label != right['gold']))]:
        records.append(dict(requirement_uid=row['requirement_uid'], status='ok', prediction=after,
                            label={'reasoning': 'fake'}, rule_corrected=False, quote_valid=False))
    (tmp_path / 'results.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records), encoding='utf-8')
    report = summarize(tmp_path)
    assert report['changes']['fixed'] == report['changes']['harmed'] == 1
    assert report['not_attempted'] == 98 and report['net_correct'] == 0
    assert report['full_oof_before']['errors'] == report['full_oof_after_only_sample_replaced']['errors'] == 356


def test_resume_does_not_rebill_recorded_rows_and_stops_after_bad_credentials(tmp_path, monkeypatch):
    prepare(tmp_path)
    requests = [json.loads(line) for line in (tmp_path / 'requests.jsonl').read_text(encoding='utf-8').splitlines()]
    first = {'requirement_uid': requests[0]['requirement_uid'], 'status': 'error'}
    (tmp_path / 'results.jsonl').write_text(json.dumps(first) + '\n', encoding='utf-8')
    calls = []
    def failed_call(client, row, prompt, schema, settings):
        calls.append(row['requirement_uid'])
        return {'requirement_uid': row['requirement_uid'], 'status': 'error', 'http_status': 401}
    monkeypatch.setattr(pilot, 'get_client', lambda: SimpleNamespace(with_options=lambda **kw: None))
    monkeypatch.setattr(pilot, 'call_one', failed_call)
    pilot.execute(tmp_path)
    assert set(calls) == {r['requirement_uid'] for r in requests[1:3]}
    assert len((tmp_path / 'results.jsonl').read_text(encoding='utf-8').splitlines()) == 3
