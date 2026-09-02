import json

import pytest

from scripts.labeling.label_dataset import (
    DEFAULT_PATH,
    EXPECTED_ROWS,
    FROZEN_SHA256,
    LabelDatasetError,
    content_digest,
    get_model_text,
    load_label_dataset,
)


def test_frozen_dataset_passes_its_own_audit():
    """동결 파일이 감사를 통과해야 한다. 실패하면 파일이 바뀐 것이다."""
    rows, meta = load_label_dataset()

    assert len(rows) == EXPECTED_ROWS
    assert meta['sha256'] == FROZEN_SHA256
    assert meta['document_count'] == 10
    assert sum(meta['label_counts'].values()) == EXPECTED_ROWS
    assert all(r['model_text'].startswith(r['requirement_name'] + '\n') for r in rows)


def test_environment_switch_loads_v3_raw_text(monkeypatch):
    monkeypatch.setenv('RFP_DATASET_VERSION', 'v3')

    rows, meta = load_label_dataset()

    assert meta['dataset_version'] == 'label_dataset_v3'
    assert get_model_text(rows[0]) == rows[0]['raw_requirement_text']


def test_v4_uses_explicit_model_text():
    rows, meta = load_label_dataset(version='v4')

    assert meta['dataset_version'] == 'label_dataset_v4'
    assert get_model_text(rows[0]) == rows[0]['model_text']


def test_unknown_dataset_version_is_rejected():
    with pytest.raises(LabelDatasetError, match='알 수 없는 데이터셋 버전'):
        load_label_dataset(version='v99')


def _valid_row(uid='doc_a:R-1'):
    rows, _ = load_label_dataset()
    row = dict(rows[0])
    row['requirement_uid'] = uid
    return row


def _write(tmp_path, rows):
    p = tmp_path / 'ds.jsonl'
    p.write_text(
        ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows),
        encoding='utf-8',
    )
    return p


def test_modified_file_is_rejected(tmp_path):
    """
    동결의 요점은 조용한 변경을 막는 것이다. 내용이 유효해도 해시가 다르면 거부한다.
    분석마다 다른 라벨을 보고 있으면 결과를 비교할 수 없기 때문이다.
    """
    rows, _ = load_label_dataset()
    tampered = [dict(r) for r in rows]
    # 값 도메인은 그대로 유효한 변경이다. 구조 검사로는 잡히지 않고 해시만 잡는다.
    tampered[0]['primary_action'] = '견적반영'
    path = _write(tmp_path, tampered)

    with pytest.raises(LabelDatasetError, match='동결 상태와 다릅니다'):
        load_label_dataset(path)


def test_byte_identical_rebuild_passes(tmp_path):
    """빌더가 결정적이라 같은 입력에서 같은 바이트가 나온다. 재생성이 동결을 깨지 않는다."""
    rows, _ = load_label_dataset()
    path = _write(tmp_path, rows)

    _, meta = load_label_dataset(path)
    assert meta['sha256'] == FROZEN_SHA256


def test_unknown_label_value_is_rejected(tmp_path):
    rows, _ = load_label_dataset()
    broken = [dict(r) for r in rows]
    broken[0]['primary_action'] = '보류'
    path = _write(tmp_path, broken)

    with pytest.raises(LabelDatasetError, match='알 수 없는 primary_action'):
        load_label_dataset(path, verify_hash=False)


def test_ragged_schema_is_rejected(tmp_path):
    """균일 스키마가 이 데이터셋의 존재 이유다. 한 행만 키가 달라도 거부한다."""
    rows, _ = load_label_dataset()
    broken = [dict(r) for r in rows]
    broken[0] = {k: v for k, v in broken[0].items() if k != 'reasoning'}
    path = _write(tmp_path, broken)

    with pytest.raises(LabelDatasetError):
        load_label_dataset(path, verify_hash=False)


def test_duplicate_uid_is_rejected(tmp_path):
    rows, _ = load_label_dataset()
    broken = [dict(r) for r in rows]
    broken[1]['requirement_uid'] = broken[0]['requirement_uid']
    path = _write(tmp_path, broken)

    with pytest.raises(LabelDatasetError, match='중복'):
        load_label_dataset(path, verify_hash=False)


def test_row_count_change_is_rejected(tmp_path):
    rows, _ = load_label_dataset()
    path = _write(tmp_path, rows[:-1])

    with pytest.raises(LabelDatasetError, match='건입니다'):
        load_label_dataset(path, verify_hash=False)


def test_nullable_fields_are_tolerated():
    """
    agency 67건은 비어 있다. 허용하되 메타에 세어 남긴다(docs/issues/002).
    """
    _, meta = load_label_dataset()

    assert meta['nullable_missing']['agency'] == 67
    assert meta['nullable_missing']['requirement_type'] == 0
    assert meta['nullable_missing']['domain'] == 0


def test_digest_ignores_the_line_ending_the_platform_checked_out():
    """같은 내용이면 CRLF든 LF든 같은 해시여야 한다.

    Windows는 `core.autocrlf`로 CRLF를, 리눅스는 LF를 받는다. 표기 차이로 동결 대조가
    실패하면 같은 저장소를 리눅스에서 clone한 것만으로 학습이 막힌다(실제로 gcube
    컨테이너에서 그렇게 막혔다).
    """
    crlf = b'{"a": 1}\r\n{"b": 2}\r\n'
    lf = b'{"a": 1}\n{"b": 2}\n'

    assert content_digest(crlf) == content_digest(lf)
    # 내용이 실제로 다르면 여전히 갈라져야 한다.
    assert content_digest(lf) != content_digest(b'{"a": 1}\n{"b": 3}\n')


def test_frozen_dataset_loads_from_a_lf_checkout(tmp_path):
    """리눅스 체크아웃과 같은 바이트를 만들어도 로더가 통과해야 한다."""
    source = DEFAULT_PATH.read_bytes()
    lf_copy = tmp_path / "label_dataset_v4.jsonl"
    lf_copy.write_bytes(source.replace(b"\r\n", b"\n"))

    rows, meta = load_label_dataset(lf_copy)

    assert len(rows) == EXPECTED_ROWS
    assert meta["sha256"] == FROZEN_SHA256
