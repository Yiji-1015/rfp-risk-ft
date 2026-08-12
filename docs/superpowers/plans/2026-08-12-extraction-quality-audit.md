# Extraction Quality Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 목록-상세표 대응과 원본 표본 검수를 완료해 요구사항 데이터셋 v0.2를 라벨링 가능한 상태로 동결한다.

**Architecture:** 기존 HTML 표 파서와 상세 요구사항 추출기는 유지한다. 별도의 목록 ID 추출기와 감사 함수를 추가해 문서별 목록 ID·상세 ID 차이를 구조화하고, 자동 감사 결과와 사람 검수 기록을 합쳐 동결 여부를 결정한다.

**Tech Stack:** Python 표준 라이브러리, `unittest`, CSV/JSONL, Markdown 감사 보고서

## Global Constraints

- 요구사항 ID 하나를 데이터 행 하나로 유지한다.
- 원문 텍스트를 요약하거나 다시 쓰지 않는다.
- PDF·HWP·HWPX 원본 대조가 끝나기 전에는 학습 데이터로 확정하지 않는다.
- 생성 파일의 버전은 `v0.2.0`으로 올리고 기존 `v0.1.0` 보고서는 보존한다.

---

### Task 1: 목록 ID와 상세 ID 자동 대조

**Files:**
- Create: `tests/test_build_dataset.py`
- Modify: `scripts/build_dataset.py`

**Interfaces:**
- Consumes: `parse_tables(markdown: str) -> list[Table]`, `extract_document(path: Path, metadata: dict) -> list[dict]`
- Produces: `extract_index_requirement_ids(markdown: str) -> list[str]`, `compare_index_and_detail_ids(index_ids: list[str], detail_ids: list[str]) -> dict`

- [ ] **Step 1: 목록표와 상세표를 구분하는 실패 테스트 작성**

```python
import unittest

from scripts.build_dataset import (
    compare_index_and_detail_ids,
    extract_index_requirement_ids,
)


class RequirementIndexAuditTests(unittest.TestCase):
    def test_extracts_ids_only_from_index_table(self):
        markdown = """
        <h2>요구사항 목록</h2>
        <table><tr><th>번호</th><th>요구사항 ID</th><th>명칭</th></tr>
        <tr><td>1</td><td>SFR-001</td><td>검색</td></tr>
        <tr><td>2</td><td>SFR-002</td><td>응답</td></tr></table>
        <h2>요구사항 세부내용</h2>
        <table><tr><td>요구사항 고유번호</td><td>SFR-001</td></tr>
        <tr><td>요구사항 명칭</td><td>검색</td></tr>
        <tr><td>요구사항 내용</td><td>검색 기능을 구축한다.</td></tr></table>
        """
        self.assertEqual(extract_index_requirement_ids(markdown), ["SFR-001", "SFR-002"])

    def test_reports_both_missing_directions(self):
        result = compare_index_and_detail_ids(
            ["SFR-001", "SFR-002"], ["SFR-001", "SFR-003"]
        )
        self.assertEqual(result["index_only_ids"], ["SFR-002"])
        self.assertEqual(result["detail_only_ids"], ["SFR-003"])
        self.assertFalse(result["is_exact_match"])
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `python -m unittest tests.test_build_dataset -v`

Expected: 두 함수 import 실패

- [ ] **Step 3: 목록 구간 추출과 집합 대조 구현**

`extract_index_requirement_ids`는 `요구사항 목록`, `요구사항 목록표`, `요구사항 총괄표` 제목 이후부터 `요구사항 세부`, `요구사항 상세` 제목 전까지의 HTML 표만 파싱한다. 각 셀에서 `ID_RE.finditer()`로 ID를 수집하고 입력 순서를 유지하면서 중복을 제거한다. `compare_index_and_detail_ids`는 다음 형태를 반환한다.

```python
{
    "index_count": len(index_ids),
    "detail_count": len(detail_ids),
    "index_only_ids": sorted(set(index_ids) - set(detail_ids)),
    "detail_only_ids": sorted(set(detail_ids) - set(index_ids)),
    "duplicate_index_ids": sorted(duplicates(index_ids)),
    "is_exact_match": index_ids가 비어 있지 않고 양방향 차이와 중복이 없음,
}
```

- [ ] **Step 4: 전체 단위 테스트 실행**

Run: `python -m unittest discover -s tests -v`

Expected: 기존 6개와 신규 2개 테스트 모두 PASS

- [ ] **Step 5: 커밋**

```bash
git add scripts/build_dataset.py tests/test_build_dataset.py
git commit -m "feat: audit requirement index coverage"
```

### Task 2: 문서별 감사 보고서와 검수 큐 생성

**Files:**
- Modify: `scripts/build_dataset.py`
- Create: `reports/extraction_audit_v0.2.0.json`
- Create: `reports/extraction_audit_v0.2.0.md`
- Create: `data/review/extraction_review_v0.2.0.csv`
- Test: `tests/test_build_dataset.py`

**Interfaces:**
- Consumes: Task 1의 `compare_index_and_detail_ids`
- Produces: `build_review_queue(records: list[dict], comparisons: dict[str, dict]) -> list[dict]`

- [ ] **Step 1: 검수 큐 우선순위 테스트 작성**

```python
def test_review_queue_includes_mismatches_and_document_samples(self):
    records = [
        {"document_id": "doc_a", "requirement_uid": "doc_a:SFR-001", "requirement_id": "SFR-001"},
        {"document_id": "doc_a", "requirement_uid": "doc_a:SFR-002", "requirement_id": "SFR-002"},
    ]
    comparisons = {
        "doc_a": {"index_only_ids": ["SFR-003"], "detail_only_ids": ["SFR-002"]}
    }
    queue = build_review_queue(records, comparisons)
    assert any(row["reason"] == "detail_only" and row["requirement_id"] == "SFR-002" for row in queue)
    assert any(row["reason"] == "index_only" and row["requirement_id"] == "SFR-003" for row in queue)
```

- [ ] **Step 2: 실패 확인**

Run: `python -m unittest tests.test_build_dataset -v`

Expected: `build_review_queue` import 실패

- [ ] **Step 3: 검수 큐 생성 구현**

모든 `index_only`, `detail_only`, 중복 ID를 포함한다. 불일치가 없는 문서에서도 문서별 첫 행·중간 행·마지막 행과 본문 최장 행을 넣는다. CSV 열은 다음으로 고정한다.

```text
document_id,requirement_uid,requirement_id,reason,source_file,source_location,
original_checked,id_match,name_match,body_complete,table_structure_ok,review_note,reviewer,reviewed_at
```

- [ ] **Step 4: v0.2 출력으로 버전 상수 변경 후 재생성**

Run: `python scripts/build_dataset.py`

Expected: `requirements_v0.2.0.jsonl`, 감사 JSON/Markdown, 검수 CSV 생성

- [ ] **Step 5: 자동 감사 기준 확인**

Run: `python -m unittest discover -s tests -v`

Expected: 전체 PASS. 감사 보고서에서 문서 10개, 빈 본문 0개, 중복 UID 0개가 유지되고 문서별 목록-상세 차이가 표시됨.

- [ ] **Step 6: 커밋**

```bash
git add scripts/build_dataset.py tests/test_build_dataset.py reports data/review
git commit -m "feat: generate extraction review queue"
```

### Task 3: 원본 대조와 데이터셋 동결

**Files:**
- Modify: `data/review/extraction_review_v0.2.0.csv`
- Create: `reports/extraction_freeze_v0.2.0.md`
- Modify: `docs/RESEARCH_DECISIONS.md`
- Modify: `CONTEXT.md`

**Interfaces:**
- Consumes: Task 2의 검수 큐와 `RFP_data/` 원본
- Produces: 라벨링 가능 여부와 잔여 예외가 명시된 동결 보고서

- [ ] **Step 1: 검수 큐의 모든 행을 원본과 대조**

각 행에 `original_checked=true`, 네 개 판정 열을 `true/false`, 검토자와 ISO 8601 시각을 기록한다. 원본에서 확인할 수 없는 행은 성공으로 간주하지 않고 `review_note`에 원인을 기록한다.

- [ ] **Step 2: 불일치 수정과 재생성**

추출 규칙 오류는 `scripts/build_dataset.py`에 재현 테스트를 먼저 추가한 후 수정한다. Markdown 변환 손실은 원본을 덮어쓰지 않고 예외 목록에 기록한다. 수정 후 다음을 실행한다.

Run: `python -m unittest discover -s tests -v`

Run: `python scripts/build_dataset.py`

Expected: 테스트 전체 PASS, 자동 감사 치명 오류 0개

- [ ] **Step 3: 동결 보고서 작성**

`reports/extraction_freeze_v0.2.0.md`에 데이터셋 해시, 문서별 행 수, 자동 감사 결과, 사람 검수 행 수, 발견·수정한 오류, 해결되지 않은 예외와 `labeling_ready: true|false`를 기록한다.

- [ ] **Step 4: 확정 결정과 인계 문서 갱신**

`labeling_ready: true`일 때만 `docs/RESEARCH_DECISIONS.md`에 v0.2 행 수와 감사 절차를 확정 기록하고, `CONTEXT.md`의 다음 작업을 “라벨링 파일럿 표본 및 JSON 스키마 동결”로 변경한다.

- [ ] **Step 5: 최종 검증**

Run: `python -m unittest discover -s tests -v`

Run: `python scripts/build_dataset.py`

Expected: 재실행 후 출력 해시가 동일하고 테스트 전체 PASS

- [ ] **Step 6: 커밋**

```bash
git add scripts tests data/review reports docs/RESEARCH_DECISIONS.md CONTEXT.md
git commit -m "data: freeze audited requirement dataset v0.2.0"
```
