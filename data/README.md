# 데이터 디렉터리

> **문서 역할:** 생성된 데이터 파일의 의미, 버전과 재생성 명령을 안내한다. 연구 설계의 근거와 확정 규칙은 [`../docs/RESEARCH_DECISIONS.md`](../docs/RESEARCH_DECISIONS.md)에 기록한다.

## `processed/requirements_v0.1.0.*`

`RFP_data/md/`의 HTML 상세 요구사항 표를 결정론적으로 추출한 비라벨 데이터다.

- 한 행은 요구사항 ID 하나다.
- 세부 불릿은 분리하지 않는다.
- `raw_requirement_text`는 표 셀의 줄바꿈을 보존한다.
- `normalized_requirement_text`는 현재 원문과 같다. 정규화 규칙은 추출 감사 후 별도 버전에서 추가한다.
- `source_file`, `source_location`, `source_sha256`로 입력 원문을 추적한다.
- 라벨링 전 `reports/extraction_audit_v0.1.0.md`의 누락·중복과 목록-상세표 대응을 확인해야 한다.

재생성:

```powershell
python scripts/build_dataset.py
```

엄격 검증:

```powershell
python scripts/build_dataset.py --strict
```

## `samples/preprocessing_sample_v0.1.0.csv`

10개 문서에서 불릿이 있는 요구사항을 우선해 문서당 2건을 고른 전처리 비교 표본이다.

- `raw_text`: 문자와 목록 경계를 보존한 최소 정리본
- `normalized_list_text`: 줄 시작 불릿을 `-`로 통일한 구조 보존본
- `flat_text`: 불릿과 줄 경계를 제거한 평탄화본

재생성:

```powershell
python scripts/preprocess_text.py --per-document 2
```

전체 요구사항 중 본문이 가장 긴 1건 생성:

```powershell
python scripts/preprocess_text.py --longest
```

원본 파일 용량이 가장 큰 RFP의 전체 요구사항 생성:

```powershell
python scripts/preprocess_text.py --largest-document
```
