# RFP Risk Analysis

공공 AI·IT 구축 RFP의 요구사항을 추출하고, 제안 견적과 계약 검토가 필요한 조항을 분류하는 연구 프로젝트다.

현재 기준 데이터셋은 10개 RFP에서 추출한 `requirements_v0.2.0`이다. 연구 방법의 확정 사항은 [`docs/RESEARCH_DECISIONS.md`](docs/RESEARCH_DECISIONS.md), 전체 방향은 [`docs/PROJECT_DIRECTION.md`](docs/PROJECT_DIRECTION.md)를 따른다.

## 빠른 시작

```powershell
python -m pip install -r requirements.txt
python -m scripts.data.build_dataset
python -m scripts.data.eda_requirements
python -m pytest tests -q
```

API 키와 사용 가능한 Gemini 모델만 확인하려면 다음 명령을 사용한다. 유료 호출은 `smoke` 명령을 명시했을 때만 수행한다.

```powershell
python -m scripts.utilities.check_gemini models
```

## 디렉터리

| 경로 | 내용 |
|---|---|
| `RFP_data/` | 원본 PDF·HWP·HWPX와 분석용 Markdown |
| `data/` | 생성 데이터셋, 전처리 표본, 사람 검수 자료 |
| `scripts/data/` | 요구사항 추출·전처리·표본·EDA |
| `scripts/labeling/` | 라벨 스키마, 검색, LLM 실험, 토큰 비용 추적 |
| `scripts/utilities/` | API 환경 점검과 유지보수 도구 |
| `notebooks/` | 짧은 실행 예제와 프로젝트 안내 |
| `reports/current/` | 현재 기준 보고서와 실험 결과 |
| `reports/archive/` | 재현성을 위해 보존한 과거 결과 |
| `tests/` | 단위·구조 검증 |

## 노트북

1. `00_project_overview.ipynb`: 프로젝트 구조와 현재 산출물 확인
2. `01_dataset_pipeline.ipynb`: 데이터셋 로드와 가벼운 EDA
3. `02_labeling_experiment.ipynb`: 라벨 검증과 토큰 비용 계산 예제

노트북은 사용법만 보여준다. 실제 로직은 `scripts/` 모듈이 기준이다.

## 산출물 정책

- `data/processed/`는 코드로 재생성하며 Git에서 제외한다.
- 현재 판단 근거는 `reports/current/`에 둔다.
- 이전 실험 결과는 삭제하거나 덮어쓰지 않고 `reports/archive/`에 둔다.
- `.env`, API 키, 캐시, 모델 바이너리는 커밋하지 않는다.

문서 우선순위와 상세 지도는 [`docs/README.md`](docs/README.md), 데이터 재생성 방법은 [`data/README.md`](data/README.md)를 참고한다.
