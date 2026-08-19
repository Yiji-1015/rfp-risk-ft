# RFP Risk Analysis

공공 AI·IT 구축 RFP의 요구사항을 추출하고, 제안 견적과 계약 검토가 필요한 조항을 분류하는 연구 프로젝트다.

현재 기준 데이터셋은 10개 RFP에서 추출한 `requirements_v0.2.0`이다. 연구 의사결정 이력은 [`docs/history/`](docs/history/), 전체 방향은 [`docs/PROJECT_DIRECTION.md`](docs/PROJECT_DIRECTION.md)를 따른다.

## 빠른 시작

```powershell
python -m pip install -r requirements.txt
python -m scripts.data.build_dataset
python -m scripts.data.eda_requirements
python -m pytest tests -q
```

Claude 라벨링은 기본적으로 dry-run이다. 아래 첫 명령은 API 키와 네트워크를 사용하지 않는다.

```powershell
python -m scripts.labeling.run_claude_labeling --limit 3
python -m scripts.labeling.run_claude_labeling --limit 3 --execute
```

기본값은 Sonnet 5, `effort=medium`, `thinking=adaptive`, `max_tokens=16000`, 5분 프롬프트 캐시다. 출력 스키마는 v4.0.0(7필드), 프롬프트는 v5다. 한 시간 캐시는 `--cache-ttl 1h`로 선택한다. 실제 호출은 `--execute`를 명시해야 하며 `.env`의 `ANTHROPIC_API_KEY`를 사용한다.

`--thinking`은 항상 명시해서 전송한다. Sonnet 5는 이 값을 생략하면 adaptive로 켜지고, 사고 토큰이 출력 토큰으로 과금되기 때문이다. 끄려면 `--thinking disabled`를 쓴다. `max_tokens`는 사고와 응답을 합친 상한이며 상한일 뿐 소비량이 아니다.

### 라벨링 전략

`--strategy`로 앵커링 방식을 고른다. 전략만 바꾸고 모델·입력·분할은 고정해야 통제 비교가 된다.

| 전략 | 앵커 인출 |
|---|---|
| `zero-shot` (기본) | 없음 |
| `fewshot-similarity` | 유사도 Top-k |
| `fewshot-stratified` | 라벨별 1개씩 균형 인출 |

```powershell
python -m scripts.labeling.run_claude_labeling --strategy fewshot-stratified --limit 3
```

few-shot 전략은 `data/anchors/anchor_pool_v1.jsonl`의 앵커 풀을 사용한다(20건, 결정 22). 형식은 [`data/anchors/README.md`](data/anchors/README.md)를 따른다. dry-run에서도 풀을 검증하고 실제 주입될 앵커의 라벨 분포를 미리 보여주므로, 유료 실행 전에 앵커 편향을 눈으로 확인할 수 있다.

## 디렉터리

| 경로 | 내용 |
|---|---|
| `RFP_data/` | 원본 PDF·HWP·HWPX와 분석용 Markdown |
| `data/` | 생성 데이터셋, 전처리 표본, 사람 검수 자료 |
| `data/anchors/` | 감사·동결된 few-shot 앵커 풀 |
| `scripts/data/` | 요구사항 추출·전처리·표본·EDA |
| `scripts/labeling/` | 라벨 스키마, 검색, LLM 실험, 토큰 비용 추적 |
| `scripts/utilities/` | API 환경 점검과 유지보수 도구 |
| `notebooks/` | 짧은 실행 예제와 프로젝트 안내 |
| `reports/current/` | 현재 기준 보고서와 실험 결과 |
| `reports/current/claude_runs/` | 라벨링 실행별 manifest와 결과 |
| `reports/archive/` | 재현성을 위해 보존한 과거 결과 |
| `tests/` | 단위·구조 검증 |

## 노트북

1. `00_project_overview.ipynb`: 프로젝트 구조와 현재 산출물 확인
2. `01_dataset_pipeline.ipynb`: 데이터셋 로드와 가벼운 EDA
3. `02_labeling_experiment.ipynb`: 라벨 검증과 토큰 비용 계산 예제
4. `03_requirements_eda.ipynb`: 판단 요소, 문맥 의존성, 조건부 유사 사례와 파일럿 커버리지 분석

## 라벨 스키마

주 라벨은 가격 산정 가능성이다(§5.1).

| 라벨 | 기준 |
|---|---|
| `통상수용` | 추가 원가 없음. 기본 수행팀 공수에 포함 |
| `견적반영` | 원가는 붙지만 원문 정보로 계산 가능 |
| `계약·질의검토` | blocker가 있어 계산 불가 |

보조 축 4개를 함께 산출한다. `blockers`(범위·책임 / 검수·성능기준 / 기술실현성 / 라이선스·공급 / 공급자종속), `cost_basis`, `domain_dependency`, `build_difficulty`. `blockers`가 비어 있고 `cost_basis`가 `없음`이면 `통상수용`이 되는 고정 규칙을 `derive_primary_action()`이 구현한다(결정 21).

상세 이력은 [`docs/history/decisions-02.md`](docs/history/decisions-02.md) 결정 16~22, 실험 결과는 [`reports/current/labeling_experiment_v0.1.0.md`](reports/current/labeling_experiment_v0.1.0.md)를 본다.

노트북은 사용법만 보여준다. 실제 로직은 `scripts/` 모듈이 기준이다.

## 산출물 정책

- `data/processed/`는 코드로 재생성하며 Git에서 제외한다.
- 현재 판단 근거는 `reports/current/`에 둔다.
- 이전 실험 결과는 삭제하거나 덮어쓰지 않고 `reports/archive/`에 둔다.
- `.env`, API 키, 캐시, 모델 바이너리는 커밋하지 않는다.

연구 설계와 확정 결정은 `docs/`의 현재 문서를, 데이터 재생성 방법은 [`data/README.md`](data/README.md)를 참고한다.
