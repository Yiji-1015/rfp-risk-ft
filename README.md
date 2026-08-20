# RFP Risk Analysis

공공 AI·IT 구축 RFP의 요구사항을 추출하고, **제안 견적과 계약 검토가 필요한 조항**을
분류하는 연구 프로젝트다.

10개 RFP에서 요구사항 1,024건을 추출해 Claude로 전수 라벨링했고, 그 결과를 동결해
`label_dataset_v2`로 확정했다. 다음 단계는 이 라벨로 전통 ML과 경량 인코더를 학습해
**비싼 모델이 정말 필요한지** 비교하는 것이다.

| 문서 | 역할 |
|---|---|
| [`docs/PROJECT_DIRECTION.md`](docs/PROJECT_DIRECTION.md) | 연구 설계와 범위 |
| [`docs/PIPELINE.md`](docs/PIPELINE.md) | 원본 RFP → 라벨 데이터셋 실행 절차 |
| [`docs/WORKLOG.md`](docs/WORKLOG.md) | 날짜별 진행 상황과 현재 위치 |
| [`docs/history/`](docs/history/) | 결정과 그 근거 |
| [`docs/issues/`](docs/issues/) | 알려진 데이터 문제 |

## 빠른 시작

```powershell
python -m pip install -r requirements.txt
python -m scripts.data.build_dataset
python -m pytest tests -q
```

```powershell
python -m scripts.labeling.label_dataset
```

마지막 명령이 확정 데이터셋의 상태를 출력한다. API 키도 네트워크도 필요 없다.

## 라벨 데이터셋

분석과 실험은 모두 `data/labels/label_dataset_v2.jsonl`에서 출발한다.

| 항목 | 값 |
|---|---|
| 건수 | 1,024 (10개 RFP 문서) |
| 주 라벨 | 통상수용 512 (50.0%) / 계약·질의검토 270 (26.4%) / 견적반영 242 (23.6%) |
| 스키마 | v4.0.0 (7필드) |
| 생성 | Claude Sonnet 5, 층화 few-shot, 프롬프트 v5 |

`agency` 필드는 문서 1건에서 비어 있다. 원본 RFP에 해당 항목이 없어서이며
데이터 결함이 아니다([issues/002](docs/issues/002-missing-source-fields.md)).

### 동결과 감사

이 파일은 **동결 데이터셋**이다. 읽기는 항상 로더를 거친다.

```python
from scripts.labeling.label_dataset import load_label_dataset

rows, meta = load_label_dataset()
```

로더가 매번 SHA-256을 대조하고 값 도메인·균일 스키마·`requirement_uid` 유일성을
검사한다. 파일이 조용히 바뀌면 분석이 도는 대신 **실패한다.** 실험 결과를 비교하려면
모두가 같은 라벨을 봐야 하는데, 조용한 변경은 눈치채기 어렵기 때문이다.

바꿔야 하면 파일을 고치는 게 아니라 **새 버전을 만들고** `FROZEN_SHA256`을 갱신하며,
그 판단은 `docs/history/`에 남긴다(결정 31).

### 실행 결과에서 다시 만들기

```powershell
python -m scripts.labeling.build_label_dataset
```

`reports/current/claude_runs/`의 실행 디렉터리 세 곳에서 직접 읽어 재생성한다.
빌더가 결정적이라 같은 입력에서 **바이트까지 동일한 파일**이 나온다.

정리하는 것은 셋이다.

- **고정 규칙 위반 보정** — 모델이 보조 축과 어긋나는 주 라벨을 낸 6건(0.6%)을 규칙에
  맞춘다. 모델 원본은 `primary_action_model`에, 보정 여부는 `rule_corrected`에 남는다.
- **균일 스키마와 명시적 출처** — 모든 행이 같은 키를 갖고 `execution_path`,
  `source_run`을 직접 들고 있다.
- **요구사항 유형 정규화** — 원본 표기 60종을 공공 SW 표준 11분류 + `컨설팅`으로 묶어
  `requirement_type_normalized`(12종)를 만든다. 원본 표기와 판단 근거
  (`requirement_type_source`)도 함께 남아 매핑을 재검증할 수 있다(결정 32).

## 라벨 스키마

주 라벨은 **가격을 산정할 수 있는가**다(§5.1).

| 라벨 | 기준 |
|---|---|
| `통상수용` | 추가 원가 없음. 기본 수행팀 공수에 포함 |
| `견적반영` | 원가는 붙지만 원문 정보로 계산 가능 |
| `계약·질의검토` | blocker가 있어 계산 불가 |

보조 축 4개를 함께 산출한다.

| 축 | 값 |
|---|---|
| `blockers` | 범위·책임 / 검수·성능기준 / 기술실현성 / 라이선스·공급 / 공급자종속 (복수) |
| `cost_basis` | 없음 / 고급·전문인력 / 장비·인프라 / 라이선스 / 외부인증 / 외주·전문기관 / 복합 |
| `domain_dependency` | 높음 / 보통 / 낮음 |
| `build_difficulty` | 높음 / 보통 / 낮음 |

`blockers`가 있으면 `계약·질의검토`, 없고 `cost_basis`가 `없음`이 아니면 `견적반영`,
둘 다 아니면 `통상수용`이다. 이 고정 규칙을 `derive_primary_action()`이 구현한다(결정 21).

상세는 [`docs/history/decisions-02.md`](docs/history/decisions-02.md) 결정 16~22를 본다.

## 라벨링 재현

라벨을 다시 생성해야 할 때만 필요하다. Claude 라벨링은 기본이 dry-run이라
`--execute` 없이는 API 키도 네트워크도 쓰지 않는다.

```powershell
python -m scripts.labeling.run_claude_labeling --limit 3
python -m scripts.labeling.run_claude_labeling --limit 3 --execute
```

기본값은 Sonnet 5, `effort=medium`, `thinking=adaptive`, `max_tokens=16000`,
5분 프롬프트 캐시, 프롬프트 v5다. 한 시간 캐시는 `--cache-ttl 1h`로 고른다.
실제 호출은 `.env`의 `ANTHROPIC_API_KEY`를 쓴다.

`--thinking`은 항상 명시해서 전송한다. Sonnet 5는 생략하면 adaptive로 켜지고 사고
토큰이 출력 토큰으로 과금되기 때문이다. `max_tokens`는 사고와 응답을 합친 상한이며
상한일 뿐 소비량이 아니다.

### 앵커링 전략

`--strategy`로 few-shot 예시를 어떻게 고를지 정한다. 전략만 바꾸고 모델·입력·분할은
고정해야 통제 비교가 된다.

| 전략 | 앵커 인출 |
|---|---|
| `zero-shot` (기본) | 없음 |
| `fewshot-similarity` | 유사도 Top-k |
| `fewshot-stratified` | 라벨별 1개씩 균형 인출 — **전수 라벨링에 사용** |
| `fewshot-global` | 모든 입력에 같은 고정 앵커 3건 |

```powershell
python -m scripts.labeling.run_claude_labeling `
  --strategy fewshot-stratified --anchor-pool data/anchors/anchor_pool_v2.jsonl --limit 3
```

앵커 풀은 `data/anchors/anchor_pool_v2.jsonl`(100건, 10개 문서)로 동결돼 있다.
전수 라벨링 전에 확정하며 실행 중 변경하지 않는다. dry-run에서도 풀을 검증하고
주입될 앵커의 라벨 분포를 미리 보여주므로 유료 실행 전에 편향을 눈으로 확인할 수 있다.

`fewshot-global`만 앵커 블록이 캐시되는 system 블록에 실린다. 앵커가 입력과 무관하게
고정되어야 캐시 프리픽스가 유지되기 때문이다. 동적 인출은 건마다 앵커가 달라져
system에 올리면 오히려 손해다(결정 29).

## 노트북

| # | 노트북 | 내용 |
|---|---|---|
| 00 | `00_project_overview.ipynb` | 프로젝트 구조와 현재 산출물 |
| 01 | `01_dataset_pipeline.ipynb` | 데이터셋 로드와 무결성 확인 |
| 02 | `02_labeling_experiment.ipynb` | 라벨 검증과 토큰 비용 계산 |
| 03 | `03_requirements_eda.ipynb` | 요구사항 자체의 EDA (라벨 이전) |
| 04 | `04_anchor_pool_analysis.ipynb` | 앵커 풀 구성과 인출 시뮬레이션 |
| 05 | `05_run_comparison.ipynb` | 실행 경로와 앵커링 전략 비교 |
| 06 | `06_label_eda.ipynb` | 라벨 분포, 규칙 감사, fold 난이도, 문구 반복 |

비교·분석은 스크립트가 아니라 노트북으로 만든다. 노트북은 사용법만 보여주고
실제 로직은 `scripts/` 모듈이 기준이다.

## 알려진 데이터 문제

[`docs/issues/`](docs/issues/)에 8건을 상태·관측·영향·대응 형식으로 정리했다.
학습 입력을 요구사항 원문으로 한정한다는 전제 아래(결정 33), 실제로 학습에 영향을
주는 것은 둘이다.

| # | 문제 | 상태 |
|---|---|---|
| [004](docs/issues/004-execution-path-confounding.md) | 라벨 100건이 다른 실행 경로로 생성됐고 문서와 교란돼 효과 측정 불가 | 수용 |
| [005](docs/issues/005-rare-label-values.md) | `cost_basis`의 `외부인증`이 3건 등 희소 값이 학습·평가 불가 | 미해결 |

나머지는 평가 보고 방식(003·006), 분석용(001·002), 입력 제외(007), 라벨 생성 쪽(008)이다.

## 다음 단계

라벨이 확정됐으므로 ML 비교로 넘어간다(§9). 각 단계는 **복잡하게 만든 값을 했는가**에
답하기 위해 존재한다.

```
1. DummyClassifier              최저 기준 (최빈 클래스만 찍기)
2. TF-IDF + Logistic / SVM      주 기준선 — 문자 n-gram 필수
3. 사전학습 문장 임베딩 + ML      의미 표현이 단어 겹침보다 나은가
4. 경량 한국어 인코더 파인튜닝    GPU 예산 대비 추가 이득이 있는가
```

먼저 정해야 할 것이 둘 있다.

- **fold 분할** — 문서 단위로 나눈다(§10.1). 문서별 라벨 분포가 2.8배 차이 나므로
  fold별 다수 클래스 기준선을 반드시 병기한다(issues/003).
- **파인튜닝 타깃** — 주 라벨 하나만 할지, 다중 헤드로 보조 축까지 예측할지.
  §12가 "처음부터 다중 과제 파인튜닝"을 제외 범위로 두고 있어 §12 수정 여부가 선결이다.

## 디렉터리

| 경로 | 내용 |
|---|---|
| `RFP_data/` | 원본 PDF·HWP·HWPX와 분석용 Markdown |
| `data/labels/` | **확정(동결) 라벨 데이터셋** — 분석은 여기서 출발 |
| `data/anchors/` | 감사·동결된 few-shot 앵커 풀 |
| `data/processed/` | 재생성 가능한 요구사항 데이터셋 (Git 제외) |
| `data/samples/` | 파일럿·표준조항·앵커후보 표본 |
| `scripts/data/` | 요구사항 추출·전처리·표본·EDA |
| `scripts/labeling/` | 라벨 스키마, 앵커 검색, LLM 실행, 데이터셋 빌더 |
| `scripts/utilities/` | API 환경 점검과 유지보수 도구 |
| `notebooks/` | 분석과 비교 |
| `reports/current/claude_runs/` | 라벨링 실행별 manifest와 원본 결과 |
| `reports/archive/` | 재현성을 위해 보존한 과거 결과 |
| `tests/` | 단위·구조 검증 |

## 산출물 정책

- `data/processed/`는 코드로 재생성하며 Git에서 제외한다.
- 유료 실행 결과(`reports/current/claude_runs/`)와 동결 산출물(`data/labels/`,
  `data/anchors/`)은 커밋한다. 다시 만들려면 돈이 들거나 결과가 달라지기 때문이다.
- 이전 실험 결과는 삭제하거나 덮어쓰지 않고 `reports/archive/`에 둔다.
- `.env`, API 키, 캐시, 모델 바이너리는 커밋하지 않는다.
