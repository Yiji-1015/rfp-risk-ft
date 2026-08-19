# 라벨 데이터셋 생성 절차

> 기준일: 2026-08-19
> 대상: `requirements_v0.2.0` 1,024건 → 라벨 데이터셋

> **문서 역할:** 원본 RFP에서 라벨 데이터셋까지 **재현 가능한 실행 순서**를 기록한다.
> 연구 설계의 근거는 [`PROJECT_DIRECTION.md`](PROJECT_DIRECTION.md), 각 선택의 이유는
> [`history/decisions-0*.md`](history/)에 있다. 이 문서는 "무엇을 어떤 순서로 실행하는가"만 다룬다.

## 전체 흐름

```
[1] 원본 RFP                RFP_data/*.pdf, *.hwpx
      ↓  (수동 변환)
[2] 분석용 Markdown          RFP_data/md/*.md
      ↓  build_dataset.py
[3] 요구사항 데이터셋         data/processed/requirements_v0.2.0.jsonl   (1,024건)
      ↓  sample_100_anchor_candidates.py + 3회 스크리닝
[4] 앵커 풀                  data/anchors/anchor_pool_v2.jsonl          (100건, 동결)
      ↓  run_claude_batch.py (층화 퓨샷)
[5] 라벨 데이터셋            reports/current/claude_runs/labels_*.jsonl (1,024건)
      ↓  (예정)
[6] ML 비교 실험
```

각 단계는 앞 단계의 산출물만 있으면 독립적으로 재실행할 수 있다.

---

## 1~2단계. 원본 → Markdown

PDF·HWP·HWPX를 분석용 Markdown으로 변환해 `RFP_data/md/`에 둔다. 이 단계는 자동화돼 있지 않다.

변환 시 손실되기 쉬운 것: 표 셀 병합, 각주, 부정문, 제공 주체. 원본과 대조 검사가 필요하다(§11.2).

**알려진 결함**: 원본 요구사항 표의 `산출정보` 열이 현재 데이터셋에 반영되지 않았다.
산출물 목록은 견적의 직접 근거이므로 `v0.3.0` 재생성 시 포함해야 한다.

---

## 3단계. 요구사항 추출

```bash
python -m scripts.data.build_dataset --strict
```

`--strict`는 ID 중복이나 필수 필드 누락이 있으면 실패한다. 문서별 서식 차이를 어댑터로 처리한다.

산출물은 `data/processed/`에 생성되며 **Git에서 제외된다.** 코드로 재생성 가능한 것은 커밋하지 않는다.

검증:

```bash
python -m scripts.data.eda_requirements     # 분포·길이·중복 리포트
python -m pytest tests/test_build_dataset.py -q
```

노트북 [`01_dataset_pipeline.ipynb`](../notebooks/01_dataset_pipeline.ipynb)에서 무결성(중복 UID, 빈 본문, 유형 미지정)을 확인할 수 있다.

---

## 4단계. 앵커 풀 구축

few-shot 프롬프트에 주입할 사례를 만든다. **전수 라벨링 전에 완료하고 동결해야 한다.**

### 4-1. 후보 표집

```bash
python -m scripts.data.sample_100_anchor_candidates
```

10개 문서 × 5개 카테고리 층화로 100건을 뽑는다(`seed=42`). 카테고리는 요구사항 ID 접두어로 분류한다.

| 카테고리 | 접두어 |
|---|---|
| 기능 | SFR, FUN, AIP, AIF, SYS |
| 인프라 | ECR, INF |
| 보안 | SER, SEC |
| 데이터연계 | DAR, DAT, INR, INT, GW |
| 관리품질제약 | 나머지 |

### 4-2. 3회 반복 스크리닝

같은 100건을 zero-shot으로 3회 라벨링한다.

```bash
for i in 1 2 3; do
  python -m scripts.labeling.run_claude_labeling --execute \
    --input data/samples/anchor_pool_100_candidates_v0.1.0.jsonl \
    --output-dir reports/current/claude_runs/anchor_pool_100_rep$i
done

python -m scripts.labeling.analyze_100_screening
```

**통과 조건은 `primary_action` 3회 만장일치 하나다.** 100건 중 81건(81%)이 통과했다.

> **한계**: `blockers`는 검사하지 않는다. 반복 일치율이 40~67%로 낮아, 앵커에 실리는
> `reasoning`이 흔들린 건이 포함될 수 있다. 또한 3회 일치는 안정성이지 정확성이 아니다(§8.3).

### 4-3. 풀 동결

통과분과 사람 확정분을 합쳐 `anchor_pool_v2.jsonl`로 확정한다.

| `provenance` | 건수 | 근거 |
|---|---:|---|
| `사람확정` | 11 | 실무자가 원문 검토 (결정 21) |
| `모델일관_3of3` | 9 | 3회 반복 일치 (결정 22) |
| `실버_3of3` | 80 | 100건 스크리닝 통과 (결정 25) |

**이후 절대 바꾸지 않는다.** 청크 결과를 풀에 되먹이면 실행 순서가 결과를 바꾸고(§11.13),
누적하면 종국에 전체가 앵커가 되어 fold 분리가 불가능해진다(§8.4).

풀 상태 확인:

```bash
# notebooks/04_anchor_pool_analysis.ipynb 실행
# 구성, 인출 시뮬레이션, 유사도 분포를 API 호출 없이 확인
```

---

## 5단계. 전수 라벨링

### 5-1. dry-run으로 조건 확인

```bash
python -m scripts.labeling.run_claude_labeling \
  --strategy fewshot-stratified \
  --anchor-pool data/anchors/anchor_pool_v2.jsonl --limit 3
```

manifest와 앵커 인출 미리보기가 출력된다. **유료 실행 전에 반드시 확인한다**(결정 15).
층화가 성립하면 주입 앵커 라벨 분포가 1:1:1로 나온다.

### 5-2. 배치 제출

```bash
python -m scripts.labeling.run_claude_batch --submit --execute \
  --input data/processed/requirements_v0.2.0.jsonl \
  --anchor-pool data/anchors/anchor_pool_v2.jsonl \
  --start 1 --limit 1024 \
  --output-dir reports/current/claude_runs/batch_full
```

배치는 동기 실행의 **50% 가격**이다. `batch_info.json`에 `batch_id`가 저장되므로
제출 후 컴퓨터를 꺼도 된다. 결과는 서버에 29일 보관된다.

> **24시간 안에 끝나지 않은 요청은 만료된다.** 29일은 완료된 결과의 보관 기간이지 처리 시한이 아니다.

### 5-3. 상태 확인과 수신

```bash
python -m scripts.labeling.run_claude_batch --status   --batch-dir reports/current/claude_runs/batch_full
python -m scripts.labeling.run_claude_batch --download --batch-dir reports/current/claude_runs/batch_full
```

### 5-4. 실패 건 재시도

생성 반복(degeneration)으로 일부가 실패한다. 실측 실패율은 **배치 0.43%, 동기 3.3%**다.

관측된 양상:

- 한국어로 시작해 다른 언어로 이탈
- "지적하신 대로 수정하겠습니다" 자기 대화 루프
- 닫는 괄호 무한 반복으로 JSON 파손
- 영어 메타 독백

전부 재시도하면 통과했다. 실패 건만 모아 다시 제출한다.

```bash
# results.jsonl에서 status != ok인 uid를 추출해 입력 파일을 만든 뒤
python -m scripts.labeling.run_claude_batch --submit --execute \
  --input data/samples/batch_retry.jsonl \
  --anchor-pool data/anchors/anchor_pool_v2.jsonl \
  --start 1 --limit <건수> \
  --output-dir reports/current/claude_runs/batch_retry
```

---

## 실행 조건 고정 사항

모든 라벨링 실행에서 아래를 동일하게 유지한다. 하나라도 다르면 통제 비교가 깨진다(§9.3).

| 항목 | 값 | 근거 |
|---|---|---|
| 모델 | `claude-sonnet-5` 단일 | 결정 13 |
| `effort` | `medium` | |
| `thinking` | `adaptive` (명시 전송) | 결정 19 — 생략하면 켜지므로 기록을 위해 항상 명시 |
| `max_tokens` | 16000 | 사고와 응답이 상한을 공유 |
| 프롬프트 캐시 | 5분 | system 프리픽스 3,440토큰이 캐시됨 |
| 스키마 | v4.0.0 | 결정 20~21 |
| 프롬프트 | v5 | 결정 21 |
| 앵커 풀 | `anchor_pool_v2.jsonl` (동결) | 결정 22 정정 |
| 인출 | `fewshot-stratified` (1:1:1) | 결정 14 |

`run_claude_labeling.py`는 manifest에 이 조건을 기록하고, **조건이 다르면 같은
`output-dir`에 이어쓰기를 거부한다.** 서로 다른 조건의 결과가 한 파일에 섞이는 것을 막는다.

---

## 비용 실측

Sonnet 5 도입가($2/$10 per MTok, 2026-08-31까지) 기준, 환율 1,416원.

| 단계 | 건수 | 방식 | 비용 |
|---|---:|---|---:|
| 앵커 스크리닝 | 100 × 3회 | 동기 | 약 2,300원 |
| 전수 라벨링 | 924 | 배치 | 약 7,500원 |
| 실패 재시도 | 4 | 배치 | 약 40원 |

건당 단가는 동기 약 19원, **배치 약 8원**이다. 프롬프트 캐시가 입력 비용의 90%를 줄인다.

---

## 산출물 위치

| 경로 | 내용 |
|---|---|
| `data/processed/requirements_v0.2.0.jsonl` | 요구사항 데이터셋 (Git 제외, 재생성 가능) |
| `data/anchors/anchor_pool_v2.jsonl` | 동결된 앵커 풀 |
| `data/samples/` | 파일럿·표준조항·앵커후보 표본 |
| `reports/current/claude_runs/*/manifest.json` | 실행 조건 (재현의 기준) |
| `reports/current/claude_runs/*/results.jsonl` | 건별 라벨과 토큰 사용량 |
| `reports/current/claude_runs/labels_*.jsonl` | 통합 라벨 데이터셋 |

---

## 재현 시 주의

1. **`.env`에 `ANTHROPIC_API_KEY`가 필요하다.** 없으면 호출 전에 멈춘다. 단가 환경변수
   (`LLM_INPUT_PRICE_PER_1M_USD`, `USD_KRW_RATE`)를 넣으면 비용이 원화로 집계된다.
2. **기본은 dry-run이다.** `--execute`를 명시해야 실제 호출이 일어난다.
3. **앵커 풀을 바꾸면 라벨이 달라진다.** 풀 파일의 SHA-256이 manifest에 기록되므로
   비교 시 반드시 대조한다.
4. **동기와 배치를 섞지 않는다.** 구조화 출력 경로가 다르다. 한 데이터셋은 한 방식으로 만든다.
