# Claude API 전환 결정 보고서

> 기준일: 2026-08-14  
> 범위: RFP 요구사항 라벨링과 3방법 비교 실험의 Gemini 호출을 Claude API로 전환하기 전 결정 사항

## 결론

단순히 SDK와 모델명만 바꾸면 안 된다. 현재 호출 코드는 프롬프트, 출력 스키마, 재시도, 앵커 풀이 여러 파일에 중복되어 있고 일부 실험 조건도 서로 다르다.

권장안은 다음과 같다.

1. Anthropic 직접 API와 공식 Python SDK를 사용한다.
2. 과거 Gemini 스크립트와 결과는 재현용으로 동결한다.
3. 새 Claude 실행기는 공급자 중립 모듈로 하나만 만든다.
4. 40건 파일럿에서 `claude-sonnet-5`와 `claude-haiku-4-5-20251001`을 비교한다.
5. 주 실험 모델은 사람 검수 결과로 하나를 고정한다. 기본 추천은 Sonnet 5다.
6. JSON 출력은 Pydantic 단일 스키마와 `client.messages.parse()`를 사용한다.
7. 40건 파일럿은 동기 API, 259건 × 3방법 본 실험은 데이터 보존 조건이 허용되면 Message Batches API를 사용한다.
8. 라벨 없는 원본 데이터가 아니라 사람 검수로 동결한 앵커 풀만 few-shot에 사용한다.
9. 새 결과는 Gemini 결과와 섞지 않고 새 데이터셋·실험 버전으로 저장한다.

## 지금 사용자가 결정할 것

아래 7개만 답하면 구현 방향이 고정된다.

### D1. Claude 접속 경로

- 선택 A: Anthropic 직접 API
- 선택 B: Amazon Bedrock 또는 Google Cloud 경유
- **추천: A**

직접 API가 코드와 운영 요소가 가장 적다. 공식 Python SDK는 `pip install anthropic`, `ANTHROPIC_API_KEY`, `Anthropic()` 조합을 사용한다. Python 3.9 이상이 필요하다. [공식 Python SDK](https://platform.claude.com/docs/en/cli-sdks-libraries/sdks/python)

### D2. 전환 범위

- 선택 A: 현재 주 실행기만 Claude로 만들고 과거 Gemini 스크립트는 동결
- 선택 B: 기존 Gemini 스크립트 4개를 모두 Claude로 직접 변환
- **추천: A**

과거 실험 스크립트까지 덮어쓰면 기존 결과의 재현 조건을 잃는다. 새 공급자 중립 실행기를 만들고 기존 파일은 `legacy` 성격으로 보존하는 편이 안전하다.

### D3. 주 모델 선택 방법

- 선택 A: Sonnet 5로 바로 고정
- 선택 B: Haiku 4.5로 바로 고정
- 선택 C: 40건을 두 모델로 실행한 뒤 사람 검수로 선택
- **추천: C. 선택 결과가 비슷하면 Haiku, 고위험 라벨 재현율 차이가 크면 Sonnet**

공식 모델 ID는 `claude-sonnet-5`와 `claude-haiku-4-5-20251001`이다. Sonnet 5는 1M 컨텍스트와 최대 128k 출력, Haiku 4.5는 200k 컨텍스트와 최대 64k 출력을 제공한다. 이 프로젝트의 한 요구사항 단위 입력에는 둘 다 충분하다. [공식 모델 비교](https://platform.claude.com/docs/en/about-claude/models/overview)

선택 규칙 추천:

- 40건 고정 검수셋 사용
- `계약·질의검토` 오분류가 Sonnet보다 1건 이하로 많고 스키마 성공률이 같으면 Haiku 선택
- 그 외에는 Sonnet 선택
- 선택 후 모든 zero/pure/stratified 조건에 같은 모델을 사용

### D4. 본 실험 실행 방식과 데이터 보존

- 선택 A: 동기 Messages API만 사용
- 선택 B: 파일럿은 동기, 본 실험은 Message Batches API
- **추천: 공개 RFP만 처리한다면 B**

Batch는 비용을 50% 줄이고 대량 비동기 처리에 맞지만, 결과 순서는 보장되지 않으므로 `custom_id=requirement_uid:method`로 결합해야 한다. 대부분 1시간 안에 끝나며 최대 24시간 처리 창을 가진다. [공식 Batch 문서](https://platform.claude.com/docs/en/build-with-claude/batch-processing)

중요한 개인정보나 비공개 제안 자료를 넣을 계획이라면 A를 선택해야 한다. Batch는 ZDR 대상이 아니며 작업 데이터가 29일 보존된다. 동기 Messages API는 계약에 따라 ZDR 적용이 가능하다. 일반 API의 표준 입력·출력 보존 기간은 최대 30일이다. [공식 데이터 보존 문서](https://platform.claude.com/docs/en/manage-claude/api-and-data-retention)

### D5. 출력 스키마

- 선택 A: 현재 파일별 스키마를 유지
- 선택 B: 단일 Pydantic 스키마로 통합
- **추천: B**

현재 코드에는 서로 호환되지 않는 두 스키마가 있다.

| 필드 | 파일럿 스키마 | 3방법 스키마 |
|---|---|---|
| `confidence` | `높음/중간/낮음` | `높음/보통/낮음` |
| `evidence` | 문자열 배열 | 문자열 |
| `missing_information` | 객체 | 문자열 |
| `domain_dependency` | 객체 | 문자열 |

Claude structured outputs는 JSON Schema에 맞춘 출력을 지원하고 Python SDK의 `client.messages.parse(..., output_format=Model)`가 Pydantic 변환과 검증을 처리한다. 다만 `refusal`이나 `max_tokens`이면 스키마가 완성되지 않을 수 있으므로 `stop_reason`을 별도로 검사해야 한다. [공식 structured outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs), [공식 stop reason 문서](https://platform.claude.com/docs/en/build-with-claude/handling-stop-reasons)

### D6. 앵커 풀 기준

- 선택 A: 기존 zero-shot 출력도 앵커 라벨로 사용
- 선택 B: 사람 검수 완료된 고정 앵커 풀만 사용
- **추천: B**

이 결정은 모델 선택보다 먼저 필요하다. 현재 `run_pilot_all_3methods_paid.py`는 라벨 없는 `target_items`로 검색기를 만든다. `format_fewshot_prompt()`는 라벨이 없으면 `견적반영`을 기본값으로 넣는다. 따라서 pure/stratified 비교에 인위적인 `견적반영` 편향이 들어갈 수 있다.

권장 앵커 풀 조건:

- 사람 검수 완료
- 세 라벨 최소 개수 명시
- 대상 문서와 동일 문서 제외
- `anchor_pool_version`, 검수자, 검수일, 원본 해시 기록
- 평가 fold의 라벨을 앵커로 사용하지 않음

### D7. 비용 한도

다음 두 값을 정해야 한다.

- 파일럿 최대 예산: 권장 `USD 10`
- 본 실험 최대 예산: 권장 `USD 50`

2026-08-14 기준 Sonnet 5는 8월 31일까지 입력 $2/MTok, 출력 $10/MTok의 한시 가격이며 이후 $3/$15다. Haiku 4.5는 $1/$5다. Batch 사용 시 표준 API 가격에서 50% 할인된다. [공식 가격](https://platform.claude.com/docs/en/about-claude/pricing)

실제 비용은 다음 식으로 실행 전에 계산한다.

```text
예상 비용 = 입력 토큰/1M × 입력 단가 + 출력 토큰/1M × 출력 단가
Batch 예상 비용 = 예상 비용 × 0.5
```

Token Counting API는 무료이며 메시지 생성과 별도 rate limit을 사용한다. [공식 token counting](https://platform.claude.com/docs/en/build-with-claude/token-counting)

## 구현 전에 반드시 고칠 현재 코드 문제

| 우선순위 | 문제 | 영향 | 조치 |
|---|---|---|---|
| P0 | 3방법 실행기의 앵커 풀이 라벨 없는 원본 | 실험 편향, stratified 의미 훼손 | 검수 완료 앵커 풀로 교체 |
| P0 | 출력 스키마 2종이 서로 다름 | 결과 비교·검증 불가 | 단일 Pydantic 모델 사용 |
| P1 | `run_labeling_pilot.py`가 429 처리에서 정의되지 않은 `err_msg` 사용 | 429 발생 시 재시도 코드가 다시 실패 | `err_str`로 수정 |
| P1 | API 키 검사와 client 생성이 import 시 실행 | 단위 테스트와 재사용 어려움 | `build_client()`와 `main()` 안으로 이동 |
| P1 | 4개 파일에 호출·재시도·프롬프트가 중복 | 수정 누락, 실험 조건 불일치 | 단일 Claude client adapter 생성 |
| P1 | 최대 30회 고정 sleep 재시도 | 긴 대기, 비용·상태 추적 어려움 | SDK 기본 2회 + 제한된 작업 재시도 |
| P1 | 모델명·보고서 제목·lineage가 실제 설정과 다름 | 재현성 훼손 | 실제 response model과 run manifest 기록 |
| P2 | Anthropic 토큰 추적이 cache write를 집계하지 않음 | 비용 추정 오차 | `cache_creation_input_tokens` 추가 |
| P2 | 과거 실행기가 `reports/archive/`에 다시 씀 | 보관 자료 덮어쓰기 가능 | legacy 실행기 읽기 전용 또는 새 run 경로 사용 |

## 권장 새 구조

```text
scripts/labeling/
├─ label_schema.py          # 단일 Pydantic 출력 모델
├─ claude_client.py         # client 생성, parse, stop_reason, token usage
├─ experiment_runner.py     # sync/batch 실행, resume, manifest
├─ prompts.py               # 고정 system prompt와 방법별 prompt
├─ anchor_retriever.py      # 검수 앵커 검색
└─ legacy/                  # 기존 Gemini 재현 스크립트
```

새 실행 결과:

```text
reports/current/runs/<run_id>/
├─ manifest.json
├─ requests.jsonl
├─ responses.jsonl
├─ failures.jsonl
├─ token_usage.json
└─ comparison.md
```

`manifest.json` 필수 항목:

- provider와 실제 model ID
- Anthropic SDK 버전
- prompt version과 SHA-256
- schema version과 SHA-256
- anchor pool version과 SHA-256
- 입력 데이터셋 버전과 SHA-256
- 실행 방식 `sync|batch`
- 시작·종료 시각
- 요청 성공·실패·refusal·truncation 건수
- 토큰과 비용 단가 스냅샷

## Claude 호출 시 고정할 기술 정책

1. 최신 모델에서는 `temperature`, `top_k`, `top_p`를 보내지 않는다. Claude Opus 4.6보다 나중에 출시된 모델은 비기본 temperature를 지원하지 않는다. [Messages API](https://platform.claude.com/docs/en/api/messages/create)
2. 응답마다 `stop_reason`을 검사한다. `end_turn`만 정상 완료로 보고 `max_tokens`, `refusal`, `model_context_window_exceeded`는 별도 실패 상태로 기록한다.
3. SDK 기본 자동 재시도는 연결 오류, 408, 409, 429, 5xx에 2회 적용된다. 애플리케이션 재시도는 최대 1회만 추가한다. [Python SDK 재시도](https://platform.claude.com/docs/en/cli-sdks-libraries/sdks/python)
4. 시스템 프롬프트와 스키마가 반복되므로 prompt caching을 시험하되, 실제 `cache_read_input_tokens`가 발생할 때만 비용 절감으로 계산한다. 캐시 읽기는 기본 입력 단가의 10%다. [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
5. API response의 `_request_id`, 실제 `response.model`, usage를 모든 결과에 저장한다.
6. 동일 입력 반복 결과가 완전히 결정적이라고 가정하지 않는다. 모델 비교는 동일 검수셋과 반복 표본으로 한다.

## 추천 실행 순서

1. D1~D7 확정
2. 앵커 풀 감사와 단일 스키마 동결
3. Claude adapter 단위 테스트 작성
4. 3건 sync dry-run
5. 40건 Sonnet/Haiku 비교
6. 사람 검수와 모델 확정
7. 비용·보존 조건 확인
8. 259건 × 3방법 본 실험
9. Gemini 결과와 Claude 결과를 별도 버전으로 비교

## 답변 양식

아래만 복사해 선택하면 된다.

```text
D1 접속: 직접 API / Bedrock·Cloud
D2 범위: 새 실행기만 / 기존 4개 전부
D3 모델: 40건 비교 후 선택 / Sonnet 5 / Haiku 4.5
D4 실행: sync만 / pilot sync + full batch
D5 스키마: 단일 Pydantic / 기존 유지
D6 앵커: 사람 검수 풀 / 기존 zero-shot 포함
D7 예산: pilot USD __ / full USD __
비공개·개인정보 자료 사용 예정: 예 / 아니오
```
