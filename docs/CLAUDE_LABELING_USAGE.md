# Claude 라벨링 실행

## 기본 설정

| 항목 | 값 |
|---|---|
| 모델 | `claude-sonnet-5` |
| effort | `medium` |
| 최대 출력 | `4096` 토큰 |
| 프롬프트 캐시 | 시스템 프롬프트 명시적 breakpoint |
| 파일럿 TTL | `5m` |
| 장시간 실행 TTL | `1h` |
| SDK 재시도 | 2회 |
| 타임아웃 | 120초 |

`temperature`, `top_p`, `top_k`, 수동 thinking은 보내지 않는다. Haiku 비교 실행에서는 지원하지 않는 `effort`도 보내지 않는다.

## 준비

```powershell
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

`.env`의 `ANTHROPIC_API_KEY`를 실제 키로 바꾼다. `.env`는 Git에서 제외된다.

## dry-run

```powershell
python -m scripts.labeling.run_claude_labeling --limit 3
```

실행 manifest만 화면에 출력한다. API 키를 읽지 않고 네트워크도 사용하지 않는다.

## 실제 파일럿

먼저 한 건으로 확인한다.

```powershell
python -m scripts.labeling.run_claude_labeling --limit 1 --execute
```

문제가 없으면 40건을 실행한다.

```powershell
python -m scripts.labeling.run_claude_labeling --execute
```

결과는 `reports/current/claude_runs/<UTC 실행 ID>/` 아래에 저장된다. 같은 실행 조건과 `--output-dir`를 다시 지정하면 성공한 UID를 건너뛰고 실패 항목만 재시도한다. 모델, effort, TTL, 입력 범위가 달라지면 결과 혼합을 막기 위해 실행을 거부한다.

## 캐시

짧은 동기 파일럿은 기본 `5m`를 사용한다. 호출 간격이 길거나 실행이 한 시간을 넘길 수 있으면 다음처럼 지정한다.

```powershell
python -m scripts.labeling.run_claude_labeling --cache-ttl 1h --execute
```

각 결과의 `metadata`에서 다음 값을 확인한다.

- `cache_creation_input_tokens`: 새 캐시를 만든 토큰
- `cache_read_input_tokens`: 기존 캐시에서 읽은 토큰

둘 다 계속 0이면 고정 prefix가 모델의 최소 캐시 길이에 못 미치거나 요청 사이에 프롬프트·모델·effort가 달라진 것이다. 캐시 적중을 유지하려면 한 실행 안에서 모델, effort, 시스템 프롬프트, 출력 스키마를 바꾸지 않는다.

프롬프트 캐시와 별개로 structured output의 컴파일된 스키마 문법은 Anthropic이 24시간 캐시한다.

## Haiku 비교

```powershell
python -m scripts.labeling.run_claude_labeling `
  --model claude-haiku-4-5-20251001 `
  --output-dir reports/current/claude_runs/haiku-pilot `
  --execute
```

Haiku 호출에는 `effort`가 전송되지 않는다.
