# 앵커 풀

Dynamic few-shot 프롬프트에 주입할 사례 모음이다. `scripts/labeling/anchor_pool.py`가 이 디렉터리의 JSONL을 읽는다.

## 파일

| 파일 | 크기 | 설명 | 근거 |
|---|---:|---|---|
| `anchor_pool_v1.jsonl` | 20건 | 초기 파일럿 기준 풀 (사람 11 + 모델 9) | [결정 22](file:///c:/Users/LLOYDK/Desktop/proposal-automation/rfp-risk-ft/docs/history/decisions-02.md) |
| `anchor_pool_v2.jsonl` | **100건** | 10개 기관 층화 3회 일관성 검증 10% 대표 풀 | [결정 25](file:///c:/Users/LLOYDK/Desktop/proposal-automation/rfp-risk-ft/docs/history/decisions-03.md) |
| `anchor_pool_v3.jsonl` | **192건** | Chunk 1(신용회복위 95건) 누적 통합 최신 앵커 풀 | [결정 26](file:///c:/Users/LLOYDK/Desktop/proposal-automation/rfp-risk-ft/docs/history/decisions-03.md) |

## 버전별 라벨 구성

| 버전 | 통상수용 | 견적반영 | 계약·질의검토 | 총 건수 |
|---|---:|---:|---:|---:|
| **v1 (초기 기준)** | 7건 | 3건 | 10건 | **20건** |
| **v2 (10% 대표 풀)** | 44건 | 25건 | 31건 | **100건** |
| **v3 (청크 누적 풀)** | **71건** | **59건** | **62건** | **192건** |

## 행 스키마

한 줄에 앵커 하나다. 아래 필드는 모두 필수이며 빈 값을 허용하지 않는다.

| 필드 | 내용 |
|---|---|
| `requirement_uid` | 원본 데이터셋의 UID. 풀 안에서 유일해야 한다. |
| `document_id` | 출처 문서. 동일 문서 앵커 차단(결정 10)에 쓰인다. |
| `requirement_name` | 요구사항명 |
| `raw_requirement_text` | 원문 전체. 요약하지 않는다. |
| `primary_action` | `통상수용` / `견적반영` / `계약·질의검토` |
| `reasoning` | 판정 이유. 프롬프트에 그대로 노출된다. |
| `review_status` | `후보` / `검토완료` / `제외` |
| `pool_version` | 예: `anchor_pool_v1`. 한 파일 안에서 하나여야 한다. |

권장 추가 필드: `persona_version`, `source_run`, `reviewed_by`, `reviewed_at`.

## 로딩 규칙

- `review_status`가 `검토완료`인 행만 프롬프트에 들어간다. 감사되지 않은 LLM 출력을 다시 예시로 쓰면 초기 오류가 확산된다(PROJECT_DIRECTION §11.13).
- 한 파일에 여러 `pool_version`이 섞이면 로딩이 실패한다.
- `fewshot-stratified` 전략은 세 라벨 모두에 검토완료 앵커가 있어야 실행된다. 하나라도 비면 층화 인출이 성립하지 않고 결정 14가 막으려던 다수 라벨 편향으로 되돌아가므로, 실행 전에 멈춘다.
- 풀 파일의 SHA-256이 run manifest에 기록된다. 풀이 바뀌면 이전 실행과 구분된다.

## 실행 중 변경 금지

앵커 풀은 실험 전에 동결한다. 실행 중 생성된 새 LLM 출력을 풀에 추가하지 않는다(PROJECT_DIRECTION §8.3). 풀을 바꿔야 하면 새 `pool_version`으로 새 파일을 만들고, 이전 파일은 지우지 않는다.
