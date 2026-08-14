# 프로젝트 문서 안내

이 디렉터리의 문서는 목적에 따라 분리한다. 같은 내용이 여러 파일에 보이면 아래 우선순위를 따른다.

## 문서 지도

- [`CLAUDE_LABELING_USAGE.md`](CLAUDE_LABELING_USAGE.md): Claude 호출·캐시·dry-run 사용법

| 문서 | 용도 | 읽는 시점 | 수정하는 시점 |
|---|---|---|---|
| [`PROJECT_DIRECTION.md`](PROJECT_DIRECTION.md) | 연구 전체 설계도 | 연구 목적, 가설, 모델과 평가 계획을 파악할 때 | 연구 범위나 전체 실험 계획이 바뀔 때 |
| [`RESEARCH_DECISIONS.md`](RESEARCH_DECISIONS.md) | 확정된 방법론 결정 원본 | 데이터셋·전처리·입력·라벨링 규칙을 구현하거나 논문에 쓸 때 | 새로운 결정을 확정하거나 기존 결정을 변경할 때 |
| [`CLAUDE_API_MIGRATION_DECISIONS.md`](CLAUDE_API_MIGRATION_DECISIONS.md) | Claude API 전환 결정 보고서 | LLM 공급자·모델·비용·실행 방식을 정할 때 | Claude 전환 결정을 확정하거나 공식 API 조건이 바뀔 때 |
| [`../CONTEXT.md`](../CONTEXT.md) | 작업 재개용 인계 메모 | 새 작업 세션을 시작할 때 | 작업 종료 전 현재 상태와 다음 할 일을 넘길 때 |
| [`../data/README.md`](../data/README.md) | 데이터 산출물 사용 안내 | 생성 파일과 재생성 명령을 확인할 때 | 데이터 파일·버전·명령이 추가되거나 바뀔 때 |
| [`../reports/current/README.md`](../reports/current/README.md) | 현재 보고서 목록 | 현재 데이터·실험 결과를 확인할 때 | 현재 기준 산출물이 바뀔 때 |
| [`../reports/archive/README.md`](../reports/archive/README.md) | 과거 실험 보관 정책 | 이전 결과를 재현하거나 비교할 때 | 과거 산출물을 추가 보관할 때 |
| [`../README.md`](../README.md) | 프로젝트 입구 | 저장소를 처음 볼 때 | 프로젝트 소개나 핵심 실행 순서가 바뀔 때 |

## 충돌 시 우선순위

1. 확정된 방법론은 `RESEARCH_DECISIONS.md`가 기준이다.
2. 전체 연구 범위와 장기 계획은 `PROJECT_DIRECTION.md`가 기준이다.
3. 실제 데이터 파일의 생성 방법은 `data/README.md`와 실행 코드가 기준이다.
4. `CONTEXT.md`는 현재 작업 상태를 전달하는 메모다. 연구 설계의 기준 문서가 아니다.

예를 들어 `PROJECT_DIRECTION.md`의 스키마가 아직 “초안”이고 `RESEARCH_DECISIONS.md`에 7열 스키마가 확정돼 있다면 7열 스키마를 따른다.

## 문서 작성 규칙

- 아이디어나 검토 중인 내용은 `PROJECT_DIRECTION.md`의 작업 가설 또는 열려 있는 결정에 둔다.
- 사용자와 합의해 확정한 내용은 근거와 함께 `RESEARCH_DECISIONS.md`에 둔다.
- 현재 실험 결과는 `reports/current/`, 과거 결과는 `reports/archive/`에 둔다.
- 같은 설명을 여러 문서에 길게 복사하지 않는다. 기준 문서로 링크한다.
- 숫자 결과에는 데이터 버전과 실행 조건을 함께 기록한다.
