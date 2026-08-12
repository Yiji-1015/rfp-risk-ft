---
status: in-progress
branch: main
timestamp: 2026-08-12T00:00:00+09:00
files_modified:
  - .gitignore
  - README.md
  - RFP_data/
  - docs/PROJECT_DIRECTION.md
---

> **문서 역할:** 다음 작업 세션이 빠르게 이어서 진행하기 위한 임시 인계 메모다.  
> **주의:** 연구 설계나 논문 방법론의 기준 문서가 아니다. 확정 결정은 [`docs/RESEARCH_DECISIONS.md`](docs/RESEARCH_DECISIONS.md), 전체 설계는 [`docs/PROJECT_DIRECTION.md`](docs/PROJECT_DIRECTION.md)를 따른다.

## Working on: RFP 위험 분석 재설계

### Summary

한국 공공 AI·IT 구축 RFP 10개를 요구사항 단위로 새로 추출하고, LLM 라벨링과 앵커링을 거쳐 전통 ML·임베딩 ML·경량 파인튜닝을 비교하는 프로젝트다. 과거 v5/v6 실험물은 모두 삭제했으며 현재 저장소는 원천 RFP, 현재 방향 문서와 안내 문서만 남긴 초기 상태다.

### Decisions Made

- 분석 대상은 `RFP_data/md/`의 공개 RFP 10개다. `RFP_data/`의 PDF·HWP·HWPX는 출처 확인용 원본이다.
- 목표는 데이터 분석·EDA·전처리·다양한 ML·파인튜닝 역량을 포트폴리오로 보여주고, 같은 실험을 KCI 논문 후보로 연결하는 것이다.
- 기준 사용자는 공공 AI·IT 구축사의 제안서 작성자다. 일반 AI·IT는 이해하지만 의약품·국방·금융 등 산업별 전문가는 아니라고 가정한다.
- 입력과 예측의 논리적 단위는 요구사항 하나다. 세부 불릿을 독립 학습 행으로 늘리지 않는다.
- 주 라벨 가설은 `통상수용 / 견적반영 / 계약·질의검토`다. 객관적 손실확률이 아니라 실무 검토 조치다.
- 도메인 의존성은 주 라벨과 분리된 보조 축이다. 발주기관 지원과 수행사 책임을 함께 판단한다.
- LLM 라벨링 자체가 연구 대상이다. zero-shot, global/random few-shot, dynamic few-shot과 위험 요인 분해 방식을 비교한다.
- 앵커링 목적은 LLM 판정의 비일관성을 줄이는 것이다. 감사·버전 관리된 앵커 풀을 실험 전에 동결하고 입력별 사례만 동적으로 선택한다.
- 평가 누수를 막기 위해 문서 단위 GroupKFold 또는 leave-one-document-out을 사용하고, 평가 문서의 라벨을 앵커로 쓰지 않는다.
- 주 ML 기준선은 문자+단어 TF-IDF와 Logistic Regression이며 Linear SVM, 고정 임베딩 ML, 경량 한국어 인코더 파인튜닝을 비교한다.
- 딥러닝 비교 실험은 PyTorch로 구현한다. `Dataset`·`DataLoader`, 학습·검증 루프와 체크포인트 관리가 포트폴리오에 드러나게 한다.
- 실험 설정은 YAML, PyTorch 가중치와 실행 지표·아티팩트는 로컬 MLflow로 관리하는 방안을 우선 검토한다. 대용량 파일 버전 관리가 필요해질 때만 DVC를 추가한다.
- GPU 예산은 약 5천 원이므로 전통 ML을 먼저 하고 파인튜닝은 작게 제한한다.
- 요구사항 추출 감사 v0.2.0을 구현했다. 1,024행, 10개 문서, 누락 본문·중복 UID 0건이며 목록 ID가 있는 9개 문서 중 6개가 완전 일치한다.
- 요구사항 데이터셋 v0.2.0을 1,024행으로 동결했고 `labeling_ready: true`다. 승인된 원문 예외와 파일 해시는 `reports/extraction_freeze_v0.2.0.md`에 기록했다.

### Remaining Work

1. 10개 RFP의 발주기관, 공고연도, 공고번호, 공개 URL, 수집일과 원본-Markdown 대응표를 완성한다.
2. 동결 데이터의 길이·유형·문서별 분포와 중첩표 포함 행을 EDA한다.
3. EDA 결과로 페르소나, 주 라벨 경계, 도메인 보조 필드와 LLM JSON 스키마를 동결한다.
4. 소규모 라벨링 파일럿으로 zero-shot과 여러 앵커링 방식을 반복·순서 변경 조건에서 비교한다.
5. 앵커 풀을 감사·동결한 뒤 전체 silver labeling, 사람 감사, ML 기준선과 PyTorch 파인튜닝을 순차 실행한다.

### Notes

- 프로젝트 Git 저장소는 `main` 브랜치지만 아직 첫 커밋이 없다. 현재 파일은 모두 untracked 상태다.
- 과거 데이터, 코드, 결과, 노트북, 방법론, 로드맵과 캐시는 영구 삭제됐다. 과거 2,030행을 복원하거나 목표 행 수로 삼지 않는다.
- 현재 파일 구조는 `.gitignore`, `README.md`, `CONTEXT.md`, `docs/PROJECT_DIRECTION.md`, `RFP_data/`뿐이다.
- 가장 안전한 다음 작업은 출처 메타데이터 표 작성과 RFP 구조 감사다. 곧바로 LLM 라벨링부터 시작하지 않는다.
- 새 세션 시작 문구: `CONTEXT.md와 docs/PROJECT_DIRECTION.md를 읽고 Remaining Work 2번부터 이어서 진행해줘.`
