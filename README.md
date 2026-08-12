# RFP Risk Analysis

한국 공공 AI·IT 구축 RFP의 요구사항을 추출하고, 수행사 관점의 견적·질의·계약 검토 필요성을 분석하는 연구 프로젝트다.

현재는 프로젝트를 처음부터 다시 설계하는 단계다. 과거 실험 기록은 참고자료로만 보존하고, 현재 연구 설계는 [`docs/PROJECT_DIRECTION.md`](docs/PROJECT_DIRECTION.md)에 분리해 정리했다.

## 문서 안내

- 문서별 역할과 우선순위: [`docs/README.md`](docs/README.md)
- 연구 전체 설계도: [`docs/PROJECT_DIRECTION.md`](docs/PROJECT_DIRECTION.md)
- 확정된 데이터셋·방법론 결정: [`docs/RESEARCH_DECISIONS.md`](docs/RESEARCH_DECISIONS.md)
- 현재 작업 인계 메모: [`CONTEXT.md`](CONTEXT.md)
- 데이터 산출물과 재생성 방법: [`data/README.md`](data/README.md)

## 현재 데이터

- `RFP_data/md/`: 분석 대상 RFP 10개의 Markdown 원문
- `RFP_data/`: 출처 확인용 PDF·HWP·HWPX 원본

## 다음 작업

1. 10개 RFP의 출처와 원문 대응 확인
2. 요구사항 단위 결정론적 추출 규칙 설계
3. 추출 결과 무결성 검사와 EDA
4. LLM 라벨링·앵커링 파일럿
5. 전통 ML 베이스라인과 경량 파인튜닝 비교

상세 방향: [`docs/PROJECT_DIRECTION.md`](docs/PROJECT_DIRECTION.md)
