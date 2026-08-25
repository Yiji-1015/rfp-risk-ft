# 요구사항 추출 감사 v0.3.0

- 추출 행: 1,024
- 포함 문서: 10/10
- 요구사항명 누락: 0
- 본문 누락: 0
- 중복 UID: 0
- 목록-상세표 완전 일치 문서: 7
- 목록-상세표 불일치 문서: 2
- 승인 정책으로 해소된 문서: 2
- 미해결 목록-상세 불일치 문서: 0
- ID 목록이 없는 문서: 1

## 문서별 행 수

| document_id | rows |
|---|---:|
| ccrs_ai_platform | 95 |
| defense_intelligent_platform | 169 |
| genai_incident_response | 67 |
| incheon_airport_digital_work | 78 |
| kac_ai_work_platform | 86 |
| kangwon_land_genai | 50 |
| kexim_ai_platform | 137 |
| koen_ai_infrastructure | 101 |
| korail_genai_isp_ismp | 49 |
| mfds_drug_ai_review | 192 |

## 목록-상세표 ID 대조

| document_id | 목록 | 상세 | 목록만 | 상세만 | 목록 중복 | 상태 |
|---|---:|---:|---:|---:|---:|---|
| ccrs_ai_platform | 95 | 95 | 0 | 0 | 0 | 일치 |
| kangwon_land_genai | 50 | 50 | 0 | 0 | 0 | 일치 |
| defense_intelligent_platform | 170 | 169 | 0 | 0 | 1 | 승인 예외 |
| koen_ai_infrastructure | 0 | 101 | 0 | 0 | 0 | 목록 ID 없음 |
| kexim_ai_platform | 137 | 137 | 0 | 0 | 0 | 일치 |
| genai_incident_response | 67 | 67 | 0 | 0 | 0 | 일치 |
| mfds_drug_ai_review | 192 | 192 | 0 | 0 | 0 | 일치 |
| incheon_airport_digital_work | 78 | 78 | 0 | 0 | 0 | 일치 |
| kac_ai_work_platform | 87 | 86 | 1 | 0 | 0 | 승인 예외 |
| korail_genai_isp_ismp | 49 | 49 | 0 | 0 | 0 | 일치 |

## 승인된 원문 예외

- `koen_ai_infrastructure`: 개수표 99건, 상세표 101건. 상세 요구사항 표 101건을 유지하고 개수표 차이를 원문 예외로 기록

## 자동 검토 대상

- 빈 문서: 없음
- 본문 누락 UID: 없음
- 중복 UID: 없음
- ID 목록 없는 문서: koen_ai_infrastructure

> 이 결과는 상세 요구사항 표의 구조만 읽은 비라벨 원문 데이터셋이다. 목록-상세표 대조와 원본 PDF/HWP 대조가 끝나기 전에는 학습 데이터로 확정하지 않는다.
