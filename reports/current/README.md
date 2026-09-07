# Current reports

현재 기준으로 사용하는 분석·검수 산출물이다.

- `eda_v0.2.0.*`: EDA 결과
- `extraction_audit_v0.2.0.*`: 추출 품질 감사 결과
- `extraction_freeze_v0.2.0.md`: 추출 범위 동결 기록
- `extraction_readiness_v0.2.0.md`: 추출 준비 상태
- `labeling_experiment_v0.1.0.md`: LLM 라벨링 실험 결과 (스키마 v2~v4, 프롬프트 v2~v5, zero-shot/few-shot 비교)
- `extraction_audit_v0.3.0.*`, `_v0.4.0.*`, `eda_v0.3.0.*`: 이후 버전의 감사·EDA
- `claude_runs/`: 동기 실행별 manifest와 원본 결과. 실행 조건이 다르면 디렉터리를 분리한다
- `claude_batches/`: 배치 실행. `batch_info.json`에 batch_id·프롬프트 sha256·모델이 남는다
- `solar_runs/`: Upstage Solar 비교 실행 (기각, 454건)
- `v4/`, `v5/`: 라벨 데이터셋 버전별 모델 비교 산출물. 파인튜닝·앙상블은 아직 `v4/`에만 있다

새 버전을 기준으로 채택하면 이전 버전은 `../archive/`로 이동한다.
