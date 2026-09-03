# 경계 겨냥 변형 — 불릿 max-pooling과 blocker 유형 타깃

- 데이터: label_dataset_v4, 앵커 제외 924건, LODO 10-fold, 학습 8문서
- 기반 모델: word+char TF-IDF balanced Logistic (기준선과 같은 학습)
- 명령: `python -m scripts.evaluation.boundary_features`
- 사전 판정 기준: 기준선 대비 10 fold 중 8 우세

| 변형 | 뜻 |
|---|---|
| baseline | 기준선 그대로 (재현 확인용) |
| maxpool_soft | 줄(15자 이상)별 계약 확률의 최댓값으로 계약 확률을 올린 뒤 argmax |
| maxpool_any | 어느 한 줄이라도 계약으로 예측되면 계약, 아니면 기준선 예측 |
| blocker_types | blocker 5종 다중 라벨 + cost_basis 7종 → 결정 21 규칙 |
| blocker_binary | blocker 유무 + 원가 유무 (2진) → 결정 21 규칙. 세분화 효과의 대조군 |

| 변형 | fold 평균 | 통합 OOF | 통상 | 견적 | 계약 | 오답 | 견적↔계약 혼동 | 우세 fold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.614 | 0.638 | 0.795 | 0.545 | 0.573 | 294 | 98 | 0/10 |
| maxpool_soft | 0.610 | 0.634 | 0.785 | 0.534 | 0.581 | 301 | 99 | 5/10 |
| maxpool_any | 0.561 | 0.593 | 0.768 | 0.428 | 0.584 | 330 | 119 | 2/10 |
| blocker_types | 0.611 | 0.624 | 0.784 | 0.554 | 0.533 | 306 | 99 | 6/10 |
| blocker_binary | 0.587 | 0.620 | 0.777 | 0.539 | 0.545 | 314 | 106 | 2/10 |
