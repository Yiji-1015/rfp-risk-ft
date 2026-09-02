# v4 통상수용 대 검토필요 2분류

- 데이터셋: `label_dataset_v4`, 동결 앵커 100건을 제외한 924건
- 평가: 학습 8 / 검증 1 / 평가 1 문서 LODO 10-fold, 3분류와 같은 분할
- 라벨: `견적반영`과 `계약·질의검토`를 `검토필요`로 합친 뒤 **처음부터 학습**
- 명령: `$env:RFP_DATASET_VERSION='v4'; python -m scripts.evaluation.binary_review`

## fold 단순 평균

| 설정 | macro F1 | 정확도 | 검토 precision | 검토 recall | 검토 F1 |
|---|---:|---:|---:|---:|---:|
| Dummy(최빈) | 0.261 | 0.361 | 0.127 | 0.400 | 0.189 |
| word 1-2gram + balanced | 0.725 | 0.758 | 0.702 | 0.754 | 0.722 |
| char 3-4gram + balanced | 0.740 | 0.769 | 0.751 | 0.750 | 0.741 |
| LinearSVC + balanced | 0.730 | 0.758 | 0.751 | 0.720 | 0.726 |
| word 1-2 + char 3-4gram + balanced | 0.746 | 0.776 | 0.739 | 0.774 | 0.750 |

## 통합 OOF (fold를 나누지 않고 전체를 한 번에)

| 설정 | macro F1 | 정확도 | 검토 precision | 검토 recall | 검토 F1 |
|---|---:|---:|---:|---:|---:|
| Dummy(최빈) | 0.354 | 0.355 | 0.333 | 0.319 | 0.326 |
| word 1-2gram + balanced | 0.767 | 0.767 | 0.745 | 0.796 | 0.770 |
| char 3-4gram + balanced | 0.781 | 0.781 | 0.776 | 0.776 | 0.776 |
| LinearSVC + balanced | 0.773 | 0.774 | 0.781 | 0.745 | 0.763 |
| word 1-2 + char 3-4gram + balanced | 0.788 | 0.788 | 0.771 | 0.805 | 0.787 |

## 참조 — 3분류 예측을 사후에 접은 값

학습 결과가 아니다. 3분류로 학습한 word+char Logistic의 예측을 같은 규칙으로
접어서 통합 OOF로 잰 값이며, 위 표의 통합 OOF와만 비교한다.

- macro F1 0.788 / 정확도 0.788 / 검토 P 0.790 R 0.769 F1 0.780
