# v5 통상수용 대 검토필요 2분류

- 데이터셋: `label_dataset_v5`, 동결 앵커 100건을 제외한 1345건
- 평가: 학습 8 / 검증 1 / 평가 1 문서 LODO 10-fold, 3분류와 같은 분할
- 라벨: `견적반영`과 `계약·질의검토`를 `검토필요`로 합친 뒤 **처음부터 학습**
- 명령: `$env:RFP_DATASET_VERSION='v5'; python -m scripts.evaluation.binary_review`

## fold 단순 평균

| 설정 | macro F1 | 정확도 | 검토 precision | 검토 recall | 검토 F1 |
|---|---:|---:|---:|---:|---:|
| Dummy(최빈) | 0.284 | 0.407 | 0.060 | 0.231 | 0.094 |
| word 1-2gram + balanced | 0.738 | 0.772 | 0.707 | 0.734 | 0.715 |
| char 3-4gram + balanced | 0.755 | 0.784 | 0.754 | 0.734 | 0.738 |
| LinearSVC + balanced | 0.759 | 0.785 | 0.762 | 0.737 | 0.745 |
| word 1-2 + char 3-4gram + balanced | 0.764 | 0.793 | 0.744 | 0.759 | 0.748 |

## 통합 OOF (fold를 나누지 않고 전체를 한 번에)

| 설정 | macro F1 | 정확도 | 검토 precision | 검토 recall | 검토 F1 |
|---|---:|---:|---:|---:|---:|
| Dummy(최빈) | 0.365 | 0.422 | 0.266 | 0.129 | 0.174 |
| word 1-2gram + balanced | 0.774 | 0.775 | 0.753 | 0.778 | 0.765 |
| char 3-4gram + balanced | 0.787 | 0.788 | 0.783 | 0.762 | 0.772 |
| LinearSVC + balanced | 0.788 | 0.789 | 0.788 | 0.756 | 0.771 |
| word 1-2 + char 3-4gram + balanced | 0.794 | 0.795 | 0.781 | 0.784 | 0.783 |
