# label_dataset_v4 기준선 재평가

- 실행 시각: 2026-08-28 16:00 KST
- 데이터셋: `label_dataset_v4` 1,024건
- 데이터셋 SHA-256: `f8c1eb25e31ea28dc11ed3eb51faaf6cbbe61f128959e857e122c8fea1167b79`
- 모델 입력: `model_text` (`요구사항명 + 줄바꿈 + 불릿 정규화 본문`)
- 평가: 동결 앵커 100건을 제외한 924건, 학습 8 / 검증 1 / 평가 1 문서 LODO 10-fold
- 명령: `$env:RFP_DATASET_VERSION='v4'; python -m scripts.evaluation.baselines`
- Git 기준 커밋: `703a6c2c69895f3a232ec9763d700bd26c2ee967` 이후 작업 트리의 v4 입력 전환 코드

## 결과

모든 값은 fold 단순 평균이다.

| 설정 | macro F1 | 정확도 | 계약 precision | 계약 recall | 계약 F1 |
|---|---:|---:|---:|---:|---:|
| Dummy(최빈) | 0.219 | 0.506 | 0.000 | 0.000 | 0.000 |
| word 1-2gram + balanced | 0.600 | 0.655 | 0.548 | 0.558 | 0.542 |
| char 3-4gram + weight 없음 | 0.507 | 0.627 | 0.740 | 0.348 | 0.451 |
| char 3-4gram + balanced | 0.608 | 0.669 | 0.587 | 0.516 | 0.542 |
| LinearSVC + balanced | 0.584 | 0.662 | 0.613 | 0.479 | 0.524 |
| word 1-2 + char 3-4gram + balanced | **0.614** | **0.676** | 0.584 | **0.568** | **0.564** |
| char 3-4gram + 요구사항 유형 + balanced | 0.552 | 0.620 | 0.546 | 0.448 | 0.487 |
| word 1-2 + char 3-4gram + 요구사항 유형 + balanced | 0.548 | 0.617 | 0.535 | 0.474 | 0.497 |
| char 3-4gram + 글자 수 + balanced | 0.582 | 0.666 | 0.548 | 0.542 | 0.527 |
| char 3-4gram + 숫자 정보 + balanced | 0.580 | 0.647 | 0.548 | 0.511 | 0.522 |
| char 3-4gram + 글자 수 + 숫자 정보 + balanced | 0.584 | 0.665 | 0.548 | 0.507 | 0.518 |
| char + 구조·숫자 + Elastic-net Logistic | 0.478 | 0.563 | 0.423 | 0.436 | 0.424 |
| SVD100 + 구조·숫자 + Logistic | 0.553 | 0.636 | 0.508 | 0.505 | 0.494 |
| SVD100 + 구조·숫자 + XGBoost | 0.553 | 0.618 | 0.509 | 0.494 | 0.487 |
| word 1-2 + char 3-4gram + ComplementNB | 0.493 | 0.602 | 0.693 | 0.343 | 0.427 |
| LinearSVC + 검증 weight | 0.586 | 0.662 | 0.606 | 0.508 | 0.538 |
| char + 유형 검증 weight | 0.600 | 0.660 | 0.577 | 0.505 | 0.528 |
| word+char + 유형 검증 weight | 0.602 | 0.674 | 0.591 | 0.523 | 0.545 |

## v3 주요 기준선과 비교

| 설정 | v3 macro F1 | v4 macro F1 | 차이 |
|---|---:|---:|---:|
| word 1-2gram + balanced | 0.581 | 0.600 | +0.019 |
| char 3-4gram + balanced | 0.601 | 0.608 | +0.007 |
| LinearSVC + balanced | 0.579 | 0.584 | +0.005 |
| word+char + balanced | 0.603 | 0.614 | +0.011 |

v4 기준선에서는 word+char Logistic이 가장 높았다. 다만 v4는 요구사항명 추가와 불릿
정규화를 동시에 적용하므로, 이 결과만으로 어느 변화가 개선을 만들었다고 분리해서 말할
수 없다. 이번 실행은 fold별 평가용 학습이며 모델 가중치는 저장하지 않았다.

## 나머지 구현 모델 재평가

2026-08-28 16:41 KST까지 경량 인코더 파인튜닝을 제외한 구현 모델을 모두 v4로
재평가했다.

| 설정 | macro F1 | 정확도 | 계약 precision | 계약 recall | 계약 F1 |
|---|---:|---:|---:|---:|---:|
| 검증 선택 char Logistic | 0.605 | 0.673 | 0.598 | 0.525 | 0.549 |
| NB-SVM | 0.554 | 0.655 | 0.637 | 0.488 | 0.539 |
| fastText | 0.364 | 0.513 | 0.425 | 0.031 | 0.057 |
| E5-small + balanced Logistic | 0.533 | 0.613 | 0.446 | 0.404 | 0.420 |
| E5-small + balanced LinearSVC | 0.549 | 0.635 | 0.473 | 0.413 | 0.429 |
| char TF-IDF + E5 검증 weight | 0.600 | 0.667 | 0.584 | 0.502 | 0.533 |
| char TF-IDF + E5 검증 weight + 숫자 정보 | 0.606 | 0.670 | 0.561 | 0.501 | 0.523 |
| char TF-IDF + 한국어 의존구문 | 0.565 | 0.645 | 0.539 | 0.477 | 0.496 |
| 세 후보 동일 가중 soft voting | 0.613 | 0.677 | 0.603 | 0.531 | 0.556 |
| 후보 하나라도 계약 검토면 검토(합집합 hard-voting 앙상블) | **0.614** | **0.676** | 0.558 | **0.603** | **0.567** |

단일 모델 최고는 word+char Logistic의 macro F1 0.614, 계약 recall 0.568이다. 세 후보 중
하나라도 계약 검토로 예측하면 검토하는 운영 규칙은 macro F1 0.614를 유지하면서 계약
recall을 0.603으로 높이고 precision은 0.558로 낮춘다. E5와 의존구문은 v4에서도 단순
word+char 또는 char 기준선을 넘지 못했다. fastText는 계약 recall 0.031로 계속 탈락이다.
이 합집합 규칙은 세 후보의 확률을 평균하는 soft voting이 아니라, 세 후보 중 하나라도
`계약·질의검토`이면 최종 검토로 올리는 hard-voting형 앙상블 운영 규칙이다.

재현 산출물:

- `data/processed/v4/classical_search_summary.json`
- `data/processed/v4/classical_search_predictions.csv`
- `data/processed/v4/multilingual-e5-small.npz`
- `data/processed/v4/dependency_features.json`
- `data/processed/v4/dependency_results.json`
- `reports/current/v4/model_candidates.json`
- `reports/current/v4/model_candidate_oof.csv`
