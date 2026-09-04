# 견적반영 대 계약·질의검토 전용 2분류

- 데이터: label_dataset_v4, 앵커 제외 경계 451건, LODO 10-fold, 학습 8문서
- 모델: word+char TF-IDF balanced Logistic. 학습·검증·평가 모두 두 클래스 행만 사용
- 명령: `python -m scripts.evaluation.boundary_binary`

| 설정 | 정확도 | macro F1 |
|---|---:|---:|
| 전용 2분류 (통상수용 제외하고 학습) | 0.710 | 0.709 |
| 3분류 모델, 두 클래스 확률만 비교한 argmax | 0.718 | 0.717 |
| 3분류 모델 원래 예측 (통상 예측은 오답) | 0.552 | – |
| 무작위 | 0.500 | – |

## fold별 전용 2분류 정확도

| 평가 문서 | n | 정확도 |
|---|---:|---:|
| ccrs_ai_platform | 62 | 0.710 |
| defense_intelligent_platform | 103 | 0.728 |
| genai_incident_response | 40 | 0.500 |
| incheon_airport_digital_work | 36 | 0.750 |
| kac_ai_work_platform | 44 | 0.773 |
| kangwon_land_genai | 22 | 0.818 |
| kexim_ai_platform | 62 | 0.742 |
| koen_ai_infrastructure | 19 | 0.737 |
| korail_genai_isp_ismp | 10 | 0.700 |
| mfds_drug_ai_review | 53 | 0.660 |

## 읽기

전용 학습과 3분류 안에서 배운 두 클래스 축이 같은 값이다. 견적↔계약 경계는 3분류 구조가
만든 간섭이 아니며, 계층형 분류기는 2단계가 이 값을 그대로 물려받으므로 후보가 되지 않는다.
3분류 원래 예측이 낮은 것은 경계 451건 중 104건을 `통상수용`으로
예측했기 때문이며, 통상↔검토 축의 오답이 견적↔계약 축과 같은 규모로 있다는 뜻이다.
