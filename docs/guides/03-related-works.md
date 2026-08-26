# Related Works 1차 수집 가이드

> 기준일: 2026-08-26  
> PDF 로컬 폴더: `C:\Users\LLOYDK\Downloads\rfp-related-works-2026-08-26`  
> PDF는 저작권과 용량 문제로 Git에 넣지 않는다.

## 1. 우선 읽을 논문

| 우선 | 논문 | 연결되는 연구 질문 | 상태 |
|---:|---|---|---|
| 1 | Kaur & Kaur (2024), *The application of AI techniques in requirements classification: a systematic mapping*, DOI `10.1007/s10462-023-10667-1` | 요구사항 분류 연구의 전체 지도. 기존 연구가 작은 구식 데이터와 보이지 않은 데이터 일반화에 약하다는 근거 | OA, 자동 다운로드 차단 |
| 2 | Moon et al. (2022), *Automated detection of contractual risk clauses from construction specifications using BERT*, DOI `10.1016/j.autcon.2022.104465` | 계약 위험 문장 분류라는 가장 가까운 응용 연구. BERT와 SVM 비교 | 구독 접근 확인 필요 |
| 3 | Tan et al. (2024), *Large Language Models for Data Annotation and Synthesis: A Survey*, DOI `10.18653/v1/2024.emnlp-main.54` | LLM 라벨 생성·평가·활용의 전체 틀 | 다운로드 완료 |
| 4 | Liu et al. (2022), *What Makes Good In-Context Examples for GPT-3?*, DOI `10.18653/v1/2022.deelio-1.10` | 유사 사례 기반 동적 앵커 선택의 직접 근거 | 다운로드 완료 |
| 5 | Gupta et al. (2023), *Coverage-based Example Selection for In-Context Learning*, DOI `10.18653/v1/2023.findings-emnlp.930` | 유사도만으로 앵커를 고르면 중복되고 커버리지를 놓칠 수 있다는 근거 | 다운로드 완료 |
| 6 | Xu et al. (2024), *In-Context Example Ordering Guided by Label Distributions*, DOI `10.18653/v1/2024.findings-naacl.167` | 같은 예시도 순서와 라벨 분포에 따라 결과가 흔들리는 문제 | 다운로드 완료 |
| 7 | Zhao et al. (2021), *Calibrate Before Use: Improving Few-Shot Performance of Language Models* | few-shot 분류의 다수 라벨·표현 편향과 보정 | 다운로드 완료 |
| 8 | Chuang et al. (2020), *Estimating Generalization under Distribution Shifts via Domain-Invariant Representations* | 처음 보는 기관 문서에서의 성능 하락과 모델 선택 위험 | 다운로드 완료 |
| 9 | Zhou et al. (2021), *Domain Generalization: A Survey*, arXiv `2103.02503` | 문서를 domain으로 보는 LODO와 domain generalization의 이론적 배경 | 다운로드 완료 |
| 10 | Dalpiaz et al. (2023), *Requirements Classification with Interpretable Machine Learning and Dependency Parsing*, DOI `10.1016/j.infsof.2023.107202` | 요구사항 분류에서 해석 가능한 전통 ML의 위치 | 다운로드 완료 |

### LLM 라벨의 양쪽 근거

| 논문 | 핵심 역할 | 상태 |
|---|---|---|
| Gilardi et al. (2023), *ChatGPT outperforms crowd-workers for text-annotation tasks*, DOI `10.1073/pnas.2305016120` | LLM 라벨링의 비용·일치도 이점을 보고한 긍정 근거 | 다운로드 완료 |
| Kristensen-McLachlan et al. (2025), *Are chatbots reliable text annotators? Sometimes*, DOI `10.1093/pnasnexus/pgaf069` | 과제·프롬프트·모델별 변동과 supervised classifier 우세를 보고한 반대 근거 | 다운로드 완료 |
| Min et al. (2022), *Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?*, DOI `10.18653/v1/2022.emnlp-main.759` | 예시의 정답 라벨 외에도 입력 분포·라벨 공간·형식이 영향을 준다는 근거 | 다운로드 완료 |
| Yoo et al. (2022), *Ground-Truth Labels Matter: A Deeper Look into Input-Label Demonstrations*, DOI `10.18653/v1/2022.emnlp-main.155` | Min et al.의 결론을 재검토하는 반대 근거. 앵커 라벨 불일치 논의에 필요 | 다운로드 완료 |
| Li et al. (2025), *Assessing Crowdsourced Annotations with LLMs*, DOI `10.18653/v1/2025.nlp4dh-1.16` | LLM을 최종 정답자가 아니라 오류 후보 선별기로 쓰는 접근 | 다운로드 완료 |
| Ul Haq et al. (2026), *A Systematic Comparison of Large Language Models for Data Annotation in NER Tasks*, DOI `10.63317/4qnuuw7rjs24` | LLM 라벨 자체뿐 아니라 downstream 모델 성능까지 평가 | 다운로드 완료 |

## 2. 모델·표현 비교에 쓸 논문

| 논문 | 우리 결과와의 연결 | 상태 |
|---|---|---|
| AlDhafer et al. (2022), *An end-to-end deep learning system for requirements classification using recurrent neural networks*, DOI `10.1016/j.infsof.2022.106877` | 단어·문자 입력을 함께 비교한다. 문자 TF-IDF 기준선과 파인튜닝 전 비교 근거 | 구독 접근 확인 필요 |
| Kaur & Kaur (2023), *Improving BERT model for requirements classification by bidirectional LSTM-CNN deep model*, DOI `10.1016/j.compeleceng.2023.108699` | 작은 PROMISE 데이터에서 복합 신경망이 보고한 성능과 우리 보수적 LODO 결과의 차이 | 구독 접근 확인 필요 |
| Dias Canedo & Cordeiro Mendes (2020), *Software Requirements Classification Using Machine Learning Algorithms* | TF-IDF·전통 ML 요구사항 분류 기준선 | 다운로드 경로 추가 확인 필요 |
| Abualhaija et al. (2022), *Deep Learning Methods for Software Requirement Classification: A Performance Study on the PURE dataset*, arXiv `2211.05286` | 다른 요구사항 데이터에서 딥러닝 비교 | 다운로드 완료 |
| Gill et al. (2023), *Software Requirements Prioritisation Using Machine Learning* | 분류가 아닌 실무 우선순위 예측과 비용 맥락 | 다운로드 완료 |
| Cicekli et al. (2023), *Developing an Advanced Software Requirements Classification Model Using BERT* | 영어 외 언어·실제 프로젝트·BERT 비교 | OA, 자동 다운로드 차단 |
| ASME (2023), *Deep Neural Networks in NLP for Classifying Requirements by Origin and Functionality*, DOI `10.1115/1.4063764` | 5개 프로젝트의 inter-document 평가가 우리 문서 일반화와 특히 가까움 | 구독 접근 확인 필요 |

## 3. 계약·조달 문서에 쓸 논문

| 논문 | 우리 결과와의 연결 | 상태 |
|---|---|---|
| El-Sayegh et al. (2025), *Automated construction contract analysis for risk and responsibility assessment using NLP and ML*, DOI `10.1016/j.compind.2025.104251` | 위험·책임 다중 분류와 입찰 전 검토 지원이라는 실무 목적이 매우 가까움 | OA, 다운로드 경로 추가 확인 필요 |
| *Inherent risks identification in a contract document through automated rule generation* (2025) | BERT·의존구문·규칙을 결합한 계약 위험 탐지 | 구독 접근 확인 필요 |
| *Classification of Bid Notice Sections using BERT, DistilBERT and ModernBERT* (2026), DOI `10.1016/j.procs.2026.02.453` | 공공조달 문서 구간 분류와 자동 감사 | OA, 다운로드 경로 추가 확인 필요 |
| *Intelligent RFQ Summarization Using NLP, Text Mining, and ML Techniques* (2021) | 견적 요청서에서 중요 요구를 놓치지 않기 위한 자동 검토 | 구독 접근 확인 필요 |

## 4. 현재 논문에 바로 쓸 연결

- **Related Works:** 요구사항 자동 분류 → 계약 위험 분류 → LLM silver label → 문서 간 일반화 순서로 구성한다.
- **Methods:** Liu, Gupta, Xu를 동적 앵커의 유사도·커버리지·순서 설계 근거로 쓴다.
- **Discussion:** 기존 요구사항 분류 연구가 주로 작은 공용 데이터와 행 단위 분할에 의존했는지 확인하고, 우리 10문서 LODO와 대비한다.
- **Negative results:** 임베딩·복합 feature가 항상 강하지 않다는 결론은 데이터 크기, 언어, 평가 분할과 함께 비교한다. 모델 이름만 나란히 놓지 않는다.
- **Limitations:** LLM 라벨 신뢰성, 앵커 민감도, 문서별 label shift를 서로 다른 한계로 쓴다.

## 5. 다음 확인 순서

1. 1순위 systematic mapping의 참고문헌에서 실제 문서 단위 분할 연구를 찾는다.
2. Moon et al.과 ASME 논문의 평가 분할이 요구사항 무작위인지 문서 분리인지 확인한다.
3. LLM 라벨 survey에서 사람 감사, 합의도, downstream 평가 권고를 추출한다.
4. 앵커 선택 논문과 현재 `anchors_used` 결과를 연결한다.
5. 논문별 데이터 크기·언어·분할·macro F1·소수 클래스 recall 표를 만든다.
