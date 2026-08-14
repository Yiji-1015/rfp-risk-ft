# 연구 설계 및 데이터셋 결정 기록 — 02

> 기준일: 2026-08-12
> 범위: 결정 9~16
> 목적: 앵커 검색, LLM 라벨링 실험, 예산 및 출력 스키마 결정의 상세 이력을 논문 작성에 재사용할 수 있도록 보존한다.

> **문서 역할:** 결정 기록 분권 002다. 기존 결정을 지우지 않고 변경 이유와 새 버전을 이어서 기록한다.

## 9. Pure TF-IDF 앵커 검색기 및 In-Memory 검색

### 결정

Dynamic Few-shot 앵커링 시 별도의 무거운 외부 벡터 DB(Chroma, Qdrant 등)나 딥러닝 임베딩 모델 서버 없이, Pure TF-IDF 및 `scikit-learn` 기반 코사인 유사도 연산으로 앵커를 동적 인출한다.

- 희소 벡터(Sparse Vector) 표현: 어절(Word) 1~2gram 및 자소/문자 n-gram 결합
- `sublinear_tf=True`를 적용하여 고빈도 단어의 편향을 완화
- In-Memory 상에서 $O(N)$ 코사인 유사도 즉시 연산 (1,024건 대상 < 0.001초)

### 근거

소규모 RFP 요구사항 데이터셋 특성상 특정 계약/기술 독소 키워드(`무상`, `지체상금`, `250TB`, `K-RMF`, `상주`)의 정확한 일치(Lexical Match)를 잡는 것이 실무 위험 판정에 가장 직관적이고 효과적이다. 외부 인프라 부하 없이 독립 실행이 가능하다.

---

## 10. Dynamic Few-shot 앵커링 시 데이터 누수 하드 차단

### 결정

Dynamic Few-shot 프롬프트를 구성할 때 Target 요구사항의 출처 문서와 동일한 `document_id`를 가진 앵커는 유사도 계산 대상에서 하드 필터링(`similarities[idx] = -1.0`)하여 제외한다.

```python
# target과 동일한 document_id를 가진 앵커는 자동 차단
if anchor.get("document_id") == target_doc_id:
    similarities[idx] = -1.0
```

### 근거

동일 RFP 문서 내부의 다른 요구사항을 앵커 예시로 주입할 경우, 기관 고유의 서식이나 문맥이 그대로 유입되어 평가(Validation/Test) 시 문서 단위 일반화 성능을 왜곡할 수 있다.

---

## 11. 3개 대표 RFP 기반 1차 라벨링 비교 파일럿

### 결정

전체 1,024건 라벨링에 앞서 대표 성격의 RFP 3개 문서(총 259건)를 선별하여 Zero-shot vs Pure TF-IDF Dynamic Few-shot 비교 파일럿을 먼저 수행한다.

- 대상 문서: `kac_ai_work_platform` (86건), `incheon_airport_digital_work` (78건), `ccrs_ai_platform` (95건)
- 비교 지표: 전체 라벨 일치율(Agreement Rate %), 라벨 전이 행렬(Shift Matrix), 앵커 도입 후 판정 변동 사유 분석

### 근거

공공 SI 수주 현실상 불합리해 보이는 문구라도 10개 기관 공통 관행인 조항(통상수용)과 특정 사업의 기습 독소조항(계약검토)을 구분하기 위해, 앵커링 도입 전후의 라벨 변화 양상을 눈으로 직접 확인하고 라벨링 기준을 유동적으로 조정하기 위함이다.

---

## 12. In-Context Few-Shot 프롬프트 고도화 계획 (llm-wiki 연계)

### 결정

Dynamic Few-shot 앵커 주입 시 단순 예시 나열 방식에서 벗어나, Target 요구사항과 앵커 간의 **공통 핵심 키워드(Overlap Terms)** 및 **TF-IDF 유사도 점수**를 프롬프트 구조에 함께 포함시키는 고도화 방안을 후속 파이프라인에 반영한다.

### 근거

인-컨텍스트 학습(In-Context Learning) 시 LLM이 단순히 앵커의 정답만 모방하는 대신, "어떤 키워드 유사성 때문에 이 예시가 인출되었는지" 맥락을 인지하게 만들어 주 라벨 판정의 인과성과 일관성을 극대화하기 위함이다.

---

## 13. 3개 대표 RFP 라벨링 파일럿(259건) 결과 및 단일 모델 통제 원칙

### 결정

3개 대표 RFP(259건)에 대해 Zero-shot vs Pure TF-IDF Dynamic Few-shot 라벨링 비교 실험을 완수하였으며, 연구의 엄밀성과 통제 변수(Controlled Experiment) 유지를 위해 100% 동일한 단일 LLM(`gemini-3.5-flash-lite`)을 사용하였다.

- **실험 수치 결과**:
  - 대상 문서: `kac_ai_work_platform` (86건), `incheon_airport_digital_work` (78건), `ccrs_ai_platform` (95건)
  - 총 요구사항 수: 259건
  - 라벨 일치 건수: 182 / 259건
  - 전체 라벨 일치율 (Agreement Rate): **70.27%**
  - 판정 변동 건수 (Action Shift): **77건 (29.73%)**

### 연구 인사이트 및 관행 반영 효과

1. **Zero-shot의 과도한 보수적 방어 판정**: Zero-shot 단독 실행 시 기술 난이도가 높거나 모호한 AI/RAG 요구사항(예: RAG 파이프라인 구축, AI 가드레일, 준실시간 평가 모니터링 등)에 대해 54건을 `계약·질의검토`로 과도하게 매핑함.
2. **앵커링(Few-shot) 도입을 통한 현실적 교정**: 타 공공 RFP 앵커 예시 3개가 프롬프트로 주입되자, LLM이 타 공공기관 구축 사업의 일반적인 SI/개발 공수 범위를 학습하여 `계약·질의검토` 54건 중 **26건(48.1%)이 현실적인 `견적반영`으로 교정**됨.
3. **도메인 통상 관행 반영 확인**: "개별 요구사항만 볼 때는 불합리해 보여도 타 기관 공통 관행인 조항"을 앵커 검색이 성공적으로 포착하여, 실무 현장에 적합한 라벨(통상수용/견적반영)을 부여함을 통계 및 질적 사례로 입증함.

### 발견된 한계 및 향후 개선 대책 (Anchor Label Bias)

- **견적반영 라벨 쏠림 현상 관찰**:
  - Dynamic Few-shot 적용 후 `견적반영` 라벨 비율이 **68.7% (178/259건)**로 증가함.
  - **원인 분석**: RFP 요구사항 문항 특성상 공수/비용 관련 항목이 다수이며, Top-k 유사도 검색 시 `견적반영` 앵커 예시가 과다 주입되어 LLM이 리스크 항목을 과도하게 안일하게 순화(Softening Bias)시킬 위험이 관찰됨.
- **향후 개선 대책 (Balanced Stratified Few-shot)**:
  - 앵커 인출 시 단순 유사도 Top-k만 추출하지 않고, **`통상수용`, `견적반영`, `계약·질의검토` 각 3가지 라벨의 대표 앵커를 1개씩 균형 있게 주입(Stratified Selection)**하여 앵커 라벨 편향을 억제하는 고도화 방안을 2차 전체 라벨링 단계에 적용하기로 결정함.

---

## 14. 학술 선행연구 기반 층화 퓨샷 검색 (Stratified Few-shot Retrieval) 채택

### 결정

Dynamic Few-shot 앵커 주입 시 발생한 68.7% '견적반영' 라벨 쏠림(Majority Label Bias)을 해결하기 위해, 최신 NLP 학술 문헌(Zhao et al., ICML 2021; Fei et al., ACL 2023; Gao et al., ICLR 2025)에서 입증된 **층화 퓨샷 검색(Stratified Retrieval: 라벨별 1:1:1 비율 강제 인출)**을 2차 비교 파일럿 및 전수 라벨링 알고리즘으로 최종 채택한다.

### 학술적 근거

1. **Zhao et al. (ICML 2021, *Calibrate Before Use*)**:
   - In-Context Learning에서 LLM은 프롬프트 내에 가장 자주 나타난 레이블을 맹목적으로 선호하는 **다수 레이블 편향(Majority Label Bias)**을 가짐을 통계적으로 입증.
2. **Fei et al. (ACL 2023, *Mitigating Label Biases for In-context Learning*)**:
   - 전문 도메인 어휘(예: '단가', '산출물', '일정')가 특정 클래스와 결합될 때 발생하는 **도메인-컨텍스트 레이블 편향(Domain-Label Bias)**을 규명.
3. **Gao et al. (ICLR 2025, *Exploring Imbalanced Annotations for Effective ICL*)**:
   - 앵커 예시의 레이블 불균형이 심할수록 LLM의 **작업 학습(Task Learning)** 능력이 파탄나며, 희소 클래스에 대한 위음성(False Negative)이 급증함을 증명.

### 실행 구현 알고리즘

- 코사인 유사도 상위 Top-3를 맹목적으로 추출하는 방식을 폐기하고,
- 앵커 후보 풀을 `통상수용`, `견적반영`, `계약·질의검토` 3개 라벨 인덱스로 분리하여 **각 라벨별로 쿼리와 가장 유사한 앵커를 1개씩(총 3개, 1:1:1 비율) 추출**하여 프롬프트에 제공함으로써 앵커 레이블 편향을 물리적으로 하드 차단함.

---

## 15. 예측 가능한 API 예산 제어 및 사전 비용 추정 원칙 (Predictable Budget Control)

### 결정

연구 및 데이터 구축 시 무분별하거나 예측 불가능한 API 비용 지출을 원천 차단하기 위해, **사전 비용 추정(Pre-Execution Cost Estimation)** 및 **예산 상한 캡(Budget Safety Cap)** 원칙을 시스템 운영 표준으로 확정한다.

- **사전 비용 계산 및 명시 (Pre-execution Estimation)**:
  - 대량 LLM API 호출 스크립트 실행 전, `(총 요구사항 수 × 프롬프트당 평균 토큰 수 × 모델 단가)`를 계산하여 **총 예상 비용(원화/달러)을 사전 계산하고 명시**한다.
- **예산 상한 캡 및 안전 멈춤 (Budget Safety Cap)**:
  - 사용자가 사전에 허용한 예상 비용 범위를 넘어서는 대규모 파이프라인의 경우, 자동으로 가동을 멈추고 사용자 승인을 받도록 안전 캡(Cap)을 설정한다.
- **투명한 토큰/비용 로깅 (Transparent Cost Logging)**:
  - API 호출 시 실제 소비된 토큰 수와 소요 비용을 실시간으로 추적 및 기록하여 예측 가능하고 신뢰할 수 있는 비용 제어를 보장한다.


---

## 16. LLM 출력 스키마 최소화 원칙 (Output Schema Minimization)

### 결정

LLM API 호출 시 출력 스키마를 **연구 목적에 실질적으로 필요한 최소 필드**로만 구성한다. 불필요한 장문 서술 필드를 포함시키면 출력 토큰 수가 폭발적으로 증가하여 비용 대비 효용이 현저히 낮아진다.

**확정 최소 출력 스키마 (3개 필드)**:

| 필드 | 역할 | 형식 |
|---|---|---|
| `primary_action` | 핵심 분류 라벨 (연구의 Y) | Enum: `통상수용` / `견적반영` / `계약·질의검토` |
| `confidence` | 라벨 불확실성 지표 | Enum: `높음` / `보통` / `낮음` |
| `reasoning` | 판단 근거 (앵커 재사용) | **1~2문장 요약형**으로 길이 제한 |

### 제거 필드 및 사유

다음 필드는 연구 핵심 목표(라벨 분류)와 직접적 관련이 낮으며, 출력 토큰을 과도하게 소비하므로 제거한다.

| 제거 필드 | 제거 사유 |
|---|---|
| `evidence` | `reasoning`과 90% 이상 내용 중복 |
| `missing_information` | 라벨링 본질과 무관, 논문 한계절 논의로 대체 가능 |
| `domain_dependency` | 라벨 분류 결정에 직접 영향 없음 |
| `risk_factors.cost_driver` | 현 파일럿 단계에서 과잉 분석 |
| `risk_factors.scope_uncertainty` | 현 파일럿 단계에서 과잉 분석 |
| `risk_factors.responsibility_risk` | 현 파일럿 단계에서 과잉 분석 |
| `risk_factors.acceptance_risk` | 현 파일럿 단계에서 과잉 분석 |

### 기대 효과

- 출력 토큰 수: 건당 약 800~1,000 토큰 → **약 80~100 토큰 (약 10배 감소)**
- 이에 따른 출력 비용도 약 10배 절감
- `reasoning`을 1~2문장으로 제한하므로, 앵커 주입 시 프롬프트 길이도 함께 단축됨

### 논문 반영 위치

- 연구 방법: 라벨링 파이프라인 설계 및 출력 구조
- 한계 및 후속 연구: 세부 리스크 요인 분류는 추후 2단계 검토 워크플로우에서 수행
