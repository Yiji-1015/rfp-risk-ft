# 동적 앵커 블록 및 프롬프트 주입 템플릿

- **버전**: `anchor-block-v1`
- **배치 위치**: User 메시지 앞단 (System 블록에 넣지 않고 User 메시지에 배치하여 System 프롬프트 캐시 유지 및 공정한 통제 비교 보장, [결정 18](file:///c:/Users/LLOYDK/Desktop/proposal-automation/rfp-risk-ft/docs/history/decisions-02.md#18-앵커-검색기의-라벨링-파이프라인-연결))
- **검색 알고리즘**:
  - `fewshot-similarity`: 쿼리와 코사인 유사도 상위 Top-k 인출 ([결정 13](file:///c:/Users/LLOYDK/Desktop/proposal-automation/rfp-risk-ft/docs/history/decisions-02.md#13-3개-대표-rfp-라벨링-파일럿259건-결과-및-단일-모델-통제-원칙))
  - `fewshot-stratified`: `통상수용` 1건 + `견적반영` 1건 + `계약·질의검토` 1건 층화 균형 인출 ([결정 14](file:///c:/Users/LLOYDK/Desktop/proposal-automation/rfp-risk-ft/docs/history/decisions-02.md#14-학술-선행연구-기반-층화-퓨샷-검색-stratified-few-shot-retrieval-채택), [결정 22](file:///c:/Users/LLOYDK/Desktop/proposal-automation/rfp-risk-ft/docs/history/decisions-02.md#22-앵커-풀-v1-구축과-첫-few-shot-실행))

---

## 1. 프롬프트 헤더 및 주입 구조

```text
[참고 사례]
아래는 다른 기관 RFP에서 이미 검토가 끝난 요구사항과 그 판정이다. 판정 기준의 눈높이를 맞추는 용도로만 쓴다.
사례와 문구가 비슷해도 제공 주체, 무상 범위, 수량 상한, 검수 기준, 책임 범위가 다르면 판정은 달라야 한다.
사례의 판정을 그대로 따라가지 말고, 아래 대상 요구사항의 원문에 근거해 판단한다.

사례 1 (유사도 {similarity} / 공통 어휘: {overlap_terms})
요구사항명: {requirement_name}
내용: {raw_requirement_text}
판정: {primary_action}
이유: {reasoning}

사례 2 (유사도 {similarity} / 공통 어휘: {overlap_terms})
...

사례 3 (유사도 {similarity} / 공통 어휘: {overlap_terms})
...

[대상 요구사항]
[요구사항 ID]: {requirement_uid}
[요구사항명]: {requirement_name}
[요구사항 내용]:
{raw_requirement_text}
```

---

## 2. 동적 인출 원칙

1. **동일 문서 마스킹 (Leakage 차단, [결정 10](file:///c:/Users/LLOYDK/Desktop/proposal-automation/rfp-risk-ft/docs/history/decisions-02.md#10-dynamic-few-shot-앵커링-시-데이터-누수-하드-차단))**:
   - `target_document_id == anchor_document_id`인 앵커는 유사도 계산에서 `-1.0`으로 하드 제외하여 동일 기관 문맥 누수를 원천 차단합니다.
2. **공통 핵심 어휘 노출 ([결정 12](file:///c:/Users/LLOYDK/Desktop/proposal-automation/rfp-risk-ft/docs/history/decisions-02.md#12-in-context-few-shot-프롬프트-고도화-계획-llm-wiki-연계))**:
   - 검색 인출 시 매칭된 단어/어휘(`overlap_terms`)를 프롬프트에 명시하여 LLM이 인과적 맥락을 인지하도록 유도합니다.
