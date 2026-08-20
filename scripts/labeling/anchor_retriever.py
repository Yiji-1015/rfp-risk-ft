#!/usr/bin/env python3
"""
RFP 요구사항 데이터 누수 방지형 Pure TF-IDF 앵커 검색기 (Anchor Retriever)

기능:
1. Pure TF-IDF (어절 1~2gram + 문자 3~4gram 결합) 기반 코사인 유사도 연산
2. 🔒 데이터 누수 방지: Target 요구사항과 동일한 document_id의 앵커는 자동으로 계산에서 제외
3. Top-K 유사 앵커 및 유사도 점수 반환
4. 층화 검색: 라벨별 1개씩 균형 인출 (결정 14)
5. 프롬프트 설명용 공통 핵심 어휘 추출 (결정 12)
"""

import sys
from typing import Any, Dict, List, Sequence, Tuple

# Windows 콘솔 utf-8 인코딩 대응
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 결정 9: 어절 n-gram만으로는 한국어 띄어쓰기 변형과 영문 코드(K-RMF, 250TB)를 놓친다.
WORD_NGRAM_RANGE = (1, 2)
CHAR_NGRAM_RANGE = (3, 4)

DEFAULT_LABELS = ("통상수용", "견적반영", "계약·질의검토")


# 결정 28: global few-shot 비교용 고정 앵커.
# 층화 인출의 실측 유사도 중앙값이 0.090이라 "유사 사례 검색"이 아니라
# "판정 기준 캘리브레이션"에 가깝다는 가설을 검증하기 위한 조건이다.
# 모든 입력에 같은 3건을 주입하되, 동일 문서 차단(결정 10)은 유지한다.
#
# 1순위는 사람 확정 앵커에서 라벨별 대표를 고른다. 2순위는 1순위와 다른 문서에서
# 고르므로, 타깃 문서가 1순위와 겹쳐도 라벨당 1건은 항상 확보된다.
GLOBAL_ANCHORS = {
    "통상수용": [
        "defense_intelligent_platform:QUR-001",  # 산출물 17종이지만 표준 절차라 통상 공수
        "korail_genai_isp_ismp:CSR-003",         # 현황분석·연동 검토는 일반 컨설팅 범위
    ],
    "견적반영": [
        "kexim_ai_platform:SER-001",             # 고급 AI 개발 공수지만 범위가 닫혀 있음
        "mfds_drug_ai_review:SFR-001",           # 외부 인증체계 연동 공수
    ],
    "계약·질의검토": [
        "kac_ai_work_platform:AIP-001",          # 라이선스·공급 가능성 미확인
        "genai_incident_response:SFR-001",       # 120B 모델 조달 가능성 미확인
    ],
}

RETRIEVER_VERSION = "tfidf-word-char-v1"


def retriever_config() -> Dict[str, Any]:
    """run manifest에 남길 검색기 설정 (§11.15 실행 조건 보존)."""
    return {
        "retriever_version": RETRIEVER_VERSION,
        "word_ngram_range": list(WORD_NGRAM_RANGE),
        "char_ngram_range": list(CHAR_NGRAM_RANGE),
        "char_analyzer": "char_wb",
        "sublinear_tf": True,
        "similarity": "cosine",
        "same_document_masking": True,
    }


class PureTfidfAnchorRetriever:
    """Pure TF-IDF 기반 데이터 누수 방지 앵커 검색 클래스"""

    def __init__(self, anchor_pool: List[Dict[str, Any]]):
        """
        :param anchor_pool: 앵커 후보 요구사항 객체 리스트
               각 객체는 minimum ['requirement_uid', 'document_id', 'requirement_name', 'raw_requirement_text'] 포함 필수
        """
        self.anchor_pool = anchor_pool

        # 어절 표현: 계약 문구의 어휘 일치를 잡는다.
        self.word_vectorizer = TfidfVectorizer(
            ngram_range=WORD_NGRAM_RANGE,
            analyzer='word',
            sublinear_tf=True,  # TF 값의 로그 스케일링으로 지나친 단어 빈도 편향 완화
            min_df=1
        )
        # 문자 표현: 띄어쓰기 변형, 조사 변화, 영문·숫자 코드를 잡는다.
        self.char_vectorizer = TfidfVectorizer(
            ngram_range=CHAR_NGRAM_RANGE,
            analyzer='char_wb',
            sublinear_tf=True,
            min_df=1
        )

        # 앵커 풀 텍스트 표현 결합 (요구사항명 + 본문)
        self.anchor_texts = [
            f"{a.get('requirement_name', '')} {a.get('raw_requirement_text', '')}"
            for a in anchor_pool
        ]

        # 앵커 풀 TF-IDF 희소 행렬 계산
        # 두 블록 각각이 L2 정규화되므로, 결합 벡터의 코사인 유사도는
        # 어절 유사도와 문자 유사도의 평균과 같다 (가중치 50:50).
        if self.anchor_texts:
            self.anchor_word_matrix = self.word_vectorizer.fit_transform(self.anchor_texts)
            self.anchor_char_matrix = self.char_vectorizer.fit_transform(self.anchor_texts)
            self.anchor_matrix = hstack(
                [self.anchor_word_matrix, self.anchor_char_matrix]
            ).tocsr()
        else:
            self.anchor_word_matrix = None
            self.anchor_char_matrix = None
            self.anchor_matrix = None

    @staticmethod
    def _target_text(target_req: Dict[str, Any]) -> str:
        return (
            f"{target_req.get('requirement_name', '')} "
            f"{target_req.get('raw_requirement_text', '')}"
        )

    def _masked_similarities(self, target_req: Dict[str, Any]) -> np.ndarray:
        """Target과 앵커 풀의 코사인 유사도. 동일 문서 앵커는 -1.0으로 마스킹."""
        target_text = self._target_text(target_req)
        target_vec = hstack(
            [
                self.word_vectorizer.transform([target_text]),
                self.char_vectorizer.transform([target_text]),
            ]
        ).tocsr()

        similarities = cosine_similarity(target_vec, self.anchor_matrix)[0]

        # 🔒 [데이터 누수 방지]: Target과 동일한 document_id를 가진 앵커는 유사도를 -1.0으로 마스킹
        target_doc_id = target_req.get("document_id")
        for idx, anchor in enumerate(self.anchor_pool):
            if anchor.get("document_id") == target_doc_id:
                similarities[idx] = -1.0
        return similarities

    def get_top_k_anchors(
        self,
        target_req: Dict[str, Any],
        top_k: int = 3
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Target 요구사항과 유사한 Top-K 앵커를 검색 (동일 문서 출처 자동 제외)

        :param target_req: 검색 대상 요구사항 객체
        :param top_k: 반환할 앵커 개수
        :return: [(anchor_object, similarity_score), ...]
        """
        if self.anchor_matrix is None or len(self.anchor_pool) == 0:
            return []

        similarities = self._masked_similarities(target_req)

        # 유사도 기준 내림차순 정렬
        sorted_indices = np.argsort(similarities)[::-1]

        # 마스킹되지 않은(유사도 > 0) 상위 top_k 앵커 추출
        results = []
        for idx in sorted_indices:
            score = float(similarities[idx])
            if score <= 0:
                break
            results.append((self.anchor_pool[idx], score))
            if len(results) >= top_k:
                break

        return results

    def get_stratified_top_k_anchors(
        self,
        target_req: Dict[str, Any],
        target_labels: Sequence[str] = None
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        학술 선행연구 기반 층화 퓨샷 검색 (Balanced Stratified Retrieval)

        '통상수용', '견적반영', '계약·질의검토' 각 라벨별로 가장 유사한 앵커를 1개씩(총 3개) 균형 추출

        :param target_req: 검색 대상 요구사항 객체
        :param target_labels: 뽑아올 라벨 카테고리 리스트 (기본: ['통상수용', '견적반영', '계약·질의검토'])
        :return: [(anchor_object, similarity_score), ...] (라벨당 1개씩 균형 세트)
        """
        if target_labels is None:
            target_labels = list(DEFAULT_LABELS)

        if self.anchor_matrix is None or len(self.anchor_pool) == 0:
            return []

        masked_sims = self._masked_similarities(target_req)

        # 라벨별 최고 유사도 앵커 1개씩 선별
        stratified_results = []

        for label in target_labels:
            best_idx = -1
            best_score = -1.0

            for idx, anchor in enumerate(self.anchor_pool):
                score = masked_sims[idx]
                if score <= 0:
                    continue

                # 앵커의 라벨 파악
                anc_label = (
                    anchor.get("primary_action") or
                    anchor.get("zero_shot_result", {}).get("primary_action") or
                    anchor.get("few_shot_result", {}).get("primary_action") or
                    anchor.get("pilot_result", {}).get("primary_action")
                )

                if anc_label == label and score > best_score:
                    best_score = score
                    best_idx = idx

            if best_idx != -1:
                stratified_results.append((self.anchor_pool[best_idx], best_score))

        # 만약 특정 라벨의 앵커가 부족하여 3개가 미달인 경우 일반 Top-K로 보충
        if len(stratified_results) < len(target_labels):
            existing_uids = {anc['requirement_uid'] for anc, _ in stratified_results}
            fallback_anchors = self.get_top_k_anchors(target_req, top_k=len(target_labels) * 2)
            for anc, score in fallback_anchors:
                if anc['requirement_uid'] not in existing_uids:
                    stratified_results.append((anc, score))
                    existing_uids.add(anc['requirement_uid'])
                    if len(stratified_results) >= len(target_labels):
                        break

        return stratified_results

    def get_global_anchors(
        self,
        target_req: Dict[str, Any],
    ) -> List[Tuple[Dict[str, Any], float]]:
        """모든 입력에 같은 앵커를 주입한다. 유사도 계산을 하지 않는다.

        동일 문서 앵커는 차단하고 같은 라벨의 다음 순위로 대체한다.
        검색을 하지 않으므로 similarity는 기록용 0.0으로 둔다.
        """
        by_uid = {a["requirement_uid"]: a for a in self.anchor_pool}
        target_doc = target_req.get("document_id")
        selected = []
        for label, uids in GLOBAL_ANCHORS.items():
            for uid in uids:
                anchor = by_uid.get(uid)
                if anchor is None or anchor.get("document_id") == target_doc:
                    continue
                selected.append((anchor, 0.0))
                break
        return selected

    def get_overlap_terms(
        self,
        target_req: Dict[str, Any],
        anchor_index: int,
        top_n: int = 5,
    ) -> List[str]:
        """
        결정 12: 앵커가 왜 인출됐는지 프롬프트에 설명하기 위한 공통 핵심 어휘.

        어절 벡터만 사용한다. 문자 n-gram은 사람이 읽을 수 없는 조각이라
        프롬프트에 넣으면 오히려 잡음이 된다.
        """
        if self.anchor_word_matrix is None:
            return []
        if not 0 <= anchor_index < len(self.anchor_pool):
            return []

        target_vec = self.word_vectorizer.transform([self._target_text(target_req)])
        anchor_vec = self.anchor_word_matrix[anchor_index]

        # 양쪽 모두에서 비중이 큰 어휘 = 가중치 곱이 큰 어휘
        shared = target_vec.multiply(anchor_vec).tocoo()
        if shared.nnz == 0:
            return []

        vocabulary = self.word_vectorizer.get_feature_names_out()
        ranked = sorted(zip(shared.col, shared.data), key=lambda item: -item[1])
        return [str(vocabulary[col]) for col, _ in ranked[:top_n]]

    def retrieve(
        self,
        target_req: Dict[str, Any],
        *,
        strategy: str,
        top_k: int = 3,
        overlap_terms: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        전략별 앵커 인출 결과를 프롬프트·기록용 dict 목록으로 반환한다.

        :param strategy: 'similarity'(결정 13) / 'stratified'(결정 14) / 'global'(결정 28)
        """
        if strategy == "similarity":
            selected = self.get_top_k_anchors(target_req, top_k=top_k)
        elif strategy == "stratified":
            selected = self.get_stratified_top_k_anchors(target_req)
        elif strategy == "global":
            selected = self.get_global_anchors(target_req)
        else:
            raise ValueError(f"지원하지 않는 앵커 검색 전략: {strategy}")

        index_by_uid = {
            anchor["requirement_uid"]: idx
            for idx, anchor in enumerate(self.anchor_pool)
        }
        # 고정 앵커는 대상과의 유사도로 뽑은 것이 아니다. 공통 어휘를 계산해 붙이면
        # 사실과 다르고, 대상마다 값이 달라져 프롬프트 캐시도 깨진다(결정 29).
        include_overlap = strategy != "global"
        rendered = []
        for anchor, score in selected:
            idx = index_by_uid[anchor["requirement_uid"]]
            rendered.append(
                {
                    "requirement_uid": anchor["requirement_uid"],
                    "document_id": anchor.get("document_id"),
                    "requirement_name": anchor.get("requirement_name", ""),
                    "raw_requirement_text": anchor.get("raw_requirement_text", ""),
                    "primary_action": anchor.get("primary_action"),
                    "reasoning": anchor.get("reasoning", ""),
                    "similarity": round(float(score), 4),
                    "overlap_terms": (
                        self.get_overlap_terms(target_req, idx, top_n=overlap_terms)
                        if include_overlap
                        else []
                    ),
                }
            )
        return rendered
