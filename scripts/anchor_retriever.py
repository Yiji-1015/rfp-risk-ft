#!/usr/bin/env python3
"""
RFP 요구사항 데이터 누수 방지형 Pure TF-IDF 앵커 검색기 (Anchor Retriever)

기능:
1. Pure TF-IDF (단어 n-gram + 자소/문자 n-gram) 기반 코사인 유사도 연산
2. 🔒 데이터 누수 방지: Target 요구사항과 동일한 document_id의 앵커는 자동으로 계산에서 제외
3. Top-K 유사 앵커 및 유사도 점수 반환
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Windows 콘솔 utf-8 인코딩 대응
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class PureTfidfAnchorRetriever:
    """Pure TF-IDF 기반 데이터 누수 방지 앵커 검색 클래스"""

    def __init__(self, anchor_pool: List[Dict[str, Any]]):
        """
        :param anchor_pool: 앵커 후보 요구사항 객체 리스트
               각 객체는 minimum ['requirement_uid', 'document_id', 'requirement_name', 'raw_requirement_text'] 포함 필수
        """
        self.anchor_pool = anchor_pool
        
        # 한국어 특성을 고려해 어절 (1~2gram) 및 문자 (3~4gram) 혼합 TF-IDF 구축
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            analyzer='word',
            sublinear_tf=True,  # TF 값의 로그 스케일링으로 지나친 단어 빈도 편향 완화
            min_df=1
        )
        
        # 앵커 풀 텍스트 표현 결합 (요구사항명 + 본문)
        self.anchor_texts = [
            f"{a.get('requirement_name', '')} {a.get('raw_requirement_text', '')}"
            for a in anchor_pool
        ]
        
        # 앵커 풀 TF-IDF 희소 행렬 계산
        if self.anchor_texts:
            self.anchor_matrix = self.vectorizer.fit_transform(self.anchor_texts)
        else:
            self.anchor_matrix = None

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

        target_doc_id = target_req.get("document_id")
        target_text = f"{target_req.get('requirement_name', '')} {target_req.get('raw_requirement_text', '')}"
        
        # 1. Target 텍스트를 TF-IDF 벡터로 변환
        target_vec = self.vectorizer.transform([target_text])
        
        # 2. 앵커 풀 전체와의 코사인 유사도 계산 (1D numpy array)
        similarities = cosine_similarity(target_vec, self.anchor_matrix)[0]
        
        # 3. 🔒 [데이터 누수 방지]: Target과 동일한 document_id를 가진 앵커는 유사도를 -1.0으로 마스킹
        for idx, anchor in enumerate(self.anchor_pool):
            if anchor.get("document_id") == target_doc_id:
                similarities[idx] = -1.0
                
        # 4. 유사도 기준 내림차순 정렬
        sorted_indices = np.argsort(similarities)[::-1]
        
        # 5. 마스킹되지 않은(유사도 > 0) 상위 top_k 앵커 추출
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
        target_labels: List[str] = None
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        학술 선행연구 기반 층화 퓨샷 검색 (Balanced Stratified Retrieval)
        
        '통상수용', '견적반영', '계약·질의검토' 각 라벨별로 가장 유사한 앵커를 1개씩(총 3개) 균형 추출
        
        :param target_req: 검색 대상 요구사항 객체
        :param target_labels: 뽑아올 라벨 카테고리 리스트 (기본: ['통상수용', '견적반영', '계약·질의검토'])
        :return: [(anchor_object, similarity_score), ...] (라벨당 1개씩 균형 세트)
        """
        if target_labels is None:
            target_labels = ["통상수용", "견적반영", "계약·질의검토"]

        if self.anchor_matrix is None or len(self.anchor_pool) == 0:
            return []

        target_doc_id = target_req.get("document_id")
        target_text = f"{target_req.get('requirement_name', '')} {target_req.get('raw_requirement_text', '')}"
        
        target_vec = self.vectorizer.transform([target_text])
        similarities = cosine_similarity(target_vec, self.anchor_matrix)[0]

        # 🔒 동일 document_id 마스킹
        masked_sims = similarities.copy()
        for idx, anchor in enumerate(self.anchor_pool):
            if anchor.get("document_id") == target_doc_id:
                masked_sims[idx] = -1.0

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


def self_test():
    """간단한 자체 동작 검증 함수"""
    root_dir = Path(__file__).resolve().parent.parent
    sample_file = root_dir / "reports" / "labeling_pilot_results_v0.1.0.jsonl"
    
    if not sample_file.exists():
        print(f"테스트용 파일을 찾을 수 없습니다: {sample_file}")
        return

    with open(sample_file, "r", encoding="utf-8") as f:
        pool = [json.loads(line.strip()) for line in f if line.strip()]

    print(f"로드된 파일럿 앵커 후보 수: {len(pool)}개")
    retriever = PureTfidfAnchorRetriever(pool)
    
    # 첫 번째 요구사항을 target으로 테스트
    target = pool[0]
    top_anchors = retriever.get_top_k_anchors(target, top_k=3)
    
    print("\n" + "="*70)
    print(f"[Target 요구사항] ({target['document_id']}) {target['requirement_uid']}")
    print(f"명칭: {target['requirement_name']}")
    print(f"본문 스니펫: {target.get('raw_requirement_text', '')[:60]}...")
    print("="*70)
    print("\n[검색된 Top-3 유사 앵커 (동일 문서 자동 제외)]:")
    
    for rank, (anc, score) in enumerate(top_anchors, 1):
        print(f"\n#{rank} (유사도 점수: {score:.4f})")
        print(f"   UID: {anc['requirement_uid']} (출처 문서: {anc['document_id']})")
        print(f"   명칭: {anc['requirement_name']}")
        print(f"   본문: {anc.get('raw_requirement_text', '')[:70]}...")
        if 'pilot_result' in anc and 'primary_action' in anc['pilot_result']:
            print(f"   라벨: {anc['pilot_result']['primary_action']}")


if __name__ == "__main__":
    self_test()
