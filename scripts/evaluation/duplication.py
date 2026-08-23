"""문서를 넘는 표준 문구 반복을 측정한다.

공공 RFP는 표준 문구를 공유한다. 감리 대응이나 개인정보보호법 준수 같은 조항이
기관을 바꿔가며 거의 그대로 나온다. **이것은 중복도 누수도 아니다.** 같은 요구사항이
두 번 들어간 것이 아니라 서로 다른 사업의 서로 다른 요구사항이 문구를 공유하는 것이고,
실제 배포에서도 새 RFP에 같은 조항이 나온다. 모델이 알아보고 답하면 그건 정답이다
(결정 34, docs/issues/006).

남는 문제는 **점수가 두 능력을 섞는다**는 것이다. 반복되는 표준 문구를 알아보는 능력과
처음 보는 요구사항을 판단하는 능력이 한 숫자로 합쳐지고, 섞이는 비율이 fold마다 다르다.
그래서 fold 간 점수를 나란히 놓으면 능력이 아니라 구성을 보게 된다.

대응은 데이터가 아니라 보고 쪽이다. 이 모듈은 각 행이 **학습에서 볼 수 있었던 문구와
얼마나 겹치는지**를 재서, 평가 점수를 전체 / 반복 제외 / 반복만으로 나눌 수 있게 한다.

임계값에 따라 결론이 크게 달라진다(0.5면 16.3%, 0.6이면 12.0%, 0.8이면 4.9%).
그래서 임계값을 기본값에 숨기지 않고 호출부가 항상 명시하게 두고, 결과에도 함께 싣는다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 문자 n-gram을 쓴다. 한국어 RFP는 띄어쓰기 변형과 기관별 용어가 많아 단어 단위로는
# 같은 문구가 다른 문구로 보인다(§9.2).
NGRAM_RANGE = (3, 4)
ANALYZER = "char_wb"

# 기본 임계값. 근거는 결정 34의 실측이고, 유일하게 옳은 값이라는 뜻은 아니다.
DEFAULT_THRESHOLD = 0.6


@dataclass(frozen=True)
class DuplicationResult:
    """각 행이 참조 집합의 문구와 얼마나 겹치는가.

    :param nearest_similarity: 행별 최근접 코사인 유사도. 참조 집합이 비었으면 0.0.
    :param nearest_uid: 그 최근접 상대의 `requirement_uid`. 없으면 None.
    :param is_repeat: `nearest_similarity >= threshold`
    :param threshold: 사용한 임계값. 결과와 함께 다녀야 해석이 가능하다.
    """

    nearest_similarity: np.ndarray
    nearest_uid: list[str | None]
    is_repeat: np.ndarray
    threshold: float

    @property
    def repeat_rate(self) -> float:
        """반복 문구로 분류된 비율."""
        return float(self.is_repeat.mean()) if len(self.is_repeat) else 0.0


def nearest_similarity(
    target_rows: Sequence[dict[str, Any]],
    reference_rows: Sequence[dict[str, Any]],
    *,
    threshold: float = DEFAULT_THRESHOLD,
) -> DuplicationResult:
    """`target_rows`의 각 행이 `reference_rows`와 얼마나 겹치는지 잰다.

    참조 집합을 인자로 받는 것이 이 함수의 요점이다. "학습에서 볼 수 있었던 것"은
    fold마다 다르므로, 전역 중복도가 아니라 **fold의 학습 집합에 대한 중복도**를 재야
    그 fold의 점수 구성을 설명할 수 있다.

    LODO에서는 학습 집합이 곧 '자기 문서를 뺀 나머지 전부'라 결과가 전역
    문서 간 중복도와 같아진다. 그래도 참조 집합을 명시적으로 받는 이유는 분할 방식이
    바뀌어도 계산이 조용히 틀리지 않게 하기 위해서다.

    벡터라이저는 target과 reference를 **합쳐서** 학습시킨다. 어휘를 한쪽에만 맞추면
    다른 쪽의 문구가 표현되지 않아 유사도가 낮게 나온다.
    """
    if not target_rows:
        raise ValueError("target_rows가 비어 있습니다.")
    if not reference_rows:
        n = len(target_rows)
        return DuplicationResult(
            nearest_similarity=np.zeros(n),
            nearest_uid=[None] * n,
            is_repeat=np.zeros(n, dtype=bool),
            threshold=threshold,
        )

    target_texts = [r["raw_requirement_text"] for r in target_rows]
    reference_texts = [r["raw_requirement_text"] for r in reference_rows]

    vectorizer = TfidfVectorizer(
        analyzer=ANALYZER,
        ngram_range=NGRAM_RANGE,
        sublinear_tf=True,
        min_df=2,
    )
    vectorizer.fit(target_texts + reference_texts)
    similarity = cosine_similarity(
        vectorizer.transform(target_texts),
        vectorizer.transform(reference_texts),
    )

    best = similarity.argmax(axis=1)
    scores = similarity[np.arange(len(target_rows)), best]
    return DuplicationResult(
        nearest_similarity=scores,
        nearest_uid=[reference_rows[j]["requirement_uid"] for j in best],
        is_repeat=scores >= threshold,
        threshold=threshold,
    )


def cross_document_similarity(
    rows: Sequence[dict[str, Any]],
    *,
    threshold: float = DEFAULT_THRESHOLD,
) -> DuplicationResult:
    """각 행에 대해 **다른 문서**의 요구사항 중 가장 비슷한 것을 찾는다.

    같은 문서 안의 유사 문구는 제외한다. 문서 단위로 분할하므로 같은 문서는 언제나
    같은 쪽(학습이든 평가든)에 함께 가고, 따라서 fold를 넘는 노출이 되지 않는다.
    """
    texts = [r["raw_requirement_text"] for r in rows]
    vectorizer = TfidfVectorizer(
        analyzer=ANALYZER,
        ngram_range=NGRAM_RANGE,
        sublinear_tf=True,
        min_df=2,
    )
    similarity = cosine_similarity(vectorizer.fit_transform(texts))

    # 자기 자신과 같은 문서의 행은 후보에서 뺀다.
    np.fill_diagonal(similarity, -1.0)
    document_ids = np.array([r["document_id"] for r in rows])
    similarity[document_ids[:, None] == document_ids[None, :]] = -1.0

    best = similarity.argmax(axis=1)
    scores = similarity[np.arange(len(rows)), best]
    return DuplicationResult(
        nearest_similarity=scores,
        nearest_uid=[rows[j]["requirement_uid"] for j in best],
        is_repeat=scores >= threshold,
        threshold=threshold,
    )
