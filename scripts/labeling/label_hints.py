"""라벨러에게 판정이 아니라 **볼 곳**을 주는 두 가지 힌트를 만든다.

1. 드문 표현 — 라벨 없이 말뭉치 통계만. 이 요구사항에는 있는데 다른 요구사항에는 드문 어절.
   `폐쇄망`·`UNIPASS`·`품질마크`처럼 아무도 목록에 넣지 않은 도메인 바늘이 저절로 올라온다.
   관행 조항의 칼(`일체`·`모든 비용`)은 어휘가 흔해 여기서는 안 잡힌다.
2. 주목 줄 — 지금 라벨로 학습한 word+char 모델이 줄 단위로 봤을 때 검토 확률이 높은 줄.
   긴 원문에서 라벨러가 지나친 줄을 짚는다. 모델이 이미 어렴풋이 아는 종류의 칼만 짚으므로
   1과 약점이 반대다. **그 문서를 보지 않은 fold 모델**을 쓴다. 전체로 학습한 모델은 그 건의
   현재 라벨을 되돌려줄 뿐이다.

둘 다 방향(위험하다/아니다)을 붙이지 않는다. 판정은 라벨러가 하고, 했으면 인용한다.
사전을 쓰지 않는 이유는 사전이 곧 결정 21을 정규식으로 다시 쓰는 것이기 때문이다.
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from scripts.labeling.label_dataset import load_label_dataset

RARE_DOC_FREQ = 0.03  # 문서빈도 3% 이하만 "드문" 것으로 본다
RARE_TOP_K = 5
LINE_MIN_CHARS = 15
LINE_TOP_K = 2
LINE_MIN_COUNT = 3  # 줄이 이만큼은 있어야 "주목 줄"이 의미가 있다


class HintBuilder:
    def __init__(self) -> None:
        # 무거운 의존은 여기서만 끌어온다. dry-run과 v5 실행은 이 모듈을 열지 않는다.
        from scripts.evaluation.baselines import WORD_CHAR_BALANCED, _fit_pipeline
        from scripts.evaluation.folds import make_lodo_folds

        rows, _ = load_label_dataset()
        self.rows = {r["requirement_uid"]: r for r in rows}
        texts = [r["model_text"] for r in rows]
        self.vec = TfidfVectorizer(analyzer="word", min_df=2, sublinear_tf=True).fit(texts)
        self.df = np.asarray((self.vec.transform(texts) > 0).sum(0)).ravel()
        self.vocab = self.vec.get_feature_names_out()
        self.rare_cutoff = RARE_DOC_FREQ * len(texts)
        # 문서별로 그 문서를 평가로 뺀 fold의 모델. 10번 학습한다.
        self.models = {}
        for fold in make_lodo_folds(rows):
            fit, _, _ = fold.split(rows)
            pipe = _fit_pipeline(WORD_CHAR_BALANCED, fit)
            clf = pipe.steps[-1][1]
            review = [list(clf.classes_).index(l) for l in ("견적반영", "계약·질의검토")]
            self.models[fold.test_document] = (Pipeline(pipe.steps[:-1]), clf, review)

    def rare_terms(self, uid: str) -> list[str]:
        x = self.vec.transform([self.rows[uid]["model_text"]]).toarray().ravel()
        order = [i for i in np.argsort(-x) if x[i] > 0 and self.df[i] <= self.rare_cutoff and len(self.vocab[i]) >= 2]
        return [str(self.vocab[i]) for i in order[:RARE_TOP_K]]

    def attended_lines(self, uid: str) -> list[str]:
        row = self.rows[uid]
        lines = [l for l in row["model_text"].splitlines() if len(l.strip()) >= LINE_MIN_CHARS]
        if len(lines) < LINE_MIN_COUNT:
            return []
        feats, clf, review = self.models[row["document_id"]]
        p = clf.predict_proba(feats.transform(lines))[:, review].sum(1)
        return [lines[i].strip() for i in np.argsort(-p)[:LINE_TOP_K]]

    def for_uid(self, uid: str) -> str:
        parts = []
        rare = self.rare_terms(uid)
        if rare:
            parts.append("[드문 표현] " + ", ".join(rare))
        lines = self.attended_lines(uid)
        if lines:
            parts.append("[주목 줄]\n" + "\n".join(lines))
        return "\n".join(parts)


if __name__ == "__main__":
    import sys

    builder = HintBuilder()
    for uid in sys.argv[1:] or ["kac_ai_work_platform:GW-007", "mfds_drug_ai_review:PSR-010"]:
        print(f"=== {uid}\n{builder.for_uid(uid)}\n")
