"""조항 종류 군집의 문서 확산도(universality)를 특징으로 넣으면 기준선이 오르는가.

2026-09-06 18:08이 "조항 종류 단위의 군집은 아직 시험하지 않았다"고 남긴 빈칸이다.
fold마다 학습 문서의 E5 임베딩으로 KMeans를 만들고, 군집별로 '학습 문서 몇 개에
걸쳐 있나'(0~1)를 세서 그 값 하나를 word+char TF-IDF에 붙인다. 평가 행은 가장 가까운
군집의 값을 받을 뿐 군집 생성과 집계 어디에도 들어가지 않는다.

결과(2026-09-07, v5): 여섯 설정 전부 기준선 아래, 우세 최대 6/13. 채택하지 않는다.
decisions-09 참조.

    RFP_DATASET_VERSION=v5 python -m scripts.evaluation.cluster_universality
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, recall_score
from sklearn.pipeline import Pipeline

from scripts.evaluation import baselines as B
from scripts.evaluation.embeddings import load_cached_embeddings
from scripts.evaluation.folds import make_lodo_folds
from scripts.labeling.label_dataset import load_label_dataset

ROOT = Path(__file__).resolve().parents[2]
LABELS = list(B.LABELS)
SPEC = B.WORD_CHAR_BALANCED


def fit_predict(train, test, extra_train=None, extra_test=None) -> list[str]:
    labels = [r["primary_action"] for r in train]
    proto = replace(SPEC, class_weight=B._resolved_class_weight(SPEC, labels)).build()
    vec, clf = Pipeline(proto.steps[:-1]), proto.steps[-1][1]
    x_train = vec.fit_transform(B._select_text(train))
    x_test = vec.transform(B._select_text(test))
    if extra_train is not None:
        x_train = sp.hstack([x_train, sp.csr_matrix(extra_train)]).tocsr()
        x_test = sp.hstack([x_test, sp.csr_matrix(extra_test)]).tocsr()
    clf.fit(x_train, labels)
    return list(clf.predict(x_test))


def span_features(emb, idx, train, test, k, seed=42):
    """군집별 학습 문서 확산도. 학습 행에는 자기 군집 값, 평가 행에는 최근접 군집 값."""
    e_train = emb[[idx[r["requirement_uid"]] for r in train]]
    e_test = emb[[idx[r["requirement_uid"]] for r in test]]
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(e_train)
    c_train, c_test = km.labels_, km.predict(e_test)
    docs = np.array([r["document_id"] for r in train])
    n_docs = len(set(docs))
    span = np.array([len(set(docs[c_train == c])) / n_docs for c in range(k)])
    return span[c_train][:, None], span[c_test][:, None]


def score(gold, pred) -> tuple[float, float]:
    return (
        f1_score(gold, pred, labels=LABELS, average="macro", zero_division=0),
        recall_score(gold, pred, labels=["계약·질의검토"], average="macro", zero_division=0),
    )


def main() -> None:
    rows, meta = load_label_dataset()
    version = meta["dataset_version"].rsplit("_", 1)[-1]
    cache = ROOT / "data" / "processed" / f"multilingual-e5-small.{version}.npz"
    emb = load_cached_embeddings(cache, rows)
    if emb is None:
        sys.exit(f"임베딩 캐시가 없거나 데이터와 맞지 않습니다: {cache}")
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    idx = {r["requirement_uid"]: i for i, r in enumerate(rows)}
    folds = make_lodo_folds(rows)

    # 신호가 있기는 한가 — 전체에서 k=80 군집의 확산도 4분위별 라벨 분포 (진단, 학습 아님)
    km = KMeans(n_clusters=80, n_init=10, random_state=42).fit(emb)
    docs_all = np.array([r["document_id"] for r in rows])
    lab_all = np.array([r["primary_action"] for r in rows])
    n_docs = len(set(docs_all))
    span_all = np.array([len(set(docs_all[km.labels_ == c])) / n_docs for c in range(80)])[km.labels_]
    print(f"확산도 4분위별 라벨 비율 (전체 {len(rows)}건, k=80 — 진단용)")
    for lo, hi in [(0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)]:
        m = (span_all >= lo) & (span_all < hi)
        if m.sum():
            print(f"  확산도 {lo:.2f}~{min(hi, 1):.2f}: {m.sum():4d}건  "
                  + "  ".join(f"{l} {(lab_all[m] == l).mean() * 100:4.1f}%" for l in LABELS))

    results = {}
    base_f, base_r = [], []
    for f in folds:
        train, _, test = f.split(rows)
        gold = [r["primary_action"] for r in test]
        m, rc = score(gold, fit_predict(train, test))
        base_f.append(m); base_r.append(rc)
    print(f"\n기준선 word+char        fold 평균 {np.mean(base_f):.4f}  검토 recall {np.mean(base_r):.4f}")
    results["baseline"] = {"macro_f1": float(np.mean(base_f)), "review_recall": float(np.mean(base_r)),
                           "folds": [float(x) for x in base_f]}

    for k in (40, 80, 160):
        for w in (1.0, 3.0):
            fs, rs = [], []
            for f in folds:
                train, _, test = f.split(rows)
                gold = [r["primary_action"] for r in test]
                a, b = span_features(emb, idx, train, test, k)
                m, rc = score(gold, fit_predict(train, test, a * w, b * w))
                fs.append(m); rs.append(rc)
            d = np.array(fs) - np.array(base_f)
            wins = int((d > 0).sum())
            print(f"k={k:<4} w={w:<3} fold 평균 {np.mean(fs):.4f} ({np.mean(d):+.4f}, 우세 {wins}/{len(folds)}, "
                  f"fold 차이 {d.min():+.3f}~{d.max():+.3f})  검토 recall {np.mean(rs):.4f}")
            results[f"k{k}_w{w}"] = {"macro_f1": float(np.mean(fs)), "delta": float(np.mean(d)), "wins": wins,
                                     "review_recall": float(np.mean(rs)), "folds": [float(x) for x in fs]}

    out = ROOT / "reports" / "current" / version / "cluster_universality.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
