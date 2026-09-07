"""E5 임베딩 군집을 점수가 아니라 진단에 쓴다.

세 가지를 잰다. 어느 것도 모델을 학습하지 않는다.

1. 문서 간 라벨 일관성 — 뜻이 가까운 조항이 **다른 문서**에서 어떤 라벨을 받았나.
   각 요구사항의 다른 문서 최근접 이웃 5건 다수결이 자기 라벨과 맞는 비율. 모델이 맞힌
   건·틀린 건·경계 혼동·v5·v6 불일치별로 나눠 본다. 문장이 비슷한데 라벨이 갈리면 그
   차이는 문서 맥락에서 온다. 같은 것을 word+char TF-IDF 유사도로도 재서 대조한다.
2. 아웃라이어 — 최근접 이웃과의 거리가 먼 상위 5%. 추출 결함 검수와 오답 겹침 확인.
3. 앵커 분포 — 동결 앵커 100건이 군집 몇 개를 덮는가(issues/008의 배경).

노트북 21이 이 모듈의 함수를 그대로 부른다.

    RFP_DATASET_VERSION=v5 python -m scripts.evaluation.cluster_diagnostics
"""
from __future__ import annotations

import collections
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score

from scripts.evaluation import baselines as B
from scripts.evaluation.embeddings import load_cached_embeddings
from scripts.labeling.label_dataset import load_label_dataset

ROOT = Path(__file__).resolve().parents[2]
LABELS = ["통상수용", "견적반영", "계약·질의검토"]
BOUNDARY = {"견적반영", "계약·질의검토"}
K_NEIGHBORS = 5
K_CLUSTERS = 80


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    z, p = 1.96, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


# ---------- 표현 ----------

def e5_matrix(rows: Sequence[dict[str, Any]], version: str) -> np.ndarray:
    """v5 캐시(`multilingual-e5-small.v5.npz`)를 읽어 단위 벡터로 만든다."""
    cache = ROOT / "data" / "processed" / f"multilingual-e5-small.{version}.npz"
    emb = load_cached_embeddings(cache, rows)
    if emb is None:
        raise FileNotFoundError(f"임베딩 캐시가 없거나 데이터와 맞지 않습니다: {cache}")
    return emb / np.linalg.norm(emb, axis=1, keepdims=True)


def tfidf_similarity(rows: Sequence[dict[str, Any]]) -> np.ndarray:
    """기준선과 같은 word+char TF-IDF 코사인. 전체에 비지도로 맞춘다(라벨 안 씀)."""
    x = B.WORD_CHAR_BALANCED.build().steps[0][1].fit_transform(B._select_text(rows)).tocsr()
    return (x @ x.T).toarray()


# ---------- 1. 문서 간 이웃 라벨 일관성 ----------

def cross_document_neighbors(sim: np.ndarray, doc: np.ndarray, lab: np.ndarray, k: int = K_NEIGHBORS):
    """다른 문서에서만 최근접 k건을 고른다. (이웃 인덱스, 이웃 유사도, 다수결 라벨, 일치 여부)"""
    masked = np.where(doc[:, None] == doc[None, :], -np.inf, sim)
    nn = np.argsort(-masked, axis=1)[:, :k]
    nn_sim = np.take_along_axis(masked, nn, axis=1)
    major = np.array([collections.Counter(lab[i]).most_common(1)[0][0] for i in nn])
    return nn, nn_sim, major, major == lab


def group_masks(rows: Sequence[dict[str, Any]], oof: dict[str, dict], v6: dict[str, str]) -> dict[str, np.ndarray]:
    """모델 정오답·경계 혼동·v5/v6 불일치·라벨별 집단."""
    uid = [r["requirement_uid"] for r in rows]
    lab = np.array([r["primary_action"] for r in rows])
    groups = {"전체": np.ones(len(rows), bool)}
    if oof:
        pred = np.array([oof[u]["word_char_logistic_pred"] if u in oof else "" for u in uid])
        evaluated = pred != ""
        wrong = evaluated & (pred != lab)
        boundary = wrong & np.array([{g, p} == BOUNDARY for g, p in zip(lab, pred)])
        groups.update({"모델이 맞힌 건": evaluated & (pred == lab), "경계 밖 오답": wrong & ~boundary, "경계 혼동": boundary})
    if v6:
        unstable = np.array([v6.get(u) != l for u, l in zip(uid, lab)])
        groups.update({"v5·v6 일치": ~unstable, "v5·v6 불일치": unstable})
    for l in LABELS:
        groups[f"라벨={l}"] = lab == l
    return groups


def consistency_table(agree: np.ndarray, groups: dict[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    out = {}
    for name, m in groups.items():
        k, t = int(agree[m].sum()), int(m.sum())
        lo, hi = wilson(k, t)
        out[name] = {"agree": k, "n": t, "rate": k / t if t else None, "wilson95": [lo, hi]}
    return out


def print_consistency(table: dict[str, dict[str, Any]], indent: str = "  ") -> None:
    for name, v in table.items():
        lo, hi = v["wilson95"]
        print(f"{indent}{name:<14} {v['agree']:4d}/{v['n']:4d} = {v['rate'] * 100:5.1f}%  (95% {lo * 100:.1f}~{hi * 100:.1f})")


# ---------- 보조 데이터 ----------

def load_oof(version: str) -> dict[str, dict]:
    path = ROOT / "reports" / "current" / version / "model_candidate_oof.csv"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8-sig", newline="") as h:
        return {r["requirement_uid"]: r for r in csv.DictReader(h)}


def load_v6() -> dict[str, str]:
    path = ROOT / "data" / "labels" / "label_dataset_v6.jsonl"
    if not path.exists():
        return {}
    return {j["requirement_uid"]: j["primary_action"]
            for j in (json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip())}


def load_anchor_uids() -> set[str]:
    path = ROOT / "data" / "anchors" / "anchor_pool_v2.jsonl"
    return {json.loads(l)["requirement_uid"] for l in path.read_text(encoding="utf-8").splitlines() if l.strip()}


# ---------- 2·3. 아웃라이어, 앵커 ----------

def outliers(sim: np.ndarray, quantile: float = 0.05) -> tuple[np.ndarray, np.ndarray, float]:
    """같은 문서 포함 최근접 유사도가 하위 `quantile`인 행. (마스크, 최근접 유사도, 임계값)"""
    n = sim.shape[0]
    nearest = np.sort(np.where(np.eye(n, dtype=bool), -np.inf, sim), axis=1)[:, -1]
    cut = float(np.quantile(nearest, quantile))
    return nearest <= cut, nearest, cut


def anchor_coverage(emb: np.ndarray, rows: Sequence[dict[str, Any]], anchors: set[str], k: int = K_CLUSTERS, seed: int = 42):
    cl = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(emb).labels_
    is_anchor = np.array([r["requirement_uid"] in anchors for r in rows])
    covered = set(cl[is_anchor])
    uncovered = ~np.isin(cl, list(covered))
    doc = np.array([r["document_id"] for r in rows])
    lab = np.array([r["primary_action"] for r in rows])
    mixed = 0
    for c in range(k):
        m = cl == c
        if len(set(doc[m])) >= 2 and collections.Counter(lab[m]).most_common(1)[0][1] / m.sum() < 0.6:
            mixed += int(m.sum())
    return cl, covered, uncovered, mixed


def main() -> None:
    rows, meta = load_label_dataset()
    version = meta["dataset_version"].rsplit("_", 1)[-1]
    out_dir = ROOT / "reports" / "current" / version
    n = len(rows)
    doc = np.array([r["document_id"] for r in rows])
    lab = np.array([r["primary_action"] for r in rows])
    oof, v6, anchors = load_oof(version), load_v6(), load_anchor_uids()
    groups = group_masks(rows, oof, v6)

    emb = e5_matrix(rows, version)
    sim_e = emb @ emb.T
    nn, nn_sim, major, agree = cross_document_neighbors(sim_e, doc, lab)
    table_e = consistency_table(agree, groups)
    print(f"1. 다른 문서 최근접 {K_NEIGHBORS}건 다수결이 자기 라벨과 일치하는 비율 ({version}, {n}건)")
    print_consistency(table_e)
    knn_e = float(f1_score(lab, major, labels=LABELS, average="macro"))
    print(f"  참고: 이 다수결을 예측으로 쓰면 macro F1 {knn_e:.3f} (학습 없음, 문서 간 kNN)")

    sim_t = tfidf_similarity(rows)
    nn_t, _, major_t, agree_t = cross_document_neighbors(sim_t, doc, lab)
    overlap = float(np.mean([len(set(a) & set(b)) / K_NEIGHBORS for a, b in zip(nn, nn_t)]))
    table_t = consistency_table(agree_t, groups)
    print(f"  대조군 — TF-IDF word+char 이웃 (E5와 이웃 겹침 {overlap * 100:.1f}%)")
    print_consistency(table_t, "    ")
    knn_t = float(f1_score(lab, major_t, labels=LABELS, average="macro"))
    print(f"    kNN 다수결 macro F1 {knn_t:.3f}")

    mask, nearest, cut = outliers(sim_e)
    print(f"\n2. 아웃라이어 — 최근접 유사도 하위 5% (임계 {cut:.3f}, {int(mask.sum())}건)")
    hangul = np.array([len(re.findall(r"[가-힣]", r["raw_requirement_text"])) / max(1, len(r["raw_requirement_text"])) for r in rows])
    length = np.array([len(r["raw_requirement_text"]) for r in rows])
    print(f"  한글 비율 중앙값  아웃라이어 {np.median(hangul[mask]):.2f} / 나머지 {np.median(hangul[~mask]):.2f}")
    print(f"  원문 길이 중앙값  아웃라이어 {np.median(length[mask]):.0f} / 나머지 {np.median(length[~mask]):.0f}")
    print("  문서별:", dict(collections.Counter(doc[mask].tolist()).most_common()))
    outlier_err = None
    if "경계 혼동" in groups:
        wrong = groups["경계 밖 오답"] | groups["경계 혼동"]
        evaluated = wrong | groups["모델이 맞힌 건"]
        e_out, n_out = int(wrong[mask].sum()), int(evaluated[mask].sum())
        e_in, n_in = int(wrong[~mask].sum()), int(evaluated[~mask].sum())
        outlier_err = {"outlier": [e_out, n_out], "rest": [e_in, n_in]}
        print(f"  모델 오답률  아웃라이어 {e_out}/{n_out} = {e_out / max(1, n_out) * 100:.1f}%  / 나머지 {e_in}/{n_in} = {e_in / max(1, n_in) * 100:.1f}%")
    top = []
    print("  가장 동떨어진 15건:")
    for i in np.argsort(nearest)[:15]:
        snippet = re.sub(r"\s+", " ", rows[i]["raw_requirement_text"])[:70]
        top.append({"uid": rows[i]["requirement_uid"], "nearest_sim": float(nearest[i]), "label": lab[i], "name": rows[i]["requirement_name"]})
        print(f"    {nearest[i]:.3f}  {rows[i]['requirement_uid']:<42} {lab[i]:<8} {rows[i]['requirement_name'][:22]:<22} | {snippet}")

    cl, covered, uncovered, mixed = anchor_coverage(emb, rows, anchors)
    print(f"\n3. 앵커 {len(anchors)}건이 덮는 군집 (k={K_CLUSTERS})")
    print(f"  앵커가 있는 군집 {len(covered)}/{K_CLUSTERS}, 앵커 없는 군집의 요구사항 {int(uncovered.sum())}건 ({uncovered.mean() * 100:.1f}%)")
    sizes = collections.Counter(cl.tolist())
    for size, c in sorted(((sizes[c], c) for c in range(K_CLUSTERS) if c not in covered), reverse=True)[:5]:
        names = collections.Counter(rows[i]["requirement_name"] for i in np.where(cl == c)[0]).most_common(3)
        print(f"    군집 {c:2d} ({size}건, 앵커 0): " + ", ".join(f"{k}×{v}" for k, v in names))
    print(f"  군집 {K_CLUSTERS}개 중 다수 라벨 점유율 60% 미만인 군집의 요구사항 {mixed}건 ({mixed / n * 100:.1f}%)")

    result = {
        "dataset": version, "rows": n, "k_neighbors": K_NEIGHBORS, "k_clusters": K_CLUSTERS,
        "cross_document_label_consistency": table_e, "knn_macro_f1": knn_e,
        "cross_document_label_consistency_tfidf": table_t, "knn_macro_f1_tfidf": knn_t,
        "neighbor_overlap_e5_tfidf": overlap,
        "outliers": {"threshold": cut, "count": int(mask.sum()), "error_rate": outlier_err, "top15": top},
        "anchors": {"clusters_covered": len(covered), "uncovered_rows": int(uncovered.sum()), "mixed_cluster_rows": mixed},
    }
    out = out_dir / "cluster_diagnostics.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"\n저장: {out}")


if __name__ == "__main__":
    sys.exit(main())
