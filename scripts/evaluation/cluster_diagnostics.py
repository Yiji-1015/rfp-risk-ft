"""E5 임베딩 군집을 점수가 아니라 진단에 쓴다.

세 가지를 잰다. 어느 것도 모델을 학습하지 않는다.

1. 문서 간 라벨 일관성 — 뜻이 가까운 조항이 **다른 문서**에서 어떤 라벨을 받았나.
   각 요구사항의 다른 문서 최근접 이웃 5건 다수결이 자기 라벨과 맞는 비율. 모델이 맞힌
   건·틀린 건·경계 혼동·v5·v6 불일치별로 나눠 본다. 문장이 비슷한데 라벨이 갈리면 그
   차이는 문서 맥락에서 온다.
2. 아웃라이어 — 최근접 이웃과의 거리가 먼 상위 5%. 추출 결함 검수와 오답 겹침 확인.
3. 앵커 분포 — 동결 앵커 100건이 군집 몇 개를 덮는가(issues/008의 배경).

    RFP_DATASET_VERSION=v5 python -m scripts.evaluation.cluster_diagnostics
"""
from __future__ import annotations

import collections
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

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


def main() -> None:
    rows, meta = load_label_dataset()
    version = meta["dataset_version"].rsplit("_", 1)[-1]
    out_dir = ROOT / "reports" / "current" / version
    emb = load_cached_embeddings(ROOT / "data" / "processed" / f"multilingual-e5-small.{version}.npz", rows)
    if emb is None:
        sys.exit("임베딩 캐시가 없거나 데이터와 맞지 않습니다.")
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    n = len(rows)
    uid = [r["requirement_uid"] for r in rows]
    doc = np.array([r["document_id"] for r in rows])
    lab = np.array([r["primary_action"] for r in rows])

    # 보조 정보: 모델 OOF, v6 라벨, 앵커
    oof_path = out_dir / "model_candidate_oof.csv"
    oof = {}
    if oof_path.exists():
        with oof_path.open(encoding="utf-8-sig", newline="") as h:
            oof = {r["requirement_uid"]: r for r in csv.DictReader(h)}
    v6_path = ROOT / "data" / "labels" / "label_dataset_v6.jsonl"
    v6 = {}
    if v6_path.exists():
        v6 = {j["requirement_uid"]: j["primary_action"]
              for j in (json.loads(l) for l in v6_path.read_text(encoding="utf-8").splitlines() if l.strip())}
    anchors = {json.loads(l)["requirement_uid"]
               for l in (ROOT / "data" / "anchors" / "anchor_pool_v2.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}

    # ---------- 1. 문서 간 최근접 이웃 라벨 일관성 ----------
    sim = emb @ emb.T
    same_doc = doc[:, None] == doc[None, :]
    sim_x = np.where(same_doc, -np.inf, sim)          # 다른 문서만
    nn = np.argsort(-sim_x, axis=1)[:, :K_NEIGHBORS]
    nn_sim = np.take_along_axis(sim_x, nn, axis=1)
    nn_major = np.array([collections.Counter(lab[i]).most_common(1)[0][0] for i in nn])
    agree = nn_major == lab

    groups = {"전체": np.ones(n, bool)}
    if oof:
        pred = np.array([oof[u]["word_char_logistic_pred"] if u in oof else "" for u in uid])
        evaluated = pred != ""
        correct = evaluated & (pred == lab)
        wrong = evaluated & (pred != lab)
        boundary = wrong & np.array([{g, p} == BOUNDARY for g, p in zip(lab, pred)])
        groups.update({"모델이 맞힌 건": correct, "경계 밖 오답": wrong & ~boundary, "경계 혼동": boundary})
    if v6:
        unstable = np.array([v6.get(u) != l for u, l in zip(uid, lab)])
        groups.update({"v5·v6 일치": ~unstable, "v5·v6 불일치": unstable})
    for l in LABELS:
        groups[f"라벨={l}"] = lab == l

    print(f"1. 다른 문서 최근접 {K_NEIGHBORS}건 다수결이 자기 라벨과 일치하는 비율 ({version}, {n}건)")
    consistency = {}
    for name, m in groups.items():
        k, t = int(agree[m].sum()), int(m.sum())
        lo, hi = wilson(k, t)
        consistency[name] = {"agree": k, "n": t, "rate": k / t if t else None, "wilson95": [lo, hi]}
        print(f"  {name:<14} {k:4d}/{t:4d} = {k / t * 100:5.1f}%  (95% {lo * 100:.1f}~{hi * 100:.1f})")
    knn_macro = None
    try:
        from sklearn.metrics import f1_score
        knn_macro = float(f1_score(lab, nn_major, labels=LABELS, average="macro"))
        print(f"  참고: 이 다수결을 예측으로 쓰면 macro F1 {knn_macro:.3f} (학습 없음, 문서 간 kNN)")
    except Exception:
        pass

    # ---------- 2. 아웃라이어 ----------
    nearest = nn_sim[:, 0]                              # 다른 문서 최근접 유사도
    any_nearest = np.sort(np.where(np.eye(n, dtype=bool), -np.inf, sim), axis=1)[:, -1]  # 같은 문서 포함
    cut = np.quantile(any_nearest, 0.05)
    outlier = any_nearest <= cut
    print(f"\n2. 아웃라이어 — 최근접 유사도 하위 5% (임계 {cut:.3f}, {int(outlier.sum())}건)")
    hangul = np.array([len(re.findall(r"[가-힣]", r["raw_requirement_text"])) / max(1, len(r["raw_requirement_text"])) for r in rows])
    length = np.array([len(r["raw_requirement_text"]) for r in rows])
    print(f"  한글 비율 중앙값  아웃라이어 {np.median(hangul[outlier]):.2f} / 나머지 {np.median(hangul[~outlier]):.2f}")
    print(f"  원문 길이 중앙값  아웃라이어 {np.median(length[outlier]):.0f} / 나머지 {np.median(length[~outlier]):.0f}")
    print("  문서별:", dict(collections.Counter(doc[outlier].tolist()).most_common()))
    print("  라벨별:", {l: int((lab[outlier] == l).sum()) for l in LABELS})
    outlier_err = None
    if oof:
        e_out, n_out = int(wrong[outlier].sum()), int(evaluated[outlier].sum())
        e_in, n_in = int(wrong[~outlier].sum()), int(evaluated[~outlier].sum())
        outlier_err = {"outlier": [e_out, n_out], "rest": [e_in, n_in]}
        print(f"  모델 오답률  아웃라이어 {e_out}/{n_out} = {e_out / max(1, n_out) * 100:.1f}%  / 나머지 {e_in}/{n_in} = {e_in / max(1, n_in) * 100:.1f}%")
    order = np.argsort(any_nearest)
    print("  가장 동떨어진 15건:")
    top = []
    for i in order[:15]:
        snippet = re.sub(r"\s+", " ", rows[i]["raw_requirement_text"])[:70]
        top.append({"uid": uid[i], "nearest_sim": float(any_nearest[i]), "label": lab[i], "name": rows[i]["requirement_name"]})
        print(f"    {any_nearest[i]:.3f}  {uid[i]:<42} {lab[i]:<8} {rows[i]['requirement_name'][:22]:<22} | {snippet}")

    # ---------- 3. 앵커 분포 ----------
    km = KMeans(n_clusters=K_CLUSTERS, n_init=10, random_state=42).fit(emb)
    cl = km.labels_
    is_anchor = np.array([u in anchors for u in uid])
    anchor_clusters = set(cl[is_anchor])
    uncovered = ~np.isin(cl, list(anchor_clusters))
    print(f"\n3. 앵커 {int(is_anchor.sum())}건이 덮는 군집 (k={K_CLUSTERS})")
    print(f"  앵커가 있는 군집 {len(anchor_clusters)}/{K_CLUSTERS}, 앵커 없는 군집의 요구사항 {int(uncovered.sum())}건 ({uncovered.mean() * 100:.1f}%)")
    sizes = collections.Counter(cl)
    big_uncovered = sorted((sizes[c], c) for c in range(K_CLUSTERS) if c not in anchor_clusters)[::-1][:5]
    for size, c in big_uncovered:
        names = collections.Counter(rows[i]["requirement_name"] for i in np.where(cl == c)[0]).most_common(3)
        print(f"    군집 {c:2d} ({size}건, 앵커 0): " + ", ".join(f"{k}×{v}" for k, v in names))
    # 군집 안 문서 간 라벨 혼재
    mixed = 0
    for c in range(K_CLUSTERS):
        m = cl == c
        if len(set(doc[m])) >= 2 and collections.Counter(lab[m]).most_common(1)[0][1] / m.sum() < 0.6:
            mixed += int(m.sum())
    print(f"  군집 {K_CLUSTERS}개 중 다수 라벨 점유율 60% 미만인 군집의 요구사항 {mixed}건 ({mixed / n * 100:.1f}%)")

    result = {
        "dataset": version, "rows": n, "k_neighbors": K_NEIGHBORS, "k_clusters": K_CLUSTERS,
        "cross_document_label_consistency": consistency, "knn_macro_f1": knn_macro,
        "outliers": {"threshold": float(cut), "count": int(outlier.sum()), "error_rate": outlier_err, "top15": top},
        "anchors": {"clusters_covered": len(anchor_clusters), "uncovered_rows": int(uncovered.sum()), "mixed_cluster_rows": mixed},
    }
    out = out_dir / "cluster_diagnostics.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
