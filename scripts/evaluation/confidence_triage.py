"""확신도로 자동 처리와 사람 검토를 가르면 자동 구간의 정확도가 얼마나 오르는지 잰다.

경계 오답은 모델이 오독한 것이 아니라 망설인 것이다 — 1·2위 확률차 중앙값이 오답 0.129,
정답 0.265(2026-09-01). 그 확률차를 문턱으로 삼아 낮은 건을 사람에게 넘기면, 나머지에서
점수가 어디까지 오르는지를 커버리지별로 낸다. 라벨을 보지 않고 확률 순위로만 가르므로
배포 시점에도 같은 방식으로 쓸 수 있다.

명령: `$env:RFP_DATASET_VERSION='v4'; python -m scripts.evaluation.confidence_triage`
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, recall_score

from scripts.labeling.label_dataset import DATASET_VERSION_ENV, DEFAULT_DATASET_KEY

ROOT = Path(__file__).resolve().parents[2]
LABELS = ("통상수용", "견적반영", "계약·질의검토")
BOUNDARY = frozenset({"견적반영", "계약·질의검토"})
MODELS = {"word+char TF-IDF": "word_char_logistic", "soft voting (TF-IDF 3종)": "soft_vote"}
COVERAGES = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5)


def triage(rows, prefix):
    gold = np.array([r["gold"] for r in rows])
    probs = np.array([[float(r[f"{prefix}_p_{l}"]) for l in LABELS] for r in rows])
    pred = np.array(LABELS)[probs.argmax(1)]
    top2 = np.sort(probs, axis=1)[:, ::-1]
    margin = top2[:, 0] - top2[:, 1]
    order = np.argsort(-margin)  # 확신 높은 것부터
    out = []
    for cov in COVERAGES:
        keep = np.zeros(len(rows), dtype=bool)
        keep[order[: int(round(cov * len(rows)))]] = True
        defer = ~keep
        boundary_deferred = int(
            sum(g != p and {g, p} == BOUNDARY for g, p in zip(gold[defer], pred[defer]))
        )
        out.append(
            dict(
                coverage=cov,
                kept=int(keep.sum()),
                deferred=int(defer.sum()),
                margin_threshold=float(margin[order[int(round(cov * len(rows))) - 1]]),
                accuracy=float((pred[keep] == gold[keep]).mean()),
                macro_f1=float(f1_score(gold[keep], pred[keep], labels=LABELS, average="macro", zero_division=0)),
                review_recall=float(recall_score(gold[keep], pred[keep], labels=[LABELS[2]], average="macro", zero_division=0)),
                deferred_accuracy=float((pred[defer] == gold[defer]).mean()) if defer.any() else None,
                boundary_errors_deferred=boundary_deferred,
            )
        )
    return out


def render(results):
    lines = []
    for name, table in results.items():
        lines += [f"## {name}", "",
                  "| 자동 처리 | 건수 | 검토로 넘김 | 확률차 문턱 | 자동 구간 정확도 | 자동 구간 macro F1 | 자동 구간 계약 recall | 넘긴 구간 정확도 | 넘긴 경계 오답 |",
                  "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for t in table:
            da = "–" if t["deferred_accuracy"] is None else f"{t['deferred_accuracy']:.3f}"
            lines.append(
                f"| {t['coverage']:.0%} | {t['kept']} | {t['deferred']} | {t['margin_threshold']:.3f} | "
                f"{t['accuracy']:.3f} | {t['macro_f1']:.3f} | {t['review_recall']:.3f} | {da} | {t['boundary_errors_deferred']} |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    version = os.getenv(DATASET_VERSION_ENV, DEFAULT_DATASET_KEY)
    out_dir = ROOT / "reports" / "current" / version
    with (out_dir / "model_candidate_oof.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    results = {name: triage(rows, prefix) for name, prefix in MODELS.items()}
    body = render(results)
    print(body)
    (out_dir / "confidence_triage.md").write_text(
        "# 확신도 선별 — 자동 처리 비율별 정확도\n\n"
        f"- 데이터: {version}, 앵커 제외 {len(rows)}건 통합 OOF (LODO 10-fold)\n"
        "- 확신도: 1위 확률 − 2위 확률. 라벨을 보지 않고 이 값의 순위로만 자동/검토를 가른다\n"
        "- 넘긴 경계 오답: 검토로 넘긴 건 중 견적↔계약 상호 혼동이었던 건수 (전체 98건)\n"
        "- 명령: `python -m scripts.evaluation.confidence_triage`\n\n" + body,
        encoding="utf-8")
    (out_dir / "confidence_triage.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"기록: {out_dir / 'confidence_triage.md'}")


if __name__ == "__main__":
    main()
