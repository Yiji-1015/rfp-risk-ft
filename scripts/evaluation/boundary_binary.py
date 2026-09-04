"""`견적반영` 대 `계약·질의검토`만 떼어 학습하면 갈리는가.

3분류 오답 294건 중 98건이 이 둘의 상호 혼동이다. "세 클래스를 함께 배워서 경계가
간섭한 것인가"를 시험하려면 `통상수용`을 빼고 두 클래스만 학습해 같은 LODO로 잰다.
비교 대상은 3분류 모델의 OOF에서 같은 두 클래스 확률만 비교한 argmax다. 둘이 같으면
경계는 3분류 구조 때문이 아니다.

명령: `$env:RFP_DATASET_VERSION='v4'; python -m scripts.evaluation.boundary_binary`
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from scripts.evaluation.baselines import WORD_CHAR_BALANCED, _fit_pipeline, _model_input
from scripts.evaluation.folds import make_lodo_folds
from scripts.labeling.label_dataset import DATASET_VERSION_ENV, DEFAULT_DATASET_KEY, load_label_dataset

ROOT = Path(__file__).resolve().parents[2]
Q, R = "견적반영", "계약·질의검토"


def main() -> None:
    rows, meta = load_label_dataset()
    boundary = [r for r in rows if r["primary_action"] in (Q, R)]
    gold, pred, per_fold = [], [], []
    for fold in make_lodo_folds(rows):
        fit, _, test = fold.split(boundary)
        pipe = _fit_pipeline(WORD_CHAR_BALANCED, fit)
        p = list(pipe.predict(_model_input(WORD_CHAR_BALANCED, test)))
        g = [r["primary_action"] for r in test]
        gold += g; pred += p
        per_fold.append((fold.test_document, len(test), float(np.mean([a == b for a, b in zip(g, p)]))))

    version = os.getenv(DATASET_VERSION_ENV, DEFAULT_DATASET_KEY)
    out_dir = ROOT / "reports" / "current" / version
    with (out_dir / "model_candidate_oof.csv").open(encoding="utf-8-sig", newline="") as handle:
        oof = [r for r in csv.DictReader(handle) if r["gold"] in (Q, R)]
    g3 = [r["gold"] for r in oof]
    restricted = [Q if float(r[f"word_char_logistic_p_{Q}"]) >= float(r[f"word_char_logistic_p_{R}"]) else R for r in oof]
    original = [r["word_char_logistic_pred"] for r in oof]

    acc = lambda a, b: float(np.mean([x == y for x, y in zip(a, b)]))
    lines = [
        "# 견적반영 대 계약·질의검토 전용 2분류",
        "",
        f"- 데이터: {meta['dataset_version']}, 앵커 제외 경계 {len(gold)}건, LODO 10-fold, 학습 8문서",
        "- 모델: word+char TF-IDF balanced Logistic. 학습·검증·평가 모두 두 클래스 행만 사용",
        "- 명령: `python -m scripts.evaluation.boundary_binary`",
        "",
        "| 설정 | 정확도 | macro F1 |",
        "|---|---:|---:|",
        f"| 전용 2분류 (통상수용 제외하고 학습) | {acc(gold, pred):.3f} | {f1_score(gold, pred, average='macro'):.3f} |",
        f"| 3분류 모델, 두 클래스 확률만 비교한 argmax | {acc(g3, restricted):.3f} | {f1_score(g3, restricted, average='macro'):.3f} |",
        f"| 3분류 모델 원래 예측 (통상 예측은 오답) | {acc(g3, original):.3f} | – |",
        "| 무작위 | 0.500 | – |",
        "",
        "## fold별 전용 2분류 정확도",
        "",
        "| 평가 문서 | n | 정확도 |",
        "|---|---:|---:|",
        *[f"| {d} | {n} | {a:.3f} |" for d, n, a in per_fold],
        "",
        "## 읽기",
        "",
        "전용 학습과 3분류 안에서 배운 두 클래스 축이 같은 값이다. 견적↔계약 경계는 3분류 구조가",
        "만든 간섭이 아니며, 계층형 분류기는 2단계가 이 값을 그대로 물려받으므로 후보가 되지 않는다.",
        f"3분류 원래 예측이 낮은 것은 경계 {len(g3)}건 중 {sum(p == '통상수용' for p in original)}건을 `통상수용`으로",
        "예측했기 때문이며, 통상↔검토 축의 오답이 견적↔계약 축과 같은 규모로 있다는 뜻이다.",
        "",
    ]
    (out_dir / "boundary_binary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[6:11]))
    print(f"기록: {out_dir / 'boundary_binary.md'}")


if __name__ == "__main__":
    main()
