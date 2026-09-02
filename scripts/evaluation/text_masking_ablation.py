#!/usr/bin/env python3
"""양식 신호를 하나씩 가려보는 전처리 ablation.

설명 보고서 감사에서 모델이 근거로 쓰는 상위 문구에 요구사항의 내용이 아니라 **작성
양식**을 가리키는 표현이 올라왔다. 이 스크립트는 그 표현을 규칙 하나씩 지운 입력으로
같은 LODO를 다시 돌려, 지금 점수가 내용에서 나오는지 양식에서 나오는지를 가른다.

점수를 올리려는 실험이 아니다. 떨어지는 폭이 곧 답이다 — 작으면 내용으로도 잡힌다는
뜻이고, 크면 새 발주처가 표기를 바꿀 때 무너진다는 뜻이다.

효과와 잡음은 평균 차이만으로 가르지 않는다. 문서가 10개뿐이라 fold 분산이 크므로
**10 fold 중 우세 fold 수**를 함께 본다(결정 2026-08-23의 판정 기준).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path
from typing import Any, Sequence

from scripts.data.preprocess_text import MASKS
from scripts.evaluation.baselines import (
    CHAR_BALANCED,
    DEFAULT_THRESHOLD,
    REVIEW_LABEL,
    WORD_BALANCED,
    WORD_CHAR_BALANCED,
    FoldResult,
    ModelSpec,
    evaluate_fold,
)
from scripts.evaluation.folds import _repeat_flags, make_lodo_folds
from scripts.labeling.label_dataset import (
    DATASET_VERSION_ENV,
    DEFAULT_DATASET_KEY,
    TEXT_MASK_ENV,
    load_label_dataset,
)

ROOT = Path(__file__).resolve().parents[2]
BASELINE = "none"
SPECS = (WORD_CHAR_BALANCED, CHAR_BALANCED, WORD_BALANCED)

MASK_LABELS = {
    "none": "원문 (v4)",
    "subject": "R1 주체 표기 → <주체>",
    "ending": "R2 서술 어미 → 어간",
    "josa": "R3 조사 제거",
}


def baseline_repeat_flags(rows: Sequence[dict[str, Any]]) -> dict[int, dict[str, bool]]:
    """반복 문구 판정을 **원문 기준으로 한 번만** 계산한다.

    이 값은 점수가 아니라 결정 34의 세 갈래 보고에만 쓰인다. 문서 간 문구 반복은
    원본 코퍼스의 성질이므로 마스킹마다 다시 재면 의미가 달라지고, 매번 다시 재면
    fold마다 큰 문자 n-gram 행렬을 새로 만들어 메모리도 감당하지 못한다.
    """
    return {fold.index: _repeat_flags(fold, rows, DEFAULT_THRESHOLD) for fold in make_lodo_folds(rows)}


def run_arm(
    rows: Sequence[dict[str, Any]],
    spec: ModelSpec,
    mask: str,
    repeat_flags: dict[int, dict[str, bool]],
) -> list[FoldResult]:
    """마스킹 규칙 하나를 켜고 fold 전체를 돌린다.

    `get_model_text`가 호출 시점에 환경변수를 읽으므로, 여기서 켜고 끄면 학습 입력과
    평가 입력이 함께 바뀐다.
    """
    previous = os.environ.get(TEXT_MASK_ENV)
    os.environ[TEXT_MASK_ENV] = mask
    try:
        return [
            evaluate_fold(fold, rows, spec, repeat_flags=repeat_flags[fold.index])
            for fold in make_lodo_folds(rows)
        ]
    finally:
        if previous is None:
            os.environ.pop(TEXT_MASK_ENV, None)
        else:
            os.environ[TEXT_MASK_ENV] = previous


def compare(baseline: Sequence[FoldResult], variant: Sequence[FoldResult]) -> dict[str, Any]:
    """기준선 대비 fold별 차이와 우세 fold 수를 센다."""
    def pull(results, key):
        return [
            result.per_class_f1[REVIEW_LABEL] if key == "review_f1" else getattr(result, key)
            for result in results
        ]

    summary: dict[str, Any] = {}
    for key in ("macro_f1", "accuracy", "review_recall", "review_precision", "review_f1"):
        base, var = pull(baseline, key), pull(variant, key)
        summary[key] = {
            "baseline": statistics.fmean(base),
            "variant": statistics.fmean(var),
            "difference": statistics.fmean(var) - statistics.fmean(base),
        }
    macro_diff = [v - b for b, v in zip(pull(baseline, "macro_f1"), pull(variant, "macro_f1"))]
    summary["fold_wins"] = sum(1 for d in macro_diff if d > 0)
    summary["fold_count"] = len(macro_diff)
    summary["fold_macro_f1_difference"] = macro_diff
    return summary


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# {report['dataset_version']} 양식 신호 마스킹 ablation",
        "",
        "- 평가: 동결 앵커 100건을 제외한 924건, 학습 8 / 검증 1 / 평가 1 문서 LODO 10-fold",
        "- 규칙은 v4 원문 위에 **하나씩만** 적용한다. 누적하지 않는다.",
        "- 값은 fold 단순 평균이며, 우세는 10 fold 중 기준선을 넘은 fold 수다.",
        f"- 명령: `$env:{DATASET_VERSION_ENV}='{report['dataset_version']}'; "
        "python -m scripts.evaluation.text_masking_ablation`",
        "",
        "## 적용한 규칙",
        "",
        "| 규칙 | 정의 |",
        "|---|---|",
        "| R1 주체 표기 | 요구를 *누가* 하는지 가리키는 말을 `<주체>`로 치환. 어절 안에 묻힌 `건설공사`는 건드리지 않는다 |",
        "| R2 서술 어미 | 의무·서술 어미를 어간으로. 남는 어간이 두 글자 미만이면 그대로 둔다 |",
        "| R3 조사 제거 | 명사 뒤 조사를 떼어 같은 어휘의 표기 변이를 합친다 |",
        "",
    ]
    for spec_name, arms in report["specs"].items():
        lines += [
            f"## {spec_name}",
            "",
            "| 입력 | macro F1 | 차이 | 우세 | 계약 recall | 차이 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        baseline = arms[BASELINE]
        lines.append(
            f"| {MASK_LABELS[BASELINE]} | {baseline['macro_f1']:.3f} | — | — | "
            f"{baseline['review_recall']:.3f} | — |"
        )
        for mask, arm in arms.items():
            if mask == BASELINE:
                continue
            diff = arm["comparison"]
            lines.append(
                f"| {MASK_LABELS[mask]} | {arm['macro_f1']:.3f} | "
                f"{diff['macro_f1']['difference']:+.3f} | "
                f"{diff['fold_wins']}/{diff['fold_count']} | "
                f"{arm['review_recall']:.3f} | {diff['review_recall']['difference']:+.3f} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    version = os.getenv(DATASET_VERSION_ENV, DEFAULT_DATASET_KEY)
    default_dir = ROOT / "reports" / "current" / version
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--masks",
        nargs="+",
        default=["ending", "josa"],
        choices=sorted(MASKS),
        help="적용할 규칙. 기준선(none)은 항상 함께 돌린다.",
    )
    parser.add_argument("--output", type=Path, default=default_dir / "text_masking_ablation.md")
    parser.add_argument("--json", type=Path, default=default_dir / "text_masking_ablation.json")
    args = parser.parse_args()

    rows, _ = load_label_dataset()
    repeat_flags = baseline_repeat_flags(rows)
    report: dict[str, Any] = {"dataset_version": version, "specs": {}}

    for spec in SPECS:
        arms: dict[str, Any] = {}
        baseline = run_arm(rows, spec, BASELINE, repeat_flags)
        arms[BASELINE] = {
            "macro_f1": statistics.fmean(r.macro_f1 for r in baseline),
            "review_recall": statistics.fmean(r.review_recall for r in baseline),
        }
        for mask in args.masks:
            variant = run_arm(rows, spec, mask, repeat_flags)
            arms[mask] = {
                "macro_f1": statistics.fmean(r.macro_f1 for r in variant),
                "review_recall": statistics.fmean(r.review_recall for r in variant),
                "comparison": compare(baseline, variant),
            }
            change = arms[mask]["comparison"]["macro_f1"]["difference"]
            print(
                f"  {spec.name:<36} {mask:<8} macro F1 {arms[mask]['macro_f1']:.3f} "
                f"({change:+.3f}, 우세 {arms[mask]['comparison']['fold_wins']}/10)"
            )
        report["specs"][spec.name] = arms

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_markdown(report), encoding="utf-8")
    args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"저장: {args.output}")
    print(f"저장: {args.json}")


if __name__ == "__main__":
    main()
