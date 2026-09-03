#!/usr/bin/env python3
"""파인튜닝 실행을 요약하고, TF-IDF 후보와 섞었을 때를 본다.

파인튜닝은 단독 점수로는 TF-IDF와 갈리지 않는다(fold 평균 0.593 대 0.614, seed 범위
0.036). 그런데 **틀리는 지점이 다르다.** 두 계열이 함께 틀리는 것은 오답의 61%뿐이고
한쪽만 틀린 것이 216건이라, 다수결로 묶으면 그중 일부를 건진다.

조합을 고르는 일 자체가 채점표를 보는 행위이므로, 문서 하나를 빼고 나머지 아홉으로
조합을 고른 뒤 뺀 문서에서 평가하는 **중첩 선택**을 함께 낸다. 두 값이 같으면 선택이
특정 문서에 기대지 않는다는 뜻이다.

멤버는 학습하지 않는다. 이미 저장된 OOF 예측만 투표시킨다.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import statistics
from pathlib import Path
from typing import Any, Sequence

from sklearn.metrics import f1_score

from scripts.labeling.label_dataset import DATASET_VERSION_ENV, DEFAULT_DATASET_KEY

ROOT = Path(__file__).resolve().parents[2]
LABELS = ("통상수용", "견적반영", "계약·질의검토")
BOUNDARY = frozenset({"견적반영", "계약·질의검토"})

# TF-IDF 계열 후보는 `model_candidate_oof.csv`의 열 이름으로 온다.
TFIDF_MEMBERS = {
    "wc": "word_char_logistic_pred",
    "ch": "char_logistic_pred",
    "e5": "tfidf_e5_hybrid_pred",
    "sv": "soft_vote_pred",
}

# 후보 조합. 점수를 보고 늘리지 않는다 — 계열을 섞는다는 원칙과 파인튜닝 멤버 수를
# 바꿔보는 것까지가 사전에 정한 범위다.
CANDIDATE_COMBOS = (
    ("wc",), ("ftL",), ("ft7",), ("sv",),
    ("wc", "ft7", "ftL"), ("wc", "e5", "ftL"), ("e5", "ft7", "ftL"),
    ("sv", "ft7", "ftL"), ("wc", "e5", "ft7"), ("wc", "ch", "ftL"),
    ("wc", "ftL", "ftM"), ("ch", "e5", "ftL"), ("wc", "e5", "ftM"),
    ("e5", "ftL", "ftM"), ("wc", "ch", "e5"), ("wc", "ch", "e5", "ft7", "ftL"),
    # 채택 조합 wc+ft+ftL의 base 멤버를 seed별로, 그리고 aux 멤버로 바꾼 것. aux가 seed 편차를
    # 줄였으니(0.038 → 0.008) 앙상블 범위도 좁아지는지 본다. 점수를 보고 고른 조합이 아니다.
    ("wc", "ft42", "ftL"), ("wc", "ft13", "ftL"),
    ("wc", "ftA42", "ftL"), ("wc", "ftA7", "ftL"), ("wc", "ftA13", "ftL"),
    ("wc", "ftA42", "ftA7", "ftA13"),
    # 두 멤버 모두 aux. seed를 어떻게 뽑아도 같은 값이 나오는지가 질문이다.
    ("wc", "ftA42", "ftAL42"), ("wc", "ftA7", "ftAL7"), ("wc", "ftA13", "ftAL13"),
)


def load_members(runs_path: Path, oof_path: Path) -> tuple[dict[str, dict[str, str]], dict, dict]:
    """멤버별 `uid -> 예측` 표와 정답·문서를 읽는다."""
    with oof_path.open(encoding="utf-8-sig", newline="") as handle:
        oof = {row["requirement_uid"]: row for row in csv.DictReader(handle)}

    runs = [json.loads(line) for line in runs_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    members: dict[str, dict[str, str]] = {}
    labels: dict[str, str] = {}
    for run in runs:
        config = run["config"]
        if config["fold"] != -1 or not run["results"][0].get("predictions"):
            continue
        # 2분류·다른 계열은 3분류 투표에 못 섞는다. 태그가 겹치면 뒤 실행이 앞을 덮어쓰므로
        # seed와 변형(마스킹·aux)을 태그에 넣는다. 기존 태그(ft7·ftL·ftM·ftS)는 그대로다.
        if config.get("binary") or "roberta" not in config["model"]:
            continue
        size = config["model"].split("-")[-1]
        seed = config["seed"]
        tag = {"small": "ftS", "large": "ftL" if seed == 42 else f"ftL{seed}"}.get(size, f"ft{seed}")
        if config["mask"]:
            tag = "ftM"
        if config.get("aux"):
            tag = f"ftAL{seed}" if size == "large" else f"ftA{seed}"
        members[tag] = {
            item["requirement_uid"]: item["pred"]
            for fold in run["results"]
            for item in fold["predictions"]
        }
        labels[tag] = f"{config['model'].split('/')[-1]}{' +마스킹' if config['mask'] else ''} seed{config['seed']}"

    uids = [uid for uid in oof if all(uid in table for table in members.values())]
    for tag, column in TFIDF_MEMBERS.items():
        members[tag] = {uid: oof[uid][column] for uid in uids}
        labels[tag] = column.replace("_pred", "")
    return members, {uid: oof[uid]["gold"] for uid in uids}, {uid: oof[uid]["test_document"] for uid in uids}


def vote(members: dict[str, dict[str, str]], names: Sequence[str], uids: Sequence[str]) -> list[str]:
    """다수결. 셋이 모두 다르면 첫 멤버를 따른다(순서가 곧 우선순위다)."""
    tables = [members[name] for name in names]
    result = []
    for uid in uids:
        counts = collections.Counter(table[uid] for table in tables).most_common()
        result.append(counts[0][0] if counts[0][1] > 1 else tables[0][uid])
    return result


def macro(gold: Sequence[str], pred: Sequence[str]) -> float:
    return float(f1_score(gold, pred, labels=list(LABELS), average="macro", zero_division=0))


def describe(gold: Sequence[str], pred: Sequence[str], documents: Sequence[str]) -> dict[str, Any]:
    """통합 OOF와 fold 평균, 라벨별 F1, 경계 혼동 수를 함께 낸다."""
    by_document: dict[str, tuple[list, list]] = collections.defaultdict(lambda: ([], []))
    for g, p, d in zip(gold, pred, documents):
        by_document[d][0].append(g)
        by_document[d][1].append(p)
    wrong = [(g, p) for g, p in zip(gold, pred) if g != p]
    return {
        "pooled_macro_f1": macro(gold, pred),
        "fold_mean_macro_f1": statistics.fmean(macro(g, p) for g, p in by_document.values()),
        "per_label_f1": dict(
            zip(LABELS, (float(v) for v in f1_score(gold, pred, labels=list(LABELS), average=None, zero_division=0)))
        ),
        "errors": len(wrong),
        "boundary_errors": sum(1 for g, p in wrong if {g, p} == BOUNDARY),
    }


def overlap(members, gold, uids, left: str, right: str) -> dict[str, Any]:
    """두 멤버의 오답이 얼마나 겹치는지. 조합을 고르기 전에 재는 값이다."""
    a = {uid for uid in uids if members[left][uid] != gold[uid]}
    b = {uid for uid in uids if members[right][uid] != gold[uid]}
    correct = sum(1 for uid in uids if gold[uid] in (members[left][uid], members[right][uid]))
    return {
        "left_errors": len(a),
        "right_errors": len(b),
        "both_wrong": len(a & b),
        "one_wrong": len(a ^ b),
        "oracle_accuracy": correct / len(uids),
    }


def nested_selection(members, gold, documents, uids, combos=CANDIDATE_COMBOS) -> dict[str, Any]:
    """문서 하나를 빼고 아홉으로 조합을 고른 뒤, 뺀 문서에서 평가한다.

    후보 목록을 인자로 받는다. 무엇을 후보로 두었는지가 결과의 일부이므로 숨기지 않는다.
    """
    picks: collections.Counter = collections.Counter()
    held: list[tuple[str, str]] = []
    for document in sorted(set(documents.values())):
        inner = [uid for uid in uids if documents[uid] != document]
        outer = [uid for uid in uids if documents[uid] == document]
        best = max(
            [c for c in combos if all(tag in members for tag in c)],
            key=lambda combo: macro([gold[uid] for uid in inner], vote(members, combo, inner)),
        )
        picks["+".join(best)] += 1
        held.extend(zip(outer, vote(members, best, outer)))
    return {
        "macro_f1": macro([gold[uid] for uid, _ in held], [p for _, p in held]),
        "selected": dict(picks),
    }


def render(report: dict[str, Any]) -> str:
    lines = [
        f"# {report['dataset_version']} 파인튜닝 실행과 앙상블",
        "",
        "- 평가: 동결 앵커 100건을 제외한 924건, 학습 8 / 검증 1 / 평가 1 문서 LODO 10-fold",
        "- 앙상블 멤버는 새로 학습하지 않는다. 저장된 OOF 예측을 다수결로 묶는다.",
        f"- 명령: `$env:{DATASET_VERSION_ENV}='{report['dataset_version']}'; "
        "python -m scripts.evaluation.finetune_ensemble`",
        "",
        "## 단일 모델",
        "",
        "| 설정 | 통합 OOF | fold 평균 | 통상수용 | 견적반영 | 계약·질의검토 | 오답 | 경계 혼동 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, value in report["singles"].items():
        per = value["per_label_f1"]
        lines.append(
            f"| {name} | {value['pooled_macro_f1']:.3f} | {value['fold_mean_macro_f1']:.3f} | "
            + " | ".join(f"{per[label]:.3f}" for label in LABELS)
            + f" | {value['errors']} | {value['boundary_errors']} |"
        )

    o = report["overlap"]
    lines += [
        "",
        "## 오답이 겹치는가 — 조합을 고르기 전에 잰 값",
        "",
        f"- word+char 오답 {o['left_errors']}건, 파인튜닝 오답 {o['right_errors']}건",
        f"- **둘 다 틀린 것 {o['both_wrong']}건**, 한쪽만 틀린 것 {o['one_wrong']}건",
        f"- 둘 중 맞은 쪽을 고를 수 있다면 정확도 {o['oracle_accuracy']:.3f}",
        "",
        "같은 계열끼리 묶은 기존 soft voting이 오르지 않았던 이유가 여기 있다. 세 후보가 모두",
        "희소 TF-IDF라 **같은 것을 틀렸다.** 파인튜닝은 계열이 달라 틀리는 자리가 다르다.",
        "",
        "## 앙상블",
        "",
        "| 조합 | 통합 OOF | fold 평균 | 통상수용 | 견적반영 | 계약·질의검토 | 오답 | 경계 혼동 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, value in report["ensembles"].items():
        per = value["per_label_f1"]
        lines.append(
            f"| {name} | {value['pooled_macro_f1']:.3f} | {value['fold_mean_macro_f1']:.3f} | "
            + " | ".join(f"{per[label]:.3f}" for label in LABELS)
            + f" | {value['errors']} | {value['boundary_errors']} |"
        )

    n = report["nested"]
    lines += [
        "",
        "## 조합 선택이 채점표에 기대는가",
        "",
        f"문서 하나를 빼고 나머지 아홉으로 조합을 고른 뒤 뺀 문서에서 평가하면 **{n['macro_f1']:.3f}**이다.",
        f"열 번의 선택 결과: {n['selected']}",
        "",
        "선택이 특정 문서에 기대고 있었다면 이 값이 내려앉는다. 같으면 조합이 안정적이라는 뜻이다.",
        "",
        "## 읽을 때 주의",
        "",
        "- 파인튜닝 멤버의 seed를 바꾸면 앙상블 점수도 바뀐다. 하나의 값이 아니라 **범위**로 읽는다.",
        "- 앙상블은 오답을 줄이지만 **경계 혼동은 줄이지 않는다.** 위 표의 마지막 두 열을 함께 본다.",
        "- 확정은 새 RFP에서 한다. 여기 모든 값은 같은 924건에서 나왔다.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    version = os.getenv(DATASET_VERSION_ENV, DEFAULT_DATASET_KEY)
    default_dir = ROOT / "reports" / "current" / version
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=default_dir / "finetune_runs.jsonl")
    parser.add_argument("--oof", type=Path, default=default_dir / "model_candidate_oof.csv")
    parser.add_argument("--output", type=Path, default=default_dir / "finetune_results.md")
    parser.add_argument("--json", type=Path, default=default_dir / "finetune_results.json")
    args = parser.parse_args()

    members, gold, documents = load_members(args.runs, args.oof)
    uids = sorted(gold)
    g = [gold[uid] for uid in uids]
    d = [documents[uid] for uid in uids]

    singles = {
        name: describe(g, [members[tag][uid] for uid in uids], d)
        for tag, name in (
            ("wc", "word+char TF-IDF"), ("ch", "char TF-IDF"), ("e5", "TF-IDF+E5"),
            ("sv", "soft voting (TF-IDF 3종)"), ("ftS", "FT small"), ("ft42", "FT base seed42"),
            ("ft7", "FT base seed7"), ("ft13", "FT base seed13"), ("ftL", "FT large"),
            ("ftL7", "FT large seed7"), ("ftL13", "FT large seed13"), ("ftM", "FT base +마스킹"),
            ("ftA42", "FT base +aux seed42"), ("ftA7", "FT base +aux seed7"),
            ("ftA13", "FT base +aux seed13"), ("ftAL42", "FT large +aux seed42"),
            ("ftAL7", "FT large +aux seed7"), ("ftAL13", "FT large +aux seed13"),
        )
        if tag in members
    }
    ensembles = {
        "+".join(combo): describe(g, vote(members, combo, uids), d)
        for combo in CANDIDATE_COMBOS
        if len(combo) > 1 and all(tag in members for tag in combo)
    }
    report = {
        "dataset_version": version,
        "evaluated": len(uids),
        "singles": singles,
        "ensembles": ensembles,
        "overlap": overlap(members, gold, uids, "wc", "ft7"),
        "nested": nested_selection(members, gold, documents, uids),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(report), encoding="utf-8")
    args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{len(uids)}건 / 멤버 {len(members)}종")
    for name, value in sorted(ensembles.items(), key=lambda kv: -kv[1]["pooled_macro_f1"])[:5]:
        print(f"  {name:<26} 통합 {value['pooled_macro_f1']:.3f}  fold평균 {value['fold_mean_macro_f1']:.3f}")
    print(f"중첩 선택 {report['nested']['macro_f1']:.3f}")
    print(f"저장: {args.output}")
    print(f"저장: {args.json}")


if __name__ == "__main__":
    main()
