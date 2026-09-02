#!/usr/bin/env python3
"""모든 모델이 한 방향으로 틀린 경계 사례를 모아 성격을 센다.

`견적반영`과 `계약·질의검토`의 상호 혼동은 TF-IDF든 파인튜닝이든 오답의 33~39%를
차지하고, 모델 계열을 바꾸거나 용량을 세 배로 키워도 비율이 움직이지 않았다. 점수로는
거기까지가 끝이므로, **아홉 개 모델이 전원 일치로 반대쪽을 고른 건**만 따로 모아 무엇이
다른지 본다.

세는 값은 두 방향에서 다르다.

- 정답이 `견적반영`인데 전원이 `계약`이라 한 건 — 라벨을 가른 근거가 **부재**인지 본다.
  "수치 목표가 없어서", "무제한 재작업 조항이 없어서" 같은 부정문이 `reasoning`에
  얼마나 나오는지 센다.
- 정답이 `계약`인데 전원이 `견적`이라 한 건 — 불확실성 표지가 **원문에** 있는지, 그리고
  그 표지가 다른 집합과 구별되는지 본다. `reasoning`은 모델이 보지 않으므로 세지 않는다.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import re
import statistics
from pathlib import Path
from typing import Any, Sequence

from scripts.labeling.label_dataset import (
    DATASET_VERSION_ENV,
    DEFAULT_DATASET_KEY,
    get_model_text,
    load_label_dataset,
)

ROOT = Path(__file__).resolve().parents[2]
BOUNDARY = ("견적반영", "계약·질의검토")
TFIDF_COLUMNS = ("word_char_logistic_pred", "char_logistic_pred", "tfidf_e5_hybrid_pred")

# 근거가 "blocker가 아니다"로 끝나는지. 이것만 `reasoning`에서 찾는다.
BLOCKER_DENIAL = re.compile(r"blocker[^.]{0,40}(아니|없)")

# 원문에서 찾는 불확실성 표지. 모델이 실제로 보는 텍스트에만 적용한다.
MARKERS = {
    "협의": r"협의",
    "특정되지·명시되지": r"(특정되지|명시되지|정해지지|구체화되지)",
    "가능 여부": r"(가능\s*여부|가능한지|여부를)",
    "추후·별도": r"(추후|별도로|향후)",
    "폐쇄망": r"폐쇄망",
    "승인·허가": r"(승인|허가)",
}


def load_predictions(oof_path: Path, runs_path: Path) -> tuple[dict, dict, dict]:
    """TF-IDF 후보와 파인튜닝 실행의 건별 예측을 한 표로 모은다."""
    with oof_path.open(encoding="utf-8-sig", newline="") as handle:
        oof = {row["requirement_uid"]: row for row in csv.DictReader(handle)}

    tables: dict[str, dict[str, str]] = {}
    for column in TFIDF_COLUMNS:
        tables[column.replace("_pred", "")] = {uid: row[column] for uid, row in oof.items()}

    for line in runs_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        run = json.loads(line)
        config = run["config"]
        if config["fold"] != -1 or not run["results"][0].get("predictions"):
            continue
        name = f"{config['model'].split('/')[-1]}|{config['mask'] or 'raw'}|seed{config['seed']}"
        tables[name] = {
            item["requirement_uid"]: item["pred"]
            for fold in run["results"]
            for item in fold["predictions"]
        }

    uids = [uid for uid in oof if all(uid in table for table in tables.values())]
    return tables, {uid: oof[uid]["gold"] for uid in uids}, {uid: oof[uid]["test_document"] for uid in uids}


def unanimous_errors(tables, gold, uids) -> list[str]:
    """경계 라벨 중 모든 모델이 **한 방향으로** 틀린 건."""
    found = []
    for uid in uids:
        answer = gold[uid]
        if answer not in BOUNDARY:
            continue
        other = BOUNDARY[0] if answer == BOUNDARY[1] else BOUNDARY[1]
        if all(table[uid] == other for table in tables.values()):
            found.append(uid)
    return found


def profile(rows: dict, uids: Sequence[str]) -> dict[str, Any]:
    """집합 하나의 성격. 부정 근거 비율, 원가 구성, 난이도, 표지 비율, 길이."""
    if not uids:
        return {}
    texts = {uid: get_model_text(rows[uid]) for uid in uids}
    return {
        "count": len(uids),
        "blocker_denied": sum(1 for uid in uids if BLOCKER_DENIAL.search(rows[uid].get("reasoning") or "")) / len(uids),
        "cost_complex": sum(1 for uid in uids if rows[uid].get("cost_basis") == "복합") / len(uids),
        "build_high": sum(1 for uid in uids if rows[uid].get("build_difficulty") == "높음") / len(uids),
        "median_length": statistics.median(len(t) for t in texts.values()),
        "markers": {
            name: sum(1 for t in texts.values() if re.search(pattern, t)) / len(uids)
            for name, pattern in MARKERS.items()
        },
    }


def render(report: dict[str, Any]) -> str:
    groups = report["groups"]
    order = [k for k in ("hard_quote", "other_quote", "hard_review", "other_review", "accept") if k in groups]
    titles = {
        "hard_quote": "전원 오답 (정답 견적반영)",
        "other_quote": "나머지 견적반영",
        "hard_review": "전원 오답 (정답 계약)",
        "other_review": "나머지 계약·질의검토",
        "accept": "통상수용 전체",
    }
    lines = [
        f"# {report['dataset_version']} 모든 모델이 틀린 경계 사례",
        "",
        f"- 대상 모델 {report['model_count']}종 — TF-IDF 후보 3종과 파인튜닝 실행 {report['model_count'] - 3}종",
        f"- 경계 라벨 {report['boundary_total']}건 중 **전원이 한 방향으로 틀린 건 {report['unanimous']}건**",
        f"- 명령: `$env:{DATASET_VERSION_ENV}='{report['dataset_version']}'; "
        "python -m scripts.evaluation.boundary_agreement`",
        "",
        "## 집합별 성격",
        "",
        "| 집합 | 건수 | 근거가 blocker 부정 | cost=복합 | 구축 높음 | 원문 길이(중앙값) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key in order:
        g = groups[key]
        lines.append(
            f"| {titles[key]} | {g['count']} | {g['blocker_denied']:.1%} | {g['cost_complex']:.1%} | "
            f"{g['build_high']:.1%} | {g['median_length']:.0f}자 |"
        )

    lines += ["", "## 원문의 불확실성 표지", "", "`reasoning`이 아니라 모델이 실제로 보는 `model_text`에서 센다.", "",
              "| 표지 | " + " | ".join(titles[k] for k in order) + " |",
              "|---" * (len(order) + 1) + "|"]
    for name in MARKERS:
        lines.append(f"| {name} | " + " | ".join(f"{groups[k]['markers'][name]:.1%}" for k in order) + " |")

    lines += [
        "",
        "## 읽는 법",
        "",
        "**정답이 견적반영인데 전원이 계약이라 한 건은 어려운 항목이 아니다.** 구축 난이도가 "
        "높은 비율이 오히려 낮다. 대신 `협의`와 `승인·허가`가 다른 견적 건의 네 배·두 배로 "
        "나온다. 상주 인력, 하자보증 SLA, 유지관리 같은 **평범한 원가 항목이 계약 어투로 "
        "쓰인** 경우이고, 모델은 그 어투를 따라간다.",
        "",
        "라벨을 가른 정보는 그 어휘가 아니라 **쓰이지 않은 조항**이다 — 수치 목표가 없어서, "
        "무제한 재작업 조항이 없어서, 조달 불확실성이 없어서 견적이 된다. 다만 근거가 "
        "blocker를 부정하는 비율은 다른 견적 건도 비슷하므로(92%), 이것은 이 11건의 특징이 "
        "아니라 **견적 라벨 전체의 성질**이다. 부재로 정의된 라벨은 빈도를 세는 표현으로 "
        "잡기 어렵고, 계약 어투가 겹치면 그중 일부가 이렇게 넘어간다.",
        "",
        "**정답이 계약인데 전원이 견적이라 한 건은 부재 문제가 아니다.** 불확실성은 원문에 "
        "있다. 다만 두 가지가 겹친다. 첫째, `협의`·`특정되지 않아` 같은 **쉬운 표지가 오히려 "
        "적고**, 대신 `폐쇄망`처럼 **문서 밖 지식이 있어야 함의를 아는 단어**로 나타난다. "
        "둘째, 이 건들은 원문이 다른 계약 건의 두 배 길이여서 그 신호가 GPU·인프라·"
        "고급인력 같은 원가 어휘에 파묻힌다.",
        "",
        "사람이 이 건들을 쉽게 판정하는 이유도 여기 있다. 892자를 세지 않고 `폐쇄망` 한 단어에 "
        "걸리기 때문이다. 빈도가 아니라 함의로 읽는다.",
        "",
        "따라서 이 사례들이 모델 계열·용량·입력 변형에 반응하지 않은 것은 성능 부족이 아니라 "
        "**과제가 요구하는 정보의 성격** 때문이다. 후속 방향은 세 갈래로 갈린다 — 부재를 "
        "명시 feature로 만들기, 문서 밖 지식을 끌어오기, 긴 원문에서 신호가 희석되지 않게 하기.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    version = os.getenv(DATASET_VERSION_ENV, DEFAULT_DATASET_KEY)
    default_dir = ROOT / "reports" / "current" / version
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof", type=Path, default=default_dir / "model_candidate_oof.csv")
    parser.add_argument("--runs", type=Path, default=default_dir / "finetune_runs.jsonl")
    parser.add_argument("--output", type=Path, default=default_dir / "boundary_agreement.md")
    parser.add_argument("--cases", type=Path, default=default_dir / "boundary_agreement_cases.csv")
    args = parser.parse_args()

    rows = {row["requirement_uid"]: row for row in load_label_dataset()[0]}
    tables, gold, documents = load_predictions(args.oof, args.runs)
    uids = sorted(gold)
    hard = unanimous_errors(tables, gold, uids)
    hard_set = set(hard)

    def pick(label, inside):
        return [u for u in uids if gold[u] == label and ((u in hard_set) == inside)]

    report = {
        "dataset_version": version,
        "model_count": len(tables),
        "boundary_total": sum(1 for u in uids if gold[u] in BOUNDARY),
        "unanimous": len(hard),
        "groups": {
            "hard_quote": profile(rows, pick("견적반영", True)),
            "other_quote": profile(rows, pick("견적반영", False)),
            "hard_review": profile(rows, pick("계약·질의검토", True)),
            "other_review": profile(rows, pick("계약·질의검토", False)),
            "accept": profile(rows, [u for u in uids if gold[u] == "통상수용"]),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(report), encoding="utf-8")
    with args.cases.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["requirement_uid", "평가문서", "정답", "전원예측", "요구사항명",
                        "blockers", "cost_basis", "build_difficulty", "원문길이", "reasoning", "본문"],
        )
        writer.writeheader()
        for uid in sorted(hard, key=lambda u: (gold[u], u)):
            row = rows[uid]
            other = BOUNDARY[0] if gold[uid] == BOUNDARY[1] else BOUNDARY[1]
            writer.writerow({
                "requirement_uid": uid, "평가문서": documents[uid], "정답": gold[uid], "전원예측": other,
                "요구사항명": row.get("requirement_name") or "", "blockers": ", ".join(row.get("blockers") or []),
                "cost_basis": row.get("cost_basis") or "", "build_difficulty": row.get("build_difficulty") or "",
                "원문길이": len(get_model_text(row)), "reasoning": row.get("reasoning") or "",
                "본문": get_model_text(row),
            })

    counts = collections.Counter(gold[u] for u in hard)
    print(f"모델 {len(tables)}종 / 경계 {report['boundary_total']}건 중 전원 오답 {len(hard)}건 {dict(counts)}")
    print(f"저장: {args.output}")
    print(f"저장: {args.cases}")


if __name__ == "__main__":
    main()
