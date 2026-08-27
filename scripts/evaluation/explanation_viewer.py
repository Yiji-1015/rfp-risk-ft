"""기존 924건 OOF 예측을 검색하는 정적 설명 화면을 만든다."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from scripts.evaluation.baselines import (
    CHAR_BALANCED,
    LABELS,
    _fit_pipeline,
    _model_input,
)
from scripts.evaluation.folds import make_lodo_folds


def character_explanation(
    text: str, fragment_contributions: Sequence[tuple[str, float]], *, limit: int = 8
) -> dict[str, Any]:
    """양의 문구 기여도를 원문 문자 위치에 겹쳐 표시할 값으로 바꾼다."""
    positive = sorted(
        ((fragment.strip(), float(value)) for fragment, value in fragment_contributions if value > 0 and fragment.strip()),
        key=lambda item: item[1],
        reverse=True,
    )
    strengths = np.zeros(len(text), dtype=float)
    for fragment, value in positive:
        start = 0
        while (position := text.find(fragment, start)) >= 0:
            strengths[position : position + len(fragment)] += value / len(fragment)
            start = position + 1
    if strengths.size and strengths.max() > 0:
        strengths /= strengths.max()
    return {
        "strengths": [round(float(value), 4) for value in strengths],
        "fragments": [
            {"text": fragment, "contribution": round(value, 4)}
            for fragment, value in positive[:limit]
        ],
    }


def build_explanation_records(
    rows: Sequence[dict[str, Any]], oof_path: Path
) -> list[dict[str, Any]]:
    """평가 당시 fold의 문자 Logistic을 재학습해 로컬 기여도를 붙인다."""
    with oof_path.open(encoding="utf-8-sig", newline="") as handle:
        oof = {record["requirement_uid"]: record for record in csv.DictReader(handle)}
    row_by_uid = {row["requirement_uid"]: row for row in rows}
    explanations: dict[str, dict[str, Any]] = {}

    for fold in make_lodo_folds(rows):
        fit_rows, _, test_rows = fold.split(rows)
        pipeline = _fit_pipeline(CHAR_BALANCED, fit_rows)
        vectorizer = pipeline.named_steps["features"]
        matrix = vectorizer.transform(_model_input(CHAR_BALANCED, test_rows))
        classifier = pipeline.named_steps["clf"]
        names = vectorizer.get_feature_names_out()
        class_positions = {label: i for i, label in enumerate(classifier.classes_)}
        reproduced = classifier.predict(matrix)

        for index, row in enumerate(test_rows):
            saved = oof[row["requirement_uid"]]
            predicted = saved["char_logistic_pred"]
            if reproduced[index] != predicted:
                raise ValueError(f"저장된 OOF 예측과 재학습 예측이 다릅니다: {row['requirement_uid']}")
            probabilities = np.array(
                [float(saved[f"char_logistic_p_{label}"]) for label in LABELS]
            )
            runner_up = LABELS[int(np.argsort(probabilities)[-2])]
            difference = (
                classifier.coef_[class_positions[predicted]]
                - classifier.coef_[class_positions[runner_up]]
            )
            vector = matrix.getrow(index)
            contributions = [
                (str(names[feature]), float(value * difference[feature]))
                for feature, value in zip(vector.indices, vector.data)
            ]
            explanations[row["requirement_uid"]] = {
                "explanation_label": predicted,
                "explanation_runner_up": runner_up,
                **character_explanation(row["raw_requirement_text"], contributions),
            }

    records = []
    for uid, saved in oof.items():
        source = row_by_uid[uid]
        explanation = explanations[uid]
        record: dict[str, Any] = {
            "requirement_uid": uid,
            "test_document": saved["test_document"],
            "raw_requirement_text": source["raw_requirement_text"],
            "gold": saved["gold"],
            "tfidf_e5_weight": float(saved["tfidf_e5_weight"]),
            "soft_vote_pred": saved["soft_vote_pred"],
            "review_union_pred": saved["review_union_pred"],
            "all_agree": saved["all_agree"].lower() == "true",
            "all_wrong": saved["all_wrong"].lower() == "true",
            "explanation_label": explanation["explanation_label"],
            "explanation_runner_up": explanation["explanation_runner_up"],
            "explanation_strengths": explanation["strengths"],
            "explanation_fragments": explanation["fragments"],
        }
        for prefix in ("char_logistic", "word_char_logistic", "tfidf_e5_hybrid", "soft_vote"):
            record[f"{prefix}_pred"] = saved[f"{prefix}_pred"]
            record[f"{prefix}_probabilities"] = [
                float(saved[f"{prefix}_p_{label}"]) for label in LABELS
            ]
        records.append(record)
    return records


def _containing_phrases(text: str, fragment: str) -> list[str]:
    phrases = []
    for match in re.finditer(r"\S+", text):
        token = re.sub(
            r"^[^0-9A-Za-z가-힣]+|[^0-9A-Za-z가-힣]+$", "", match.group()
        )
        if fragment.casefold() in token.casefold() and 2 <= len(token) <= 20:
            phrases.append(token)
    if phrases:
        return list(dict.fromkeys(phrases))
    return [fragment] if len(re.findall(r"[0-9A-Za-z가-힣]", fragment)) >= 2 else []


def aggregate_phrase_evidence(
    records: Sequence[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """겹치는 문자 n-gram 기여도를 원문에서 읽을 수 있는 단어로 합친다."""
    totals: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    requirements: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for record in records:
        label = record["explanation_label"]
        for fragment in record["explanation_fragments"]:
            phrases = _containing_phrases(
                record["raw_requirement_text"], fragment["text"]
            )
            if not phrases:
                continue
            share = float(fragment["contribution"]) / len(phrases)
            for phrase in phrases:
                totals[label][phrase] += share
                requirements[label][phrase].add(record["requirement_uid"])
    return {
        label: [
            {
                "phrase": phrase,
                "total_contribution": round(value, 4),
                "requirement_count": len(requirements[label][phrase]),
            }
            for phrase, value in sorted(
                totals[label].items(), key=lambda item: item[1], reverse=True
            )
        ]
        for label in LABELS
    }


def _write_phrase_cloud(
    evidence: Sequence[dict[str, Any]], path: Path, label: str
) -> None:
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "rfp-risk-ft-matplotlib")
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    installed = {font.name for font in font_manager.fontManager.ttflist}
    plt.rcParams["font.family"] = next(
        (
            font
            for font in ("Malgun Gothic", "NanumGothic", "Noto Sans CJK KR")
            if font in installed
        ),
        "DejaVu Sans",
    )
    colors = {
        "통상수용": "#2f855a",
        "견적반영": "#b7791f",
        "계약·질의검토": "#c53030",
    }
    items = list(evidence[:36])
    maximum = max((item["total_contribution"] for item in items), default=1.0)
    fig, axis = plt.subplots(figsize=(12, 7), dpi=160)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    axis.set_title(f"{label} 예측에 기여한 문구", fontsize=18, pad=18)
    for index, item in enumerate(items):
        row, column = divmod(index, 4)
        weight = (item["total_contribution"] / maximum) ** 0.5
        size = max(10, min(29, 11 + 18 * weight, 180 / len(item["phrase"])))
        axis.text(
            (column + 0.5) / 4,
            0.92 - row * 0.105,
            item["phrase"],
            ha="center",
            va="center",
            fontsize=size,
            color=colors[label],
            alpha=0.55 + 0.45 * weight,
        )
    fig.text(
        0.5,
        0.02,
        "OOF 문자 Logistic의 TF-IDF × 계수 차이 합계 · 실제 위험의 인과 설명이 아님",
        ha="center",
        fontsize=9,
        color="#666666",
    )
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_portable_exports(
    records: Sequence[dict[str, Any]], output_dir: Path
) -> list[Path]:
    """다른 도구에서 읽을 JSON·CSV와 사람용 문구 클라우드를 저장한다."""
    output_dir.mkdir(parents=True, exist_ok=True)
    explanation_path = output_dir / "model_explanations.json"
    explanation_path.write_text(
        json.dumps(records, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    evidence = aggregate_phrase_evidence(records)
    phrase_path = output_dir / "model_explanation_phrases.csv"
    with phrase_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "label",
                "rank",
                "phrase",
                "total_contribution",
                "requirement_count",
            ),
        )
        writer.writeheader()
        for label in LABELS:
            for rank, item in enumerate(evidence[label], start=1):
                writer.writerow({"label": label, "rank": rank, **item})
    cloud_paths = []
    for label, name in zip(LABELS, ("accept", "quote", "review")):
        path = output_dir / f"wordcloud_{name}.png"
        _write_phrase_cloud(evidence[label], path, label)
        cloud_paths.append(path)
    return [explanation_path, phrase_path, *cloud_paths]


def render_html(records: Sequence[dict[str, Any]]) -> str:
    payload = json.dumps(records, ensure_ascii=False, separators=(",", ":")).replace(
        "</", "<\\/"
    )
    return _HTML.replace("__RECORDS__", payload)


_HTML = """<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RFP 요구사항 설명</title>
<style>
:root { color-scheme: light dark; --bg:#f5f7fb; --panel:#fff; --text:#172033; --muted:#667085; --line:#d9dfeb; --accept:#2f855a; --quote:#b7791f; --review:#c53030; --soft:#edf2f7; }
@media (prefers-color-scheme: dark) { :root { --bg:#101522; --panel:#171e2d; --text:#edf2f7; --muted:#a8b1c2; --line:#344054; --soft:#273044; } }
* { box-sizing:border-box; } body { margin:0; background:var(--bg); color:var(--text); font:15px/1.55 system-ui,"Malgun Gothic",sans-serif; }
main { max-width:1100px; margin:auto; padding:24px; } h1 { font-size:24px; margin:0 0 4px; } h2 { font-size:17px; margin:0 0 12px; } p { margin:6px 0; }
.muted { color:var(--muted); } .panel { background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:18px; margin-top:16px; }
.controls { display:grid; grid-template-columns:180px 1fr 1fr; gap:12px; } label { display:grid; gap:5px; font-weight:600; } select,input { width:100%; padding:9px 10px; border:1px solid var(--line); border-radius:8px; background:var(--panel); color:var(--text); font:inherit; }
.meta { display:flex; flex-wrap:wrap; gap:8px 16px; margin-bottom:14px; } .badge { padding:2px 8px; border-radius:999px; background:var(--soft); }
.requirement { font-size:18px; line-height:2; word-break:keep-all; overflow-wrap:anywhere; }
mark { color:inherit; border-radius:3px; padding:2px 0; background:color-mix(in srgb, var(--mark) calc(18% + var(--strength) * 55%), transparent); }
.accept { --mark:var(--accept); } .quote { --mark:var(--quote); } .review { --mark:var(--review); }
table { width:100%; border-collapse:collapse; } th,td { padding:9px 8px; border-bottom:1px solid var(--line); text-align:right; } th:first-child,td:first-child { text-align:left; } th { color:var(--muted); font-weight:600; }
.winner { font-weight:700; } .fragments { display:flex; flex-wrap:wrap; gap:8px; padding:0; list-style:none; } .fragments li { background:var(--soft); padding:5px 9px; border-radius:7px; }
.result { font-size:17px; font-weight:700; } .legend { display:flex; flex-wrap:wrap; gap:14px; margin-top:10px; }
.dot { display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:5px; } .dot.accept{background:var(--accept)} .dot.quote{background:var(--quote)} .dot.review{background:var(--review)}
@media(max-width:720px){ main{padding:14px}.controls{grid-template-columns:1fr}.table-wrap{overflow-x:auto}.requirement{font-size:16px} }
</style>
</head>
<body><main>
<h1>RFP 요구사항 설명</h1>
<p class="muted">평가에서 제외된 문서로 예측한 OOF 924건입니다. 색은 문자 TF-IDF Logistic의 계산 근거이며 실제 위험 원인을 뜻하지 않습니다.</p>
<section class="panel controls" aria-label="요구사항 선택">
<label>문서<select id="document-filter"></select></label>
<label>검색<input id="requirement-search" type="search" placeholder="UID 또는 문구 검색"></label>
<label>요구사항<select id="requirement-select"></select></label>
</section>
<section class="panel" aria-live="polite">
<div class="meta"><span id="uid" class="badge"></span><span>정답 <strong id="gold"></strong></span><span id="agreement"></span></div>
<h2>문자 Logistic이 예측을 고른 근거</h2>
<div id="requirement" class="requirement"></div>
<div class="legend"><span><i class="dot accept"></i>통상수용</span><span><i class="dot quote"></i>견적반영</span><span><i class="dot review"></i>계약·질의검토</span></div>
<p class="muted">예측 클래스와 2위 클래스의 계수 차이에 TF-IDF를 곱했습니다. 진할수록 예측 선택에 더 크게 기여했습니다.</p>
<ul id="fragments" class="fragments"></ul>
</section>
<section class="panel">
<h2>세 모델 확률과 투표</h2>
<div class="table-wrap"><table><thead><tr><th>모델</th><th>통상수용</th><th>견적반영</th><th>계약검토</th><th>예측</th></tr></thead><tbody id="model-table"></tbody></table></div>
<p class="result">검토 우선 규칙: <span id="union-result"></span></p>
<p class="muted">확률은 모델끼리 비교하기 위한 값이며 보정된 실제 위험 확률은 아닙니다. E5 임베딩 차원은 색칠 설명에서 제외했습니다.</p>
</section>
</main>
<script id="records" type="application/json">__RECORDS__</script>
<script>
const rows=JSON.parse(document.getElementById('records').textContent);
const doc=document.getElementById('document-filter'), search=document.getElementById('requirement-search'), select=document.getElementById('requirement-select');
const labels=['통상수용','견적반영','계약·질의검토'];
const models=[['char_logistic','문자 TF-IDF'],['word_char_logistic','단어+문자 TF-IDF'],['tfidf_e5_hybrid','TF-IDF+E5'],['soft_vote','동일 가중 평균']];
const esc=s=>String(s).replace(/[&<>\"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',"'":'&#39;'}[c]));
const cls=l=>l==='통상수용'?'accept':l==='견적반영'?'quote':'review';
for(const value of ['전체',...new Set(rows.map(r=>r.test_document))]) doc.add(new Option(value,value));
function filtered(){const q=search.value.trim().toLowerCase(); return rows.filter(r=>(doc.value==='전체'||r.test_document===doc.value)&&(!q||r.requirement_uid.toLowerCase().includes(q)||r.raw_requirement_text.toLowerCase().includes(q)));}
function refreshList(){const previous=select.value, found=filtered(); select.replaceChildren(...found.map(r=>new Option(`${r.requirement_uid} · ${r.raw_requirement_text.slice(0,45)}`,r.requirement_uid))); if(found.some(r=>r.requirement_uid===previous))select.value=previous; render(found.find(r=>r.requirement_uid===select.value)||found[0]);}
function highlighted(r){return [...r.raw_requirement_text].map((ch,i)=>{const strength=r.explanation_strengths[i]||0; return strength?`<mark class="${cls(r.explanation_label)}" style="--strength:${strength}">${esc(ch)}</mark>`:esc(ch)}).join('');}
function render(r){if(!r){document.getElementById('requirement').textContent='검색 결과가 없습니다.';return} document.getElementById('uid').textContent=r.requirement_uid; document.getElementById('gold').textContent=r.gold; document.getElementById('agreement').textContent=r.all_agree?'세 모델 합의':'세 모델 불일치'; document.getElementById('requirement').innerHTML=highlighted(r); document.getElementById('fragments').innerHTML=r.explanation_fragments.map(f=>`<li><strong>${esc(f.text)}</strong> · ${f.contribution.toFixed(3)}</li>`).join('')||'<li>원문에서 직접 대응되는 양의 조각 없음</li>'; document.getElementById('model-table').innerHTML=models.map(([key,name])=>{const p=r[key+'_probabilities'], pred=r[key+'_pred']; return `<tr><td>${name}</td>${p.map((v,i)=>`<td class="${labels[i]===pred?'winner':''}">${v.toFixed(3)}</td>`).join('')}<td class="${cls(pred)} winner">${pred}</td></tr>`}).join(''); document.getElementById('union-result').textContent=r.review_union_pred;}
doc.addEventListener('change',refreshList); search.addEventListener('input',refreshList); select.addEventListener('change',()=>render(rows.find(r=>r.requirement_uid===select.value))); refreshList();
</script></body></html>"""


def _main() -> None:
    from scripts.labeling.label_dataset import load_label_dataset

    parser = argparse.ArgumentParser()
    parser.add_argument("--oof", type=Path, default=Path("reports/model_candidate_oof.csv"))
    parser.add_argument("--output", type=Path, default=Path("reports/explanation_viewer.html"))
    args = parser.parse_args()
    rows, _ = load_label_dataset()
    records = build_explanation_records(rows, args.oof)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_html(records), encoding="utf-8")
    exports = write_portable_exports(records, args.output.parent)
    print(f"설명 화면 저장: {args.output} ({len(records)}건)")
    print("추가 산출물: " + ", ".join(str(path) for path in exports))


if __name__ == "__main__":
    _main()
