#!/usr/bin/env python3
"""Create reproducible text variants for preprocessing ablation experiments."""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data" / "processed" / "requirements_v0.1.0.jsonl"
DEFAULT_OUTPUT = ROOT / "data" / "samples" / "preprocessing_sample_v0.1.0.csv"
LONGEST_OUTPUT = ROOT / "data" / "samples" / "preprocessing_longest_v0.1.0.csv"
LARGEST_DOCUMENT_OUTPUT = (
    ROOT / "data" / "samples" / "preprocessing_largest_document_v0.1.0.csv"
)

# Only line-initial list markers are normalized. Hyphens inside terms such as
# "AI-플랫폼" and numeric values such as "1-2년" are therefore preserved.
SYMBOL_BULLETS = "◦○●•ㆍ▪■□◆◇▶▷►▸▹‣⁃∙·"
SYMBOL_BULLET_RE = re.compile(
    rf"^(?P<indent>[ \t]*)(?P<marker>[{re.escape(SYMBOL_BULLETS)}]|[-–—])"
    rf"(?P<space>[ \t]*)(?P<body>\S.*|)$"
)


def normalize_common(text: str) -> str:
    """Apply loss-minimizing normalization shared by every text variant."""
    text = html.unescape(text or "")
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_symbol_bullet_line(line: str) -> bool:
    return SYMBOL_BULLET_RE.match(line) is not None


def normalize_bullets(text: str) -> str:
    """Map heterogeneous symbolic bullets to '-' while retaining item lines."""
    text = normalize_common(text)
    normalized: list[str] = []
    for line in text.splitlines():
        match = SYMBOL_BULLET_RE.match(line)
        if match:
            body = match.group("body").strip()
            normalized.append(f"- {body}".rstrip())
        else:
            normalized.append(line)
    return "\n".join(normalized).strip()


def flatten_list_text(text: str) -> str:
    """Remove symbolic list markers and line boundaries without deleting words."""
    text = normalize_bullets(text)
    parts: list[str] = []
    for line in text.splitlines():
        line = re.sub(r"^-\s*", "", line).strip()
        if line:
            parts.append(line)
    return re.sub(r"\s+", " ", " ".join(parts)).strip()


def make_model_text(requirement_name: str | None, body: str, variant: str) -> str:
    combined = "\n".join(part for part in [requirement_name or "", body] if part)
    if variant == "raw":
        return normalize_common(combined)
    if variant == "normalized-list":
        return normalize_bullets(combined)
    if variant == "flat":
        return flatten_list_text(combined)
    raise ValueError(f"unknown variant: {variant}")


# --- 전처리 ablation용 마스킹 --------------------------------------------------
#
# 설명 보고서에서 모델이 근거로 쓰는 문구를 감사한 결과, 요구사항의 내용이 아니라
# **작성 양식**을 가리키는 표현이 상위에 올라왔다. 아래 세 규칙은 그 양식 신호를 하나씩
# 지워보기 위한 것이며, 각각 한 문장으로 정의된다. p값으로 고른 단어 목록이 아니므로
# 이 10문서에 나오지 않은 표기에도 같은 규칙이 적용된다.

# 요구를 *누가* 하는지 가리키는 말. 조사가 붙은 형태까지 한 번에 잡는다.
SUBJECT_TERMS = (
    "제안사", "계약상대자", "사업수행사", "사업자", "수급인", "용역업체", "수행사",
    "발주기관", "발주처", "발주자", "수요기관", "주관기관", "위원회", "공사", "공단",
)
SUBJECT_TOKEN = "<주체>"

# 의무·서술 어미. 떼고 남는 어간이 두 글자 미만이면 어절을 그대로 둔다.
VERB_ENDINGS = (
    "하여야 한다", "해야 한다", "하여야", "하여", "하고", "하는", "한다", "해야",
    "합니다", "하며", "하도록",
)
_ENDING_RE = re.compile(rf"(?:{'|'.join(map(re.escape, VERB_ENDINGS))})$")

# 명사 뒤 조사. 어간이 두 글자 미만이면 떼지 않는다("정의"의 "의"를 조사로 보지 않는다).
# ponytail: 형태소 분석기 없이 정규식으로 근사한다. 어미 변화까지 묶어야 하면 올린다.
JOSA = (
    "에서의", "으로써", "으로서", "에게서", "이라도", "으로", "에서", "에게",
    "까지", "부터", "보다", "라도", "이나", "와의", "과의", "은", "는", "이",
    "가", "을", "를", "의", "에", "도", "만", "나", "와", "과", "로",
)
_JOSA_RE = re.compile(rf"(?:{'|'.join(JOSA)})$")

# 어절 = 공백이 아닌 연속 구간. 줄바꿈과 들여쓰기는 건드리지 않고 본문만 바꾼다.
_WORD_EDGE_RE = re.compile(
    r"^(?P<head>[^0-9A-Za-z가-힣]*)(?P<body>.*?)(?P<tail>[^0-9A-Za-z가-힣]*)$"
)

# 주체 표기는 어절 안에 묻힌 형태를 건드리면 안 된다. `건설공사`·`공사비`의 "공사"는
# 발주기관이 아니라 공사(工事)다. 앞뒤가 한글·영숫자가 아닐 때만 치환한다.
_SUBJECT_RE = re.compile(
    r"(?<![0-9A-Za-z가-힣])"
    rf"(?:{'|'.join(sorted(map(re.escape, SUBJECT_TERMS), key=len, reverse=True))})"
    rf"(?:{'|'.join(JOSA)})?(?:는|은)?"
    r"(?![0-9A-Za-z가-힣])"
)


def strip_suffix(token: str, pattern: re.Pattern[str], *, minimum: int = 2) -> str:
    """어절 끝의 접미 표현을 떼되, 남는 어간이 너무 짧으면 그대로 둔다."""
    match = pattern.search(token)
    if not match or match.start() < minimum:
        return token
    return token[: match.start()]


def _map_tokens(text: str, transform) -> str:
    """어절의 양끝 기호는 보존한 채 본문만 바꾼다. 공백과 줄바꿈은 그대로 둔다."""

    def replace(match: re.Match[str]) -> str:
        parts = _WORD_EDGE_RE.match(match.group())
        body = parts.group("body")
        return f"{parts.group('head')}{transform(body) if body else body}{parts.group('tail')}"

    return re.sub(r"\S+", replace, text)


def mask_subjects(text: str) -> str:
    """요구의 주체 표기를 하나의 자리표시자로 바꾼다."""
    return _SUBJECT_RE.sub(SUBJECT_TOKEN, text)


def normalize_endings(text: str) -> str:
    """의무·서술 어미를 어간으로 정규화한다."""
    return _map_tokens(text, lambda body: strip_suffix(body, _ENDING_RE))


def strip_particles(text: str) -> str:
    """명사 뒤 조사를 떼어 같은 어휘의 표기 변이를 합친다."""
    return _map_tokens(text, lambda body: strip_suffix(body, _JOSA_RE))


MASKS = {
    "subject": mask_subjects,
    "ending": normalize_endings,
    "josa": strip_particles,
}


def apply_mask(text: str, mask: str | None) -> str:
    """마스킹 규칙을 적용한다. `None`이나 `'none'`이면 원문 그대로다.

    `'subject+josa'`처럼 `+`로 이으면 적은 순서대로 겹쳐 적용한다. 단일 규칙 비교가
    통제 실험이고 결합은 그 뒤의 탐색이므로, 결합 결과는 사전 등록된 결과가 아니다.
    """
    if not mask or mask == "none":
        return text
    for name in mask.split("+"):
        if name not in MASKS:
            raise ValueError(f"unknown mask: {name} (사용 가능: {', '.join(MASKS)})")
        text = MASKS[name](text)
    return text


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def choose_sample(records: list[dict], per_document: int) -> list[dict]:
    """Choose bullet-bearing examples first, then fill deterministically."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[record["document_id"]].append(record)

    selected: list[dict] = []
    for document_id in sorted(grouped):
        rows = grouped[document_id]
        with_bullets = [
            row
            for row in rows
            if any(is_symbol_bullet_line(line) for line in normalize_common(row["raw_requirement_text"]).splitlines())
        ]
        without_bullets = [row for row in rows if row not in with_bullets]
        selected.extend((with_bullets + without_bullets)[:per_document])
    return selected


def choose_longest(records: list[dict]) -> list[dict]:
    """Return the single requirement with the longest extracted body."""
    if not records:
        return []
    return [
        max(
            records,
            key=lambda row: (
                len(row.get("raw_requirement_text") or ""),
                row["requirement_uid"],
            ),
        )
    ]


def choose_largest_document(records: list[dict]) -> list[dict]:
    """Return all requirements from the largest source Markdown file."""
    source_files = {
        row["source_file"]: ROOT / "RFP_data" / "md" / row["source_file"]
        for row in records
    }
    if not source_files:
        return []
    largest_source = max(
        source_files,
        key=lambda name: (source_files[name].stat().st_size, name),
    )
    return [row for row in records if row["source_file"] == largest_source]


def write_sample(records: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "requirement_uid",
        "document_id",
        "requirement_id",
        "requirement_type",
        "requirement_name",
        "raw_text",
        "normalized_list_text",
        "flat_text",
    ]
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            body = record["raw_requirement_text"]
            writer.writerow(
                {
                    "requirement_uid": record["requirement_uid"],
                    "document_id": record["document_id"],
                    "requirement_id": record["requirement_id"],
                    "requirement_type": record["requirement_type"],
                    "requirement_name": record["requirement_name"],
                    "raw_text": make_model_text(record["requirement_name"], body, "raw"),
                    "normalized_list_text": make_model_text(
                        record["requirement_name"], body, "normalized-list"
                    ),
                    "flat_text": make_model_text(record["requirement_name"], body, "flat"),
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--per-document", type=int, default=2)
    parser.add_argument(
        "--longest",
        action="store_true",
        help="select only the requirement with the longest raw body",
    )
    parser.add_argument(
        "--largest-document",
        action="store_true",
        help="select every requirement from the largest source Markdown file",
    )
    args = parser.parse_args()
    if args.per_document < 1:
        parser.error("--per-document must be at least 1")

    records = read_jsonl(args.input)
    if args.longest and args.largest_document:
        parser.error("--longest and --largest-document cannot be used together")
    if args.longest:
        sample = choose_longest(records)
        default_output = LONGEST_OUTPUT
    elif args.largest_document:
        sample = choose_largest_document(records)
        default_output = LARGEST_DOCUMENT_OUTPUT
    else:
        sample = choose_sample(records, args.per_document)
        default_output = DEFAULT_OUTPUT
    output = args.output or default_output
    write_sample(sample, output)
    print(f"wrote {len(sample)} rows from {len({r['document_id'] for r in sample})} documents")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
