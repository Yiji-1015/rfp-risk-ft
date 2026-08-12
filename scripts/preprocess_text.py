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


ROOT = Path(__file__).resolve().parents[1]
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
