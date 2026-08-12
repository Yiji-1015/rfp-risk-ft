#!/usr/bin/env python3
"""
RFP 요구사항 동결 데이터셋 v0.2.0 EDA 스크립트

수행 작업:
1. 문서별 및 요구사항 유형별 행 수 집계
2. 본문 문자 길이, 단어 수, 공백 분할 토큰 길이 분포 통계 (min, max, mean, median, P90, P95, P99)
3. 중첩표(' | ') 포함 행, 특이 ID(canonical != source_id) 행, 최단/최장 본문 사례 추출
4. JSON 결과(reports/eda_v0.2.0.json), Markdown 보고서(reports/eda_v0.2.0.md), 
   Jupyter Notebook(notebooks/eda_v0.2.0.ipynb) 자동 생성
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any


def load_dataset(filepath: str) -> List[Dict[str, Any]]:
    """jsonl 데이터셋 로드"""
    records = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def calculate_stats(values: List[int]) -> Dict[str, Any]:
    """수치 리스트의 주요 통계량 계산"""
    if not values:
        return {"min": 0, "max": 0, "mean": 0.0, "median": 0.0, "p90": 0, "p95": 0, "p99": 0}
    sorted_v = sorted(values)
    n = len(sorted_v)
    
    def percentile(p: float) -> float:
        k = (n - 1) * p
        f = int(k)
        c = f + 1
        if c < n:
            return sorted_v[f] + (k - f) * (sorted_v[c] - sorted_v[f])
        return float(sorted_v[f])

    return {
        "min": sorted_v[0],
        "max": sorted_v[-1],
        "mean": round(sum(sorted_v) / n, 2),
        "median": round(percentile(0.50), 2),
        "p90": round(percentile(0.90), 2),
        "p95": round(percentile(0.95), 2),
        "p99": round(percentile(0.99), 2),
    }


def analyze_dataset(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """데이터셋 EDA 종합 분석"""
    total_count = len(records)
    
    doc_counts = {}
    type_counts = {}
    doc_type_counts = {}
    
    char_lengths = []
    word_counts = []
    
    doc_char_lengths = {}
    
    id_mismatches = []
    nested_table_records = []
    
    for r in records:
        doc_id = r.get("document_id", "unknown")
        req_type = r.get("requirement_type", "unknown")
        req_uid = r.get("requirement_uid", "")
        req_id = r.get("requirement_id", "")
        source_req_id = r.get("source_requirement_id", "")
        raw_text = r.get("raw_requirement_text", "")
        
        # 문서별 집계
        doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1
        
        # 유형별 집계
        type_counts[req_type] = type_counts.get(req_type, 0) + 1
        
        # 문서 x 유형 교차 집계
        if doc_id not in doc_type_counts:
            doc_type_counts[doc_id] = {}
        doc_type_counts[doc_id][req_type] = doc_type_counts[doc_id].get(req_type, 0) + 1
        
        # 본문 길이
        c_len = len(raw_text)
        w_len = len(raw_text.split())
        
        char_lengths.append(c_len)
        word_counts.append(w_len)
        
        if doc_id not in doc_char_lengths:
            doc_char_lengths[doc_id] = []
        doc_char_lengths[doc_id].append(c_len)
        
        # 특이 ID 검사
        if req_id != source_req_id:
            id_mismatches.append({
                "requirement_uid": req_uid,
                "document_id": doc_id,
                "requirement_id": req_id,
                "source_requirement_id": source_req_id,
                "requirement_name": r.get("requirement_name", "")
            })
            
        # 중첩표 검사 (' | ' 포함 여부)
        if " | " in raw_text:
            nested_table_records.append({
                "requirement_uid": req_uid,
                "document_id": doc_id,
                "requirement_id": req_id,
                "requirement_name": r.get("requirement_name", ""),
                "char_length": c_len
            })

    # 전체 본문 길이 통계
    overall_char_stats = calculate_stats(char_lengths)
    overall_word_stats = calculate_stats(word_counts)
    
    # 문서별 본문 길이 통계
    doc_char_stats = {
        doc_id: calculate_stats(lengths)
        for doc_id, lengths in doc_char_lengths.items()
    }
    
    # 본문 최단/최장 Top 5
    sorted_by_len = sorted(records, key=lambda x: len(x.get("raw_requirement_text", "")))
    shortest_samples = [
        {
            "requirement_uid": r.get("requirement_uid"),
            "document_id": r.get("document_id"),
            "requirement_id": r.get("requirement_id"),
            "requirement_name": r.get("requirement_name"),
            "char_length": len(r.get("raw_requirement_text", "")),
            "snippet": r.get("raw_requirement_text", "")[:100].replace("\n", " ")
        }
        for r in sorted_by_len[:5]
    ]
    longest_samples = [
        {
            "requirement_uid": r.get("requirement_uid"),
            "document_id": r.get("document_id"),
            "requirement_id": r.get("requirement_id"),
            "requirement_name": r.get("requirement_name"),
            "char_length": len(r.get("raw_requirement_text", "")),
            "snippet": r.get("raw_requirement_text", "")[:100].replace("\n", " ")
        }
        for r in sorted_by_len[-5:][::-1]
    ]

    return {
        "dataset_version": records[0].get("dataset_version", "v0.2.0") if records else "v0.2.0",
        "total_records": total_count,
        "document_counts": doc_counts,
        "type_counts": type_counts,
        "document_type_counts": doc_type_counts,
        "char_length_stats": overall_char_stats,
        "word_count_stats": overall_word_stats,
        "document_char_length_stats": doc_char_stats,
        "id_mismatches": id_mismatches,
        "nested_table_count": len(nested_table_records),
        "nested_table_samples": nested_table_records[:10],
        "shortest_samples": shortest_samples,
        "longest_samples": longest_samples
    }


def generate_markdown_report(eda_result: Dict[str, Any]) -> str:
    """Markdown 종합 보고서 텍스트 생성"""
    lines = []
    lines.append(f"# 요구사항 데이터셋 v0.2.0 EDA 보고서")
    lines.append("")
    lines.append(f"- **동결 데이터셋**: `data/processed/requirements_v0.2.0.jsonl`")
    lines.append(f"- **총 행 수**: {eda_result['total_records']:,} 행 (요구사항 ID 1개 = 1행)")
    lines.append(f"- **대상 문서 수**: {len(eda_result['document_counts'])} 개")
    lines.append(f"- **중첩표 포함 행 수**: {eda_result['nested_table_count']} 행")
    lines.append(f"- **특이 ID (Canonical != Source) 행 수**: {len(eda_result['id_mismatches'])} 행")
    lines.append("")
    
    # 1. 문서별 요구사항 수
    lines.append("## 1. 문서별 요구사항 행 수")
    lines.append("")
    lines.append("| document_id | 행 수 | 비율 (%) |")
    lines.append("|---|---:|---:|")
    for doc_id, count in sorted(eda_result['document_counts'].items(), key=lambda x: x[1], reverse=True):
        ratio = (count / eda_result['total_records']) * 100
        lines.append(f"| `{doc_id}` | {count:,} | {ratio:.1f}% |")
    lines.append(f"| **합계** | **{eda_result['total_records']:,}** | **100.0%** |")
    lines.append("")

    # 2. 요구사항 유형별 분포
    lines.append("## 2. 요구사항 유형별 분포")
    lines.append("")
    lines.append("| requirement_type | 행 수 | 비율 (%) |")
    lines.append("|---|---:|---:|")
    for req_type, count in sorted(eda_result['type_counts'].items(), key=lambda x: x[1], reverse=True):
        ratio = (count / eda_result['total_records']) * 100
        lines.append(f"| `{req_type}` | {count:,} | {ratio:.1f}% |")
    lines.append("")

    # 3. 본문 길이 통계 (문자 수 및 단어 수)
    lines.append("## 3. 본문 길이 및 단어 수 분포")
    lines.append("")
    lines.append("### 전체 데이터셋 길이 통계")
    lines.append("")
    lines.append("| 구분 | 최소 | 중앙값 | 평균 | P90 | P95 | P99 | 최대 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    c_s = eda_result['char_length_stats']
    w_s = eda_result['word_count_stats']
    lines.append(f"| **문자 수 (자)** | {c_s['min']} | {c_s['median']} | {c_s['mean']} | {c_s['p90']} | {c_s['p95']} | {c_s['p99']} | {c_s['max']} |")
    lines.append(f"| **단어 수 (어절)** | {w_s['min']} | {w_s['median']} | {w_s['mean']} | {w_s['p90']} | {w_s['p95']} | {w_s['p99']} | {w_s['max']} |")
    lines.append("")

    lines.append("### 문서별 본문 문자 길이 통계")
    lines.append("")
    lines.append("| document_id | 최소 | 중앙값 | 평균 | P95 | 최대 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for doc_id, ds in sorted(eda_result['document_char_length_stats'].items()):
        lines.append(f"| `{doc_id}` | {ds['min']} | {ds['median']} | {ds['mean']} | {ds['p95']} | {ds['max']} |")
    lines.append("")

    # 4. 특이 사례 및 극단치
    lines.append("## 4. 특이사항 및 극단치 분석")
    lines.append("")
    lines.append("### 4.1 Canonical ID와 Source ID 불일치 건")
    lines.append("")
    if eda_result['id_mismatches']:
        lines.append("| requirement_uid | requirement_id | source_requirement_id | 요구사항명 |")
        lines.append("|---|---|---|---|")
        for item in eda_result['id_mismatches']:
            lines.append(f"| `{item['requirement_uid']}` | `{item['requirement_id']}` | `{item['source_requirement_id']}` | {item['requirement_name']} |")
    else:
        lines.append("- 불일치 건 없음")
    lines.append("")

    lines.append("### 4.2 중첩표(' | ') 포함 행 요약")
    lines.append("")
    lines.append(f"- 총 **{eda_result['nested_table_count']}개** 요구사항에 중첩표 구분자(` | `)가 포함되어 있습니다.")
    lines.append("")

    lines.append("### 4.3 최단 본문 요구사항 (Top 5)")
    lines.append("")
    lines.append("| requirement_uid | 문자 수 | 스니펫 |")
    lines.append("|---|---:|---|")
    for item in eda_result['shortest_samples']:
        lines.append(f"| `{item['requirement_uid']}` | {item['char_length']} | {item['snippet']}... |")
    lines.append("")

    lines.append("### 4.4 최장 본문 요구사항 (Top 5)")
    lines.append("")
    lines.append("| requirement_uid | 문자 수 | 스니펫 |")
    lines.append("|---|---:|---|")
    for item in eda_result['longest_samples']:
        lines.append(f"| `{item['requirement_uid']}` | {item['char_length']} | {item['snippet']}... |")
    lines.append("")

    return "\n".join(lines)


def generate_jupyter_notebook(eda_result: Dict[str, Any]) -> Dict[str, Any]:
    """EDA 결과를 포함하는 주피터 노트북 (.ipynb) JSON 데이터 생성"""
    cells = []
    
    # 셀 1: 제목 및 개요
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# RFP 요구사항 데이터셋 v0.2.0 EDA 노트북\n",
            "\n",
            "이 노트북은 `data/processed/requirements_v0.2.0.jsonl` (총 1,024행) 동결 데이터셋의 문서별/유형별 분포, 본문 길이, 중첩표 및 특이 ID를 탐색하고 시각화합니다."
        ]
    })
    
    # 셀 2: 환경 설정 및 임포트
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "import json\n",
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "from pathlib import Path\n",
            "\n",
            "# 한글 폰트 설정 (Windows/Linux/Mac 공통 지원 시도)\n",
            "plt.rcParams['font.family'] = 'Malgun Gothic' if os.name == 'nt' else 'AppleGothic'\n",
            "plt.rcParams['axes.unicode_minus'] = False\n",
            "\n",
            "dataset_path = Path('../data/processed/requirements_v0.2.0.jsonl')\n",
            "if not dataset_path.exists():\n",
            "    dataset_path = Path('data/processed/requirements_v0.2.0.jsonl')\n",
            "\n",
            "df = pd.read_json(dataset_path, lines=True)\n",
            "print(f\"데이터셋 로드 완료: 총 {len(df):,} 행\")\n",
            "df.head(3)"
        ]
    })
    
    # 셀 3: 문서별 행 수 집계
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 1. 문서별 요구사항 개수 집계\n",
            "doc_counts = df['document_id'].value_counts().reset_index()\n",
            "doc_counts.columns = ['document_id', 'count']\n",
            "doc_counts['ratio_pct'] = (doc_counts['count'] / len(df) * 100).round(2)\n",
            "display(doc_counts)\n",
            "\n",
            "# 시각화\n",
            "plt.figure(figsize=(10, 5))\n",
            "sns.barplot(data=doc_counts, x='count', y='document_id', palette='viridis')\n",
            "plt.title('문서별 요구사항 행 수 분포')\n",
            "plt.xlabel('요구사항 수')\n",
            "plt.ylabel('문서 ID')\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    })

    # 셀 4: 요구사항 유형별 분포
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 2. 요구사항 유형별 분포\n",
            "type_counts = df['requirement_type'].value_counts().reset_index()\n",
            "type_counts.columns = ['requirement_type', 'count']\n",
            "display(type_counts)\n",
            "\n",
            "# 시각화\n",
            "plt.figure(figsize=(10, 4))\n",
            "sns.barplot(data=type_counts, x='count', y='requirement_type', palette='magma')\n",
            "plt.title('요구사항 유형별 분포')\n",
            "plt.xlabel('행 수')\n",
            "plt.ylabel('요구사항 유형')\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    })

    # 셀 5: 본문 길이 분포 (문자 수 & 단어 수)
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 3. 본문 문자 길이 및 단어 수 파생 변수 생성\n",
            "df['char_len'] = df['raw_requirement_text'].apply(len)\n",
            "df['word_len'] = df['raw_requirement_text'].apply(lambda x: len(x.split()))\n",
            "\n",
            "print(\"=== 문자 길이 통계 ===\")\n",
            "print(df['char_len'].describe(percentiles=[0.5, 0.9, 0.95, 0.99]))\n",
            "\n",
            "# 히스토그램 시각화\n",
            "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
            "sns.histplot(df['char_len'], bins=40, kde=True, ax=axes[0], color='skyblue')\n",
            "axes[0].set_title('본문 문자 길이 분포')\n",
            "axes[0].set_xlabel('문자 수')\n",
            "\n",
            "sns.boxplot(data=df, x='char_len', y='document_id', ax=axes[1], palette='coolwarm')\n",
            "axes[1].set_title('문서별 본문 문자 길이 박스플롯')\n",
            "axes[1].set_xlabel('문자 수')\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    })

    # 셀 6: 중첩표 및 특이 ID 분석
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 4. 중첩표(' | ') 및 특이 ID 검사\n",
            "df['has_nested_table'] = df['raw_requirement_text'].apply(lambda x: ' | ' in x)\n",
            "df['id_mismatch'] = df['requirement_id'] != df['source_requirement_id']\n",
            "\n",
            "print(f\"중첩표 포함 행 수: {df['has_nested_table'].sum()} 건\")\n",
            "print(f\"Canonical ID != Source ID 불일치 행 수: {df['id_mismatch'].sum()} 건\")\n",
            "\n",
            "if df['id_mismatch'].sum() > 0:\n",
            "    display(df[df['id_mismatch']][['requirement_uid', 'requirement_id', 'source_requirement_id', 'requirement_name']])"
        ]
    })

    notebook = {
        "cells": cells,
        "metadata": {
            "language_info": {
                "name": "python",
                "version": "3.10"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    return notebook


def main():
    root_dir = Path(__file__).resolve().parent.parent
    dataset_path = root_dir / "data" / "processed" / "requirements_v0.2.0.jsonl"
    
    if not dataset_path.exists():
        print(f"오류: 데이터셋 파일을 찾을 수 없습니다 -> {dataset_path}")
        sys.exit(1)
        
    records = load_dataset(str(dataset_path))
    eda_result = analyze_dataset(records)
    
    # 1. JSON 보고서 저장
    reports_dir = root_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / "eda_v0.2.0.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(eda_result, f, ensure_ascii=False, indent=2)
    print(f"JSON 보고서 생성 완료: {json_path}")
    
    # 2. Markdown 보고서 저장
    md_report = generate_markdown_report(eda_result)
    md_path = reports_dir / "eda_v0.2.0.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_report)
    print(f"Markdown 보고서 생성 완료: {md_path}")
    
    # 3. Jupyter Notebook (.ipynb) 저장
    notebook_dir = root_dir / "notebooks"
    notebook_dir.mkdir(parents=True, exist_ok=True)
    nb_path = notebook_dir / "eda_v0.2.0.ipynb"
    nb_data = generate_jupyter_notebook(eda_result)
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb_data, f, ensure_ascii=False, indent=2)
    print(f"Jupyter Notebook 생성 완료: {nb_path}")


if __name__ == "__main__":
    main()
