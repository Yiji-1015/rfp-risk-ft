#!/usr/bin/env python3
"""
RFP 요구사항 동결 데이터셋 v0.2.0 EDA 스크립트

수행 작업:
1. 문서별 및 원문/정규화 요구사항 유형별 행 수 집계
2. 본문 문자 길이, 단어 수, 공백 분할 토큰 길이 분포 통계 (min, max, mean, median, P90, P95, P99)
3. 중첩표(' | ') 포함 행, 승인된 특이 ID(canonical != source_id) 원문 예외 행, 최단/최장 본문 사례 추출
4. JSON 결과(reports/current/eda_v0.2.0.json), Markdown 보고서(reports/current/eda_v0.2.0.md), 
   Jupyter Notebook(notebooks/eda_v0.2.0.ipynb) 자동 생성
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any


def normalize_requirement_type(raw_type: str) -> str:
    """원문 요구사항 유형을 대표 대분류로 정규화 매핑"""
    if not raw_type or raw_type == "None" or raw_type is None:
        return "미지정 (None)"
    
    t = str(raw_type).strip()
    
    if any(k in t for k in ["기능", "SFR", "AI 활용 업무", "AI 기반 솔루션", "그룹웨어", "서비스"]):
        return "기능 요구사항"
    if "성능" in t or "PER" in t:
        return "성능 요구사항"
    if "보안" in t or "SER" in t:
        return "보안 요구사항"
    if "데이터" in t or "DAR" in t or "ECM" in t:
        return "데이터 요구사항"
    if "품질" in t or "QUR" in t:
        return "품질 요구사항"
    if "인터페이스" in t or "INR" in t:
        return "인터페이스 요구사항"
    if "제약" in t or "COR" in t:
        return "제약사항"
    if "테스트" in t or "TER" in t:
        return "테스트 요구사항"
    if any(k in t for k in ["장비", "인프라", "시스템", "AI 플랫폼 및 인프라"]):
        return "인프라·장비 요구사항"
    if any(k in t for k in ["프로젝트", "PMR", "PSR", "컨설팅", "CNR", "CUR", "거버넌스", "안전"]):
        return "프로젝트 관리·지원 요구사항"
        
    return "기타"


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
    raw_type_counts = {}
    norm_type_counts = {}
    doc_type_counts = {}
    
    char_lengths = []
    word_counts = []
    
    doc_char_lengths = {}
    
    id_mismatches = []
    nested_table_records = []
    
    for r in records:
        doc_id = r.get("document_id", "unknown")
        raw_req_type = str(r.get("requirement_type", "None"))
        norm_req_type = normalize_requirement_type(raw_req_type)
        
        req_uid = r.get("requirement_uid", "")
        req_id = r.get("requirement_id", "")
        source_req_id = r.get("source_requirement_id", "")
        raw_text = r.get("raw_requirement_text", "")
        
        # 문서별 집계
        doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1
        
        # 원문 및 정규화 유형별 집계
        raw_type_counts[raw_req_type] = raw_type_counts.get(raw_req_type, 0) + 1
        norm_type_counts[norm_req_type] = norm_type_counts.get(norm_req_type, 0) + 1
        
        # 문서 x 정규화 유형 교차 집계
        if doc_id not in doc_type_counts:
            doc_type_counts[doc_id] = {}
        doc_type_counts[doc_id][norm_req_type] = doc_type_counts[doc_id].get(norm_req_type, 0) + 1
        
        # 본문 길이
        c_len = len(raw_text)
        w_len = len(raw_text.split())
        
        char_lengths.append(c_len)
        word_counts.append(w_len)
        
        if doc_id not in doc_char_lengths:
            doc_char_lengths[doc_id] = []
        doc_char_lengths[doc_id].append(c_len)
        
        # 승인된 원문 ID 예외 검사 (Canonical ID != Source ID)
        if req_id != source_req_id:
            id_mismatches.append({
                "requirement_uid": req_uid,
                "document_id": doc_id,
                "requirement_id": req_id,
                "source_requirement_id": source_req_id,
                "requirement_name": r.get("requirement_name", ""),
                "note": "원문 이중 하이픈(CUR-CM--001) 추적용 승인 예외 보존건"
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
        "raw_type_counts": raw_type_counts,
        "normalized_type_counts": norm_type_counts,
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
    lines.append(f"- **승인된 원문 예외 ID (Canonical != Source) 행 수**: {len(eda_result['id_mismatches'])} 행 (의도적 원문 추적 보존건)")
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

    # 2. 정규화 요구사항 유형별 분포
    lines.append("## 2. 요구사항 유형별 분포 (정규화 대분류)")
    lines.append("")
    lines.append("| 정규화 유형 (normalized_requirement_type) | 행 수 | 비율 (%) |")
    lines.append("|---|---:|---:|")
    for req_type, count in sorted(eda_result['normalized_type_counts'].items(), key=lambda x: x[1], reverse=True):
        ratio = (count / eda_result['total_records']) * 100
        lines.append(f"| `{req_type}` | {count:,} | {ratio:.1f}% |")
    lines.append("")

    lines.append("### 원문 표기 요구사항 유형 분포 (Raw Types Top 15)")
    lines.append("")
    lines.append("| 원문 표기 유형 | 행 수 | 비율 (%) |")
    lines.append("|---|---:|---:|")
    sorted_raw = sorted(eda_result['raw_type_counts'].items(), key=lambda x: x[1], reverse=True)
    for raw_type, count in sorted_raw[:15]:
        ratio = (count / eda_result['total_records']) * 100
        lines.append(f"| `{raw_type}` | {count:,} | {ratio:.1f}% |")
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
    lines.append("## 4. 승인된 원문 예외 및 특이 사례 분석")
    lines.append("")
    lines.append("### 4.1 승인된 원문 예외 ID (Canonical ID != Source ID)")
    lines.append("")
    lines.append("> 이 항목은 결함이 아니라 원문 오타/불일치의 추적성을 보존하기 위해 `requirements_v0.2.0` 데이터셋 및 `extraction_freeze_v0.2.0.md`에 명시적으로 의도하여 남긴 승인 예외(Policy Resolved) 항목입니다.")
    lines.append("")
    if eda_result['id_mismatches']:
        lines.append("| requirement_uid | Canonical `requirement_id` | Source `source_requirement_id` | 요구사항명 | 비고 |")
        lines.append("|---|---|---|---|---|")
        for item in eda_result['id_mismatches']:
            lines.append(f"| `{item['requirement_uid']}` | `{item['requirement_id']}` | `{item['source_requirement_id']}` | {item['requirement_name']} | {item.get('note', '')} |")
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
            "이 노트북은 `data/processed/requirements_v0.2.0.jsonl` (총 1,024행) 동결 데이터셋의 문서별/유형별 분포, 본문 길이, 중첩표 및 승인된 원문 예외 ID를 탐색하고 시각화합니다."
        ]
    })
    
    # 셀 2: 환경 설정 및 임포트 & 정규화 함수
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
            "\n",
            "def normalize_req_type(raw_type):\n",
            "    if not raw_type or raw_type == 'None' or str(raw_type) == 'nan':\n",
            "        return '미지정 (None)'\n",
            "    t = str(raw_type).strip()\n",
            "    if any(k in t for k in ['기능', 'SFR', 'AI 활용 업무', 'AI 기반 솔루션', '그룹웨어', '서비스']):\n",
            "        return '기능 요구사항'\n",
            "    if '성능' in t or 'PER' in t:\n",
            "        return '성능 요구사항'\n",
            "    if '보안' in t or 'SER' in t:\n",
            "        return '보안 요구사항'\n",
            "    if '데이터' in t or 'DAR' in t or 'ECM' in t:\n",
            "        return '데이터 요구사항'\n",
            "    if '품질' in t or 'QUR' in t:\n",
            "        return '품질 요구사항'\n",
            "    if '인터페이스' in t or 'INR' in t:\n",
            "        return '인터페이스 요구사항'\n",
            "    if '제약' in t or 'COR' in t:\n",
            "        return '제약사항'\n",
            "    if '테스트' in t or 'TER' in t:\n",
            "        return '테스트 요구사항'\n",
            "    if any(k in t for k in ['장비', '인프라', '시스템', 'AI 플랫폼 및 인프라']):\n",
            "        return '인프라·장비 요구사항'\n",
            "    if any(k in t for k in ['프로젝트', 'PMR', 'PSR', '컨설팅', 'CNR', 'CUR', '거버넌스', '안전']):\n",
            "        return '프로젝트 관리·지원 요구사항'\n",
            "    return '기타'\n",
            "\n",
            "df['normalized_type'] = df['requirement_type'].apply(normalize_req_type)\n",
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

    # 셀 4: 정규화 요구사항 유형별 분포
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 2. 정규화 요구사항 유형별 분포 (통합 대분류)\n",
            "norm_type_counts = df['normalized_type'].value_counts().reset_index()\n",
            "norm_type_counts.columns = ['normalized_type', 'count']\n",
            "norm_type_counts['ratio_pct'] = (norm_type_counts['count'] / len(df) * 100).round(2)\n",
            "display(norm_type_counts)\n",
            "\n",
            "# 시각화 (정규화 대분류)\n",
            "plt.figure(figsize=(10, 5))\n",
            "sns.barplot(data=norm_type_counts, x='count', y='normalized_type', palette='crest')\n",
            "plt.title('정규화 요구사항 유형별 분포 (통합 대분류)')\n",
            "plt.xlabel('행 수')\n",
            "plt.ylabel('정규화 유형')\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    })

    # 셀 5: 원문 표기 요구사항 유형 분포
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 원문 표기 유형 Top 15 시각화 (세부 표기 차이 확인용)\n",
            "raw_type_counts = df['requirement_type'].value_counts().head(15).reset_index()\n",
            "raw_type_counts.columns = ['raw_requirement_type', 'count']\n",
            "\n",
            "plt.figure(figsize=(10, 6))\n",
            "sns.barplot(data=raw_type_counts, x='count', y='raw_requirement_type', palette='magma')\n",
            "plt.title('원문 표기 요구사항 유형 분포 (Top 15 Raw Types)')\n",
            "plt.xlabel('행 수')\n",
            "plt.ylabel('원문 표기 유형')\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    })

    # 셀 6: 본문 길이 분포 (문자 수 & 단어 수)
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

    # 셀 7: 승인된 원문 예외 ID 및 중첩표 분석
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 4. 승인된 원문 ID 예외(Policy Resolved) 및 중첩표(' | ') 검사\n",
            "df['has_nested_table'] = df['raw_requirement_text'].apply(lambda x: ' | ' in x)\n",
            "df['id_mismatch'] = df['requirement_id'] != df['source_requirement_id']\n",
            "\n",
            "print(f\"중첩표 포함 행 수: {df['has_nested_table'].sum()} 건\")\n",
            "print(f\"승인된 원문 예외 ID (Canonical ID != Source ID) 행 수: {df['id_mismatch'].sum()} 건 (의도된 추적용 보존)\")\n",
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
    root_dir = Path(__file__).resolve().parents[2]
    dataset_path = root_dir / "data" / "processed" / "requirements_v0.2.0.jsonl"
    
    if not dataset_path.exists():
        print(f"오류: 데이터셋 파일을 찾을 수 없습니다 -> {dataset_path}")
        sys.exit(1)
        
    records = load_dataset(str(dataset_path))
    eda_result = analyze_dataset(records)
    
    # 1. JSON 보고서 저장
    reports_dir = root_dir / "reports" / "current"
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
