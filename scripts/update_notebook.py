#!/usr/bin/env python3
"""
eda_v0.2.0.ipynb 주피터 노트북 셀 구조 갱신 스크립트
"""

import json
import os
from pathlib import Path

def update_notebook():
    nb_path = Path("notebooks/eda_v0.2.0.ipynb")
    if not nb_path.exists():
        return
        
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # 마크다운 및 코드 셀 정돈
    # 셀 4(정규화 대분류)가 맨 처음에 크게 보이도록 강조
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            src = "".join(cell["source"])
            if "normalized_type" in src and "sns.barplot" in src:
                # 정규화 차트 색상 및 크기 더욱 강조
                cell["source"] = [
                    "# 2. 정규화 요구사항 유형별 분포 (11개 통합 대분류)\n",
                    "norm_type_counts = df['normalized_type'].value_counts().reset_index()\n",
                    "norm_type_counts.columns = ['normalized_type', 'count']\n",
                    "norm_type_counts['ratio_pct'] = (norm_type_counts['count'] / len(df) * 100).round(2)\n",
                    "display(norm_type_counts)\n",
                    "\n",
                    "# 시각화 (정규화 대분류)\n",
                    "plt.figure(figsize=(10, 5))\n",
                    "ax = sns.barplot(data=norm_type_counts, x='count', y='normalized_type', palette='crest')\n",
                    "plt.title('요구사항 유형별 분포 (11개 정규화 대분류)', fontsize=14, fontweight='bold')\n",
                    "plt.xlabel('행 수 (건)')\n",
                    "plt.ylabel('정규화 대분류 유형')\n",
                    "for p in ax.patches:\n",
                    "    width = p.get_width()\n",
                    "    ax.annotate(f'{int(width)}건', (width + 2, p.get_y() + p.get_height() / 2.), va='center')\n",
                    "plt.tight_layout()\n",
                    "plt.show()\n"
                ]

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=2)
    print("Updated eda_v0.2.0.ipynb structure")

if __name__ == "__main__":
    update_notebook()
