"""
eval_case_utils.py — 检索评测的公共工具

被 eval_qdrant_cases.py / eval_lightrag_cases.py 共用：
  - load_cases:      从测试用例 docx 的表格里读出用例（用例与评测永不脱节）
  - point_coverage:  期望要点在实际输出里的覆盖率（中文子串 + 数字兜底）
  - write_report:    把汇总指标 + 逐条结果写成 docx 报告
"""

import re
import os
from typing import List, Dict, Any

from docx import Document
from docx.shared import Pt
from docx.oxml.ns import qn


def load_cases(docx_path: str) -> List[Dict[str, str]]:
    """读测试用例 docx 的第一张表 → [{id, query, expect_file, expect_points, evidence}]"""
    doc = Document(docx_path)
    table = doc.tables[0]
    header = [c.text.strip() for c in table.rows[0].cells]
    ev_idx = next((i for i, h in enumerate(header) if "依据块" in h), None)
    cases = []
    for row in table.rows[1:]:  # 跳过表头
        cells = [c.text.strip() for c in row.cells]
        if len(cells) < 4 or not cells[1]:
            continue
        cases.append({
            "id": cells[0],
            "query": cells[1],
            "expect_file": cells[2],
            "expect_points": cells[3],
            "evidence": cells[ev_idx] if ev_idx is not None and ev_idx < len(cells) else "",
        })
    return cases


def _normalize(s: str) -> str:
    """去标点空白和箭头/连接符，只做纯粹的内容比对"""
    return re.sub(r"[\s，。、；：（）()【】《》\"'·.!?,;:~\-—*→←↔+*/=×]+", "", s)


def _number_tokens(s: str) -> List[str]:
    """抽出所有数字（不含连接符）。范围写法不敏感：
    "1.08~1.15"、"1.08-1.15"、"1.08 至 1.15" 都抽出 ['1.08', '1.15']，
    比对的是数值本身，不是书写格式。"""
    return re.findall(r"\d+(?:\.\d+)?", s)


def _bigrams(s: str) -> set:
    return {s[i:i + 2] for i in range(len(s) - 1)}


def point_hit(point: str, text: str) -> bool:
    """单个期望要点是否命中。

    三级判定（从严到宽）：
    1. 归一化整句子串命中 → 直接算中（原文摘录场景）
    2. 要点含数字：数字 token 必须全部精确出现（数字是事实底线，
       改写的答案数字错了就是错了），同时 bigram 重合 ≥ 0.35
    3. 纯文字要点：bigram 重合 ≥ 0.5 —— LLM 改写会换措辞换语序，
       但两个字的片段大多保得住；这个阈值对"同义改写"宽容、
       对"答非所问"仍然有效
    """
    np_, nt = _normalize(point), _normalize(text)
    if np_ and np_ in nt:
        return True
    bg = _bigrams(np_)
    if not bg:
        return False
    ratio = sum(1 for g in bg if g in nt) / len(bg)
    nums = _number_tokens(point)
    if nums:
        return all(n in text for n in nums) and ratio >= 0.35
    return ratio >= 0.5


def point_coverage(expect_points: str, text: str) -> "tuple[float, int, int]":
    """(覆盖率, 命中数, 要点总数)"""
    points = [p.strip() for p in re.split(r"[；;]", expect_points) if p.strip()]
    if not points:
        return 1.0, 0, 0
    hits = sum(1 for p in points if point_hit(p, text))
    return hits / len(points), hits, len(points)


def file_stem(name: str) -> str:
    return os.path.splitext(os.path.basename(name))[0].strip()


def write_report(path: str, title: str, subtitle: str,
                 metrics: List["tuple[str, str]"],
                 headers: List[str], rows: List[List[str]],
                 conclusion: str) -> None:
    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(10.5)
    style.element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")

    doc.add_heading(title, level=0)
    doc.add_paragraph(subtitle)

    doc.add_heading("一、汇总指标", level=1)
    t = doc.add_table(rows=1, cols=2)
    t.style = "Table Grid"
    t.rows[0].cells[0].text = "指标"
    t.rows[0].cells[1].text = "结果"
    for run in t.rows[0].cells[0].paragraphs[0].runs:
        run.font.bold = True
    for run in t.rows[0].cells[1].paragraphs[0].runs:
        run.font.bold = True
    for k, v in metrics:
        c = t.add_row().cells
        c[0].text = k
        c[1].text = v

    doc.add_heading("二、逐条结果", level=1)
    t2 = doc.add_table(rows=1, cols=len(headers))
    t2.style = "Table Grid"
    for i, h in enumerate(headers):
        cell = t2.rows[0].cells[i]
        cell.text = h
        for run in cell.paragraphs[0].runs:
            run.font.bold = True
    for r in rows:
        cells = t2.add_row().cells
        for i, v in enumerate(r):
            cells[i].text = str(v)

    doc.add_heading("三、结论", level=1)
    for line in conclusion.split("\n"):
        if line.strip():
            doc.add_paragraph(line.strip())

    doc.save(path)
    print(f"报告已生成: {path}")
