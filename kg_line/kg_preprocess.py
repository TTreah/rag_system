"""
kg_preprocess.py — 知识图谱入库前的文本预处理

针对评测中发现的问题：数字规则类条款（"高溢价 MVP 投放 10-20 台"、
"零售价=批发价×1.20~1.45"）在 LLM 实体抽取时容易被压缩丢失——
它们不是实体间关系，而是条件分支细节，图谱结构对它们天然弱势。

对策：入库前把含"数字+单位"的条款段落标记为独立小节。两个作用：
    1. 服务端按 token 分块时，前后空行 + 标记让条款更可能独立成块，
       不被上下文稀释
    2. 【关键数值条款】标记对 LLM 抽取是显著性信号，实体描述里更可能
       保留具体数字

同时给全文前置文档名头（与 Qdrant 线的上下文头策略对齐）。
"""

import re

# 数字 + 计量单位，才算"数值条款"（避免把纯编号段落也标出来）
_NUM_UNIT = re.compile(
    r"\d+(?:\.\d+)?\s*(?:~|～|-)?\s*(?:\d+(?:\.\d+)?)?\s*"
    r"(?:台|元|万元|元/|%|％|个|个工作日|工作日|天|小时|倍|条|款|项|类|级|层)"
)

MARKER = "【关键数值条款】"


def enhance_numeric_clauses(text: str, doc_title: str = "") -> str:
    """标记数值条款段落 + 前置文档名头。"""
    paragraphs = re.split(r"\n{2,}", text)
    out = []
    marked = 0
    for p in paragraphs:
        stripped = p.strip()
        if stripped and _NUM_UNIT.search(stripped) and MARKER not in stripped:
            # 段落较长时只标记，不拆分——保持上下文完整，靠空行促进独立成块
            out.append(f"{MARKER}\n{stripped}")
            marked += 1
        else:
            out.append(p)

    body = "\n\n".join(out)
    header = f"【文档：{doc_title}】\n\n" if doc_title else ""
    return header + body
