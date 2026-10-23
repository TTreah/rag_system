"""
qa_augment.py — 入库时为 chunk 生成假设问题（QA 对增强索引）

解决的问题（40 条评测的残余失败全是这一类）：
    "车子不好卖了想下架"  vs  文档原文 "市场衰退…启动退市评估"
    "小孩上学骑的车"      vs  文档原文 "学生通勤流量款"
    查询和文档之间隔着一层口语→书面语翻译，向量模型跨不过去。
    检索侧的改写（query_rewriter）能缓解但会丢词；入库侧直接给每个
    chunk 配上"用户可能怎么问"，口语查询命中假设问题，把 chunk 带出来。

做法：入库时用 LLM 为每个 chunk 生成 3 个口语化假设问题，
拼接在 chunk 文本后一起向量化（payload 里仍存原文，展示不受影响）。
成本：每个 chunk 一次小 LLM 调用，仅入库时发生一次。
"""

import os
import json
import logging
import urllib.request

from augment.query_rewriter import _load_api_key, LLM_HOST, LLM_MODEL

logger = logging.getLogger(__name__)

_PROMPT = (
    "以下是企业制度文档的一个片段。请站在一线员工的角度，生成 {n} 个"
    "他们可能用来查找这段内容的口语化提问（就像平时说话那样，"
    "不要用制度文件的正式措辞）。每行一个问题，只输出问题本身。\n\n"
    "文档片段：\n{text}"
)


def generate_questions(text: str, n: int = 3, timeout: int = 90) -> list:
    """为一个 chunk 生成 n 个口语化假设问题。失败返回 []（不阻塞入库）。"""
    key = _load_api_key()
    if not key:
        logger.warning("未找到 DEEPSEEK_API_KEY，跳过假设问题生成")
        return []

    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user",
                      "content": _PROMPT.format(n=n, text=text[:1500])}],
        "temperature": 0.7,
        "max_tokens": 1500,  # flash 推理模型：给 reasoning 留余量
    }
    req = urllib.request.Request(
        f"{LLM_HOST}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        raw = body["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.warning("假设问题生成失败（跳过该 chunk 的增强）: %s", e)
        return []

    # 解析：按行拆分，剥掉序号/项目符号
    import re
    questions = []
    for line in raw.splitlines():
        line = re.sub(r"^[\s\-\*\d\.、（\)）]+", "", line).strip()
        if line and len(line) >= 4:
            questions.append(line)
    return questions[:n]
