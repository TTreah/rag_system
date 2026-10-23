"""
asset_describer.py — 为文档中的图片/表格资产生成检索用描述文字

设计（对应用户需求）：表格、图片落盘保存，描述文字入向量库并关联文件路径；
检索命中描述时，把资产文件路径一并返回，供上级智能体取用素材。

限制（如实说明）：当前没有部署生成式视觉模型，图片"看不到内容"，
描述基于图片所在页的上下文文本推断生成；表格描述则基于真实表格内容。
后续部署 Qwen3-VL-Instruct 等生成式 VLM 后，只需替换 describe_image。
"""

import json
import logging
import urllib.request

from augment.query_rewriter import _load_api_key, LLM_HOST, LLM_MODEL

logger = logging.getLogger(__name__)

_TABLE_PROMPT = (
    "这是从企业文档《{title}》（{label}）中提取的一个表格。\n"
    "表格内容：\n{table}\n\n"
    "请写一段 80 字以内的检索用描述，说明这是什么表格、包含哪些关键信息"
    "（关键列名、重要数字）。只输出描述文字。"
)

_IMAGE_PROMPT = (
    "企业文档《{title}》的{label}处嵌入了一张图片。\n"
    "该页的文字上下文：\n{context}\n\n"
    "请根据上下文推断这张图片可能展示的内容，写一段 60 字以内的检索用描述"
    "（说明图片位置、可能的主题）。只输出描述文字，不要说明这是推断。"
)


def _ask_llm(prompt: str, timeout: int = 120) -> str:
    # flash 是推理模型：reasoning 占大头且慢，30s 超时会被误杀
    key = _load_api_key()
    if not key:
        return ""
    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 1500,  # flash 是推理模型，reasoning 会占用 token
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
        return body["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.warning("资产描述生成失败: %s", e)
        return ""


def describe_table(title: str, label: str, context: str, table_text: str) -> str:
    desc = _ask_llm(_TABLE_PROMPT.format(
        title=title, label=label, table=table_text[:800]))
    if not desc:
        # fail-open：LLM 不可用时退化为模板描述，保证资产仍能被检索到
        first_row = table_text.split("\n", 1)[0][:80]
        desc = f"《{title}》{label}的表格，首行：{first_row}"
    return desc


def describe_image(title: str, label: str, context: str) -> str:
    desc = _ask_llm(_IMAGE_PROMPT.format(
        title=title, label=label, context=context[:600]))
    if not desc:
        desc = f"《{title}》{label}处嵌入的图片。"
    return desc
