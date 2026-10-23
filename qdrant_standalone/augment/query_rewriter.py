"""
query_rewriter.py — 口语查询 → 书面语改写（检索前预处理）

解决的问题（来自 40 条评测的真实失败）：
    "车子不好卖了想下架"  检索不到  "市场衰退…启动退市评估"
    "用户喜不喜欢"        检索不到  "价值认可度、试骑访谈、问卷调研"
    embedding 对零字面重合的口语→书面转换力不从心，sparse 通道直接作废。
    检索前用 LLM 把查询改写成制度文档的措辞，两个通道就都活了。

设计要点：
    - fail-open：LLM 不可用 / 超时 / 返回为空时原样返回查询，检索绝不因改写挂掉
    - 进程内缓存：同一条查询只改写一次
    - key 不落代码：环境变量 DEEPSEEK_API_KEY 优先，其次读 LightRAG 部署的 .env

使用方式:
    from augment.query_rewriter import rewrite_query
    q = rewrite_query("车子不好卖了想下架怎么办")
    # → "产品市场衰退时的退市评估与退市流程"
"""

import os
import json
import logging
import urllib.request
import urllib.error

logger = logging.getLogger(__name__)

LLM_HOST = "https://api.deepseek.com/v1"
LLM_MODEL = "deepseek-flash"
TIMEOUT = 60   # flash 推理模型较慢

# LightRAG 部署的 .env（借它的 key，避免重复配置）
LIGHTRAG_ENV = "/Users/trinxai/LightRAG/.env"

_PROMPT = (
    "你是企业制度知识库的查询改写助手。把用户的口语化提问改写成企业制度文档"
    "中可能出现的书面措辞，用于向量检索。要求：\n"
    "1. 输出只有改写后的查询，一句话，不要解释\n"
    "2. 保留原意，补充可能的书面同义词（如：下架→退市/停售，喜不喜欢→认可度/接受度）\n"
    "3. 不要回答问题本身\n\n"
    "用户提问：{query}"
)

_cache: dict = {}


def _load_api_key() -> str:
    key = os.environ.get("DEEPSEEK_API_KEY")
    if key:
        return key
    try:
        with open(LIGHTRAG_ENV, encoding="utf-8") as f:
            for line in f:
                if line.startswith(("LLM_BINDING_API_KEY=", "OPENAI_API_KEY=")):
                    return line.split("=", 1)[1].strip()
    except OSError:
        pass
    return ""


def rewrite_query(query: str, enabled: bool = True) -> str:
    """改写查询；任何失败都返回原查询。"""
    if not enabled:
        return query
    if query in _cache:
        return _cache[query]

    key = _load_api_key()
    if not key:
        logger.warning("未找到 DEEPSEEK_API_KEY，跳过查询改写")
        return query

    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": _PROMPT.format(query=query)}],
        "temperature": 0.0,
        "max_tokens": 600,   # flash 推理模型：给 reasoning 留余量
    }
    req = urllib.request.Request(
        f"{LLM_HOST}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        rewritten = body["choices"][0]["message"]["content"].strip().strip('"')
        if not rewritten or len(rewritten) > 200:
            raise ValueError(f"改写结果异常: {rewritten!r}")
    except Exception as e:
        logger.warning("查询改写失败，使用原查询: %s", e)
        rewritten = query

    _cache[query] = rewritten
    if rewritten != query:
        logger.info("查询改写: %s → %s", query, rewritten)
    return rewritten
