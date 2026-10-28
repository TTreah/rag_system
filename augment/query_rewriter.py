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

from config import (
    DEEPSEEK_BASE_URL as LLM_HOST,
    DEEPSEEK_MODEL as LLM_MODEL,
    DEEPSEEK_API_KEY,
    DEEPSEEK_ENV_FILE,
)
TIMEOUT = 60   # flash 推理模型较慢

_PROMPT = (
    "你是企业制度知识库的查询改写助手。把用户的口语化提问改写成企业制度文档"
    "中可能出现的书面措辞，用于向量检索。要求：\n"
    "1. 输出只有改写后的查询，一句话，不要解释\n"
    "2. 保留原意，补充可能的书面同义词（如：下架→退市/停售，喜不喜欢→认可度/接受度）\n"
    "3. 不要回答问题本身\n\n"
    "用户提问：{query}"
)

_cache: dict = {}
_CACHE_PATH = os.path.expanduser("~/.cache/rag_query_rewrites.json")
_CACHE_VERSION = f"{LLM_MODEL}|v2"
_cache_loaded = False


def _load_persistent_cache():
    global _cache_loaded
    if _cache_loaded:
        return
    _cache_loaded = True
    try:
        with open(_CACHE_PATH, encoding="utf-8") as f:
            data = json.load(f)
        if data.get("version") == _CACHE_VERSION:
            _cache.update(data.get("items", {}))
    except (OSError, ValueError):
        pass


def _save_persistent_cache():
    if os.environ.get("RAG_REWRITE_CACHE", "1") == "0":
        return
    try:
        os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
        tmp = _CACHE_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"version": _CACHE_VERSION, "items": _cache}, f,
                      ensure_ascii=False, indent=1)
        os.replace(tmp, _CACHE_PATH)
    except OSError:
        pass


def _load_api_key() -> str:
    # config 已加载本目录 .env；真实环境变量优先
    key = os.environ.get("DEEPSEEK_API_KEY") or DEEPSEEK_API_KEY
    if key:
        return key
    # 可选兜底：指向另一个含 LLM_BINDING_API_KEY/OPENAI_API_KEY 的 .env
    if DEEPSEEK_ENV_FILE:
        try:
            with open(DEEPSEEK_ENV_FILE, encoding="utf-8") as f:
                for line in f:
                    if line.startswith(("LLM_BINDING_API_KEY=", "OPENAI_API_KEY=")):
                        return line.split("=", 1)[1].strip().strip('"').strip("'")
        except OSError:
            pass
    return ""


def rewrite_query(query: str, enabled: bool = True) -> str:
    """改写查询；任何失败都返回原查询。"""
    if not enabled:
        return query
    _load_persistent_cache()
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
    _save_persistent_cache()
    if rewritten != query:
        logger.info("查询改写: %s → %s", query, rewritten)
    return rewritten
