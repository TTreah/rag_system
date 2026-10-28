"""
config.py — 统一加载本地 .env 配置（正本）

所有可能随部署环境变化、或不应写入代码的值（API key、服务地址、模型名、
源目录、collection 名）都放在本目录的 .env 中；真实环境变量优先于 .env。
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.environ.get("RAG_ENV_FILE") or os.path.join(BASE_DIR, ".env")


def load_env(path: str = ENV_PATH) -> None:
    """把 .env 读入 os.environ（不覆盖已存在的真实环境变量）。"""
    try:
        with open(path, encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[7:].strip()
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key:
                    os.environ.setdefault(key, value)
    except OSError:
        pass


load_env()

# ── DeepSeek（查询改写 / QA 增强 / 资产描述 / LightRAG 抽取）──
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-flash")
DEEPSEEK_ENV_FILE = os.environ.get("DEEPSEEK_ENV_FILE", "")

# ── 本地服务 ──
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
MINERU_URL = os.environ.get("MINERU_URL", "http://localhost:8000/file_parse")
LIGHTRAG_URL = os.environ.get("LIGHTRAG_URL", "http://localhost:9621")
LIGHTRAG_CEO_URL = os.environ.get("LIGHTRAG_CEO_URL", "http://localhost:9622")
LIGHTRAG_API_KEY = os.environ.get("LIGHTRAG_API_KEY", "")
LIGHTRAG_EMBED_PORT = int(os.environ.get("LIGHTRAG_EMBED_PORT", "18001"))

# ── 本地模型 ──
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "Qwen/Qwen3-VL-Embedding-2B")
RERANK_MODEL = os.environ.get("RERANK_MODEL", "Qwen/Qwen3-VL-Reranker-2B")
VLM_MODEL = os.environ.get("VLM_MODEL", "Qwen/Qwen3-VL-2B-Instruct")

# ── 数据源与目标库 ──
SOURCE_DIR = os.environ.get(
    "SOURCE_DIR",
    "/Volumes/AI-Project/QLD-UAT/KnowledgeBase/KB-Product/document-rule",
)
COLLECTION = os.environ.get("COLLECTION", "news_collection")

# ── 测试用例目录（评测脚本使用）──
TEST_CASE_DIR = os.environ.get(
    "TEST_CASE_DIR", "/Users/trinxai/Documents/pythonfile/测试用例"
)
