"""
run_news.py — 文档批量向量化入库脚本
扫描指定目录 → 解析所有支持类型的文件 → 分块 → 向量化 → Qdrant 入库。

使用方式:
    python run_news.py              # 完整流程（需要 Qdrant 运行中）
    python run_news.py --local      # 仅本地：解析 + 分块 + 向量化，跳过 Qdrant
    python run_news.py --search "关键词"  # 检索已入库数据

适用场景:
    - Hermes agent 定时任务：定期扫描共享盘，增量更新向量库
    - 手动批量入库：一次性导入大量文档
"""

import os

# ============================================================
#  可调参数（按实际环境修改）
# ============================================================
SOURCE_DIR = r"D:\pythonfile\Qdrant\PDF"     # 要扫描的目录（支持混合文件类型）
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "news_collection"          # 目标 Qdrant collection

# 分块参数
CHUNK_STRATEGY = "recursive"                 # fixed / recursive / sentence / markdown
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64

# 嵌入参数
EMBEDDING_MODEL = "bge-m3"                  # BGE-M3: 1024维，多语言
EMBEDDING_BATCH_SIZE = 16

# 检索参数
TOP_K = 10

# ============================================================

from vectorize_pipeline import VectorizePipeline, PipelineConfig


def run_ingest():
    """完整流程：扫描 → 解析 → 分块 → 向量化 → Qdrant 入库"""
    config = PipelineConfig(
        chunk_strategy=CHUNK_STRATEGY,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        embedding_model=EMBEDDING_MODEL,
        embedding_use_flag=True,
        embedding_batch_size=EMBEDDING_BATCH_SIZE,
        qdrant_url=QDRANT_URL,
        qdrant_collection=COLLECTION_NAME,
        qdrant_force_recreate=True,
    )

    pipeline = VectorizePipeline(config)
    results = pipeline.run([SOURCE_DIR], store_in_qdrant=True)
    _print_summary(results)
    return results


def _print_summary(results):
    print("\n" + "=" * 60)
    print(f"{'Ingest Summary':^60}")
    print("=" * 60)
    total_chunks = 0
    for r in results:
        fname = os.path.basename(r["file"])
        fname = fname.encode("gbk", errors="replace").decode("gbk")
        chunks = r["chunk_count"]
        total_chunks += chunks
        print(f"  {fname}")
        print(f"    type={r['file_type']}, chunks={chunks}")
    print(f"\n  Total: {len(results)} files, {total_chunks} chunks")
    print(f"  Source: {SOURCE_DIR}")
    print(f"  Qdrant: {QDRANT_URL} / {COLLECTION_NAME}")


# ============================================================
if __name__ == "__main__":
    run_ingest()
