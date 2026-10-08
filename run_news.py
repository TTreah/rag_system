"""
run_news.py — 文档批量向量化入库脚本
扫描指定目录 → 解析所有支持类型的文件 → 分块 → 向量化 → Qdrant 入库。

使用方式:
    python run_news.py                    # 增量入库（需要 Qdrant 运行中）
    python run_news.py --recreate         # 先删除 collection 再全量重建
    python run_news.py --local            # 仅本地：解析 + 分块 + 向量化，跳过 Qdrant
    python run_news.py --search "关键词"   # 检索已入库数据，不执行入库

适用场景:
    - Hermes agent 定时任务：定期扫描共享盘，增量更新向量库
    - 手动批量入库：一次性导入大量文档
"""

import os
import sys
import argparse

# Windows 控制台统一 UTF-8 输出，避免中文/意大利语文件名乱码（VSCode 终端默认 UTF-8）
if sys.stdout and getattr(sys.stdout, "encoding", "").lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

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


def _safe(s) -> str:
    """输出兜底（UTF-8 下无需转码，仅防御 None/非字符串）。"""
    if s is None:
        return ""
    return str(s)


def build_config(recreate: bool = False) -> PipelineConfig:
    return PipelineConfig(
        chunk_strategy=CHUNK_STRATEGY,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        embedding_model=EMBEDDING_MODEL,
        embedding_use_flag=True,
        embedding_batch_size=EMBEDDING_BATCH_SIZE,
        qdrant_url=QDRANT_URL,
        qdrant_collection=COLLECTION_NAME,
        qdrant_force_recreate=recreate,
    )


def run_ingest(local: bool = False, recreate: bool = False):
    """扫描 → 解析 → 分块 → 向量化 → (可选) Qdrant 入库"""
    if not os.path.isdir(SOURCE_DIR):
        print(f"[错误] 源目录不存在: {SOURCE_DIR}")
        print(f"        请修改脚本顶部的 SOURCE_DIR 后重试。")
        return []

    pipeline = VectorizePipeline(build_config(recreate))
    results = pipeline.run([SOURCE_DIR], store_in_qdrant=not local)
    _print_summary(results, local=local)
    return results


def run_search(query: str, top_k: int = TOP_K):
    """检索已入库数据（不执行入库）"""
    pipeline = VectorizePipeline(build_config())
    results = pipeline.search(query, top_k=top_k)

    if not results:
        print("\n  (无结果)")
        return results

    print(f"\n查询: {query}   (top_k={top_k})")
    for i, r in enumerate(results):
        print(f"\n  [{i + 1}] score={r['score']:.4f}")
        print(f"      file: {_safe(os.path.basename(r.get('source_file', '')))}")
        print(f"      text: {_safe(r.get('text', '')[:150])}")
    return results


def _print_summary(results, local: bool = False):
    print("\n" + "=" * 60)
    print(f"{'Ingest Summary':^60}")
    print("=" * 60)
    if not results:
        print("  没有处理任何文件。")
        print(f"  源目录: {SOURCE_DIR}")
        return

    total_chunks = 0
    for r in results:
        chunks = r["chunk_count"]
        total_chunks += chunks
        print(f"  {_safe(os.path.basename(r['file']))}")
        print(f"    type={r['file_type']}, chunks={chunks}")
    print(f"\n  Total: {len(results)} files, {total_chunks} chunks")
    print(f"  Source: {SOURCE_DIR}")
    print(f"  Qdrant: {'(跳过 — local 模式)' if local else f'{QDRANT_URL} / {COLLECTION_NAME}'}")


# ============================================================

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="文档批量向量化入库 / 检索",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--local", action="store_true",
                        help="仅本地处理，跳过 Qdrant 写入")
    parser.add_argument("--recreate", action="store_true",
                        help="先删除 collection 再全量重建（默认是增量 upsert）")
    parser.add_argument("--search", metavar="QUERY",
                        help="检索已入库数据，不执行入库")
    parser.add_argument("--top-k", type=int, default=TOP_K,
                        help=f"检索返回条数（默认 {TOP_K}）")
    args = parser.parse_args(argv)

    if args.search:
        results = run_search(args.search, top_k=args.top_k)
        return 0 if results else 1

    results = run_ingest(local=args.local, recreate=args.recreate)
    # 一个文件都没处理（源目录不存在 / 无可识别文件）→ 非零退出，
    # 否则定时任务会把"什么都没做"误报为成功
    return 0 if results else 1


if __name__ == "__main__":
    sys.exit(main())
