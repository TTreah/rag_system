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

import os, sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
del _sys


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
from config import SOURCE_DIR, QDRANT_URL, COLLECTION as _CFG_COLLECTION, EMBEDDING_MODEL as _CFG_EMBED  # 要扫描的目录（支持混合文件类型）
COLLECTION_NAME = _CFG_COLLECTION          # 目标 Qdrant collection

# 分块参数
CHUNK_STRATEGY = "recursive"                 # fixed / recursive / sentence / markdown
CHUNK_SIZE = 768
CHUNK_OVERLAP = 100

# 嵌入参数
EMBEDDING_MODEL = _CFG_EMBED                  # Qwen3-VL-Embedding-2B: 2048维，多模态
EMBEDDING_BATCH_SIZE = 16

# 检索参数
TOP_K = 3

# ============================================================

from qdrant_line.vectorize_pipeline import VectorizePipeline, PipelineConfig


def _safe(s) -> str:
    """输出兜底（UTF-8 下无需转码，仅防御 None/非字符串）。"""
    if s is None:
        return ""
    return str(s)


def build_config(recreate: bool = False, collection: str = None,
                 qa_augment: bool = False) -> PipelineConfig:
    return PipelineConfig(
        chunk_strategy=CHUNK_STRATEGY,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        qa_augment=qa_augment,
        embedding_model=EMBEDDING_MODEL,
        embedding_use_qwen3_vl=True,
        embedding_batch_size=EMBEDDING_BATCH_SIZE,
        qdrant_url=QDRANT_URL,
        qdrant_collection=collection or COLLECTION_NAME,
        qdrant_force_recreate=recreate,
    )


def run_ingest(local: bool = False, recreate: bool = False,
               roots: list = None, collection: str = None,
               qa_augment: bool = False):
    """扫描 → 解析 → 分块 → 向量化 → (可选) Qdrant 入库"""
    roots = roots or [SOURCE_DIR]
    missing = [r for r in roots if not os.path.isdir(r)]
    if missing:
        print(f"[错误] 源目录不存在: {missing}")
        return []

    pipeline = VectorizePipeline(build_config(recreate, collection, qa_augment))
    results = pipeline.run(roots, store_in_qdrant=not local)
    _print_summary(results, local=local, roots=roots,
                   collection=collection or COLLECTION_NAME)
    return results


def run_search(query: str, top_k: int = TOP_K, collection: str = None):
    """检索已入库数据（不执行入库）。

    统一走 RAGSearcher：查询改写 + 混合召回 + reranker 重排 + 相关性闸门，
    与 rag_search.py 行为完全一致。
    """
    from qdrant_line.rag_search import RAGSearcher
    searcher = RAGSearcher(qdrant_url=QDRANT_URL,
                           collection=collection or COLLECTION_NAME)
    results = searcher.search(query, top_k=top_k)

    if not results:
        print("\n  (无结果 —— 相关性闸门判定该查询与库内文档无关，已拒答)")
        return results

    print(f"\n查询: {query}   (top_k={top_k})")
    for i, r in enumerate(results):
        score = r.get("rerank_score", r["score"])
        print(f"\n  [{i + 1}] score={score:.4f}")
        print(f"      file: {_safe(os.path.basename(r.get('source_file', '')))}")
        print(f"      text: {_safe(r.get('text', '')[:150])}")
    return results


def _print_summary(results, local: bool = False, roots=None, collection=None):
    print("\n" + "=" * 60)
    print(f"{'Ingest Summary':^60}")
    print("=" * 60)
    if not results:
        print("  没有处理任何文件。")
        print(f"  源目录: {roots or [SOURCE_DIR]}")
        return

    total_chunks = 0
    for r in results:
        chunks = r["chunk_count"]
        total_chunks += chunks
        print(f"  {_safe(os.path.basename(r['file']))}")
        print(f"    type={r['file_type']}, chunks={chunks}")
    print(f"\n  Total: {len(results)} files, {total_chunks} chunks")
    print(f"  Source: {roots or [SOURCE_DIR]}")
    print(f"  Qdrant: {'(跳过 — local 模式)' if local else f'{QDRANT_URL} / {collection or COLLECTION_NAME}'}")


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
    parser.add_argument("--root", action="append", metavar="DIR",
                        help="源目录，可重复指定（默认脚本顶部 SOURCE_DIR）")
    parser.add_argument("--collection", default=None,
                        help=f"目标 Qdrant collection（默认 {COLLECTION_NAME}）")
    parser.add_argument("--qa-augment", action="store_true",
                        help="开启 QA 假设问题增强（与 CEO-files 同质量构建需开启）")
    parser.add_argument("--top-k", type=int, default=TOP_K,
                        help=f"检索返回条数（默认 {TOP_K}）")
    args = parser.parse_args(argv)

    if args.search:
        results = run_search(args.search, top_k=args.top_k,
                             collection=args.collection)
        return 0 if results else 1

    results = run_ingest(local=args.local, recreate=args.recreate,
                         roots=args.root, collection=args.collection,
                         qa_augment=args.qa_augment)
    # 一个文件都没处理（源目录不存在 / 无可识别文件）→ 非零退出，
    # 否则定时任务会把"什么都没做"误报为成功
    return 0 if results else 1


if __name__ == "__main__":
    sys.exit(main())
