"""
kb_add.py — 新增文档入库（单文件粒度）

对指定文件做完整入库：解析 → 归一化 → 分块 → QA增强 → 嵌入 → Qdrant。
已在库中的文件会跳过（除非 --force）。

"哪些文件要新增"由外部扫描+数据库脚本决定，本脚本只管执行。

用法:
    python qdrant_line/kb_add.py /path/to/新文件.pdf [更多文件...]
    python qdrant_line/kb_add.py /path/dir/ --collection CEO-files
"""

import os
import sys
import argparse
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qdrant_line.vectorize_pipeline import (
    VectorizePipeline, PipelineConfig, canonical_path,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    stream=sys.stdout)
logger = logging.getLogger(__name__)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="新增文档入库")
    ap.add_argument("paths", nargs="+", help="文件或目录")
    ap.add_argument("--collection", default="news_collection")
    ap.add_argument("--force", action="store_true",
                    help="已在库中也强制重入（改内容请用 kb_update.py）")
    ap.add_argument("--qa-augment", action="store_true",
                    help="开启 QA 假设问题增强（每 chunk 一次 LLM 调用）")
    args = ap.parse_args(argv)

    config = PipelineConfig(qdrant_collection=args.collection,
                            qa_augment=args.qa_augment)
    pipeline = VectorizePipeline(config)

    # 展开目录 + 规范化
    targets = []
    for p in args.paths:
        cp = canonical_path(p)
        if os.path.isdir(cp):
            targets.extend(pipeline._collect_files([cp]))
        else:
            targets.append(cp)

    # 已入库判断：查主库 payload（没有独立清单）
    existing = pipeline.list_indexed_files()
    todo = [f for f in targets if args.force or f not in existing]
    skipped = len(targets) - len(todo)
    if skipped:
        logger.info("跳过已入库: %d 个（--force 可强制）", skipped)
    if not todo:
        print("没有需要新增的文件。")
        return 0

    results = pipeline.run(todo, store_in_qdrant=True)
    if not results:
        print("[错误] 没有文件入库成功。")
        return 1

    total = sum(r["chunk_count"] for r in results)
    print(f"\n新增完成: {len(results)} 个文件, {total} chunks → {args.collection}")
    for r in results:
        print(f"  {os.path.basename(r['file'])}: {r['chunk_count']} chunks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
