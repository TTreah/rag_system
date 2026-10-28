"""
kb_update.py — 修改已入库文档（删除该文档全部旧块 → 重新入库）

为什么必须先删后入：点 ID 是 hash(文件路径|块内序号)，文件改短后旧的
尾部块没有新点覆盖，不删会留下检索得到、内容已不存在的"幽灵块"。

定位方式（三选一，可混合；按文件名/哈希时直接查主库 payload，无独立清单）:
    按路径    python qdrant_line/kb_update.py /path/文件.pdf
    按文件名  python qdrant_line/kb_update.py --name 用户画像管理规范
    按哈希    python qdrant_line/kb_update.py --hash <sha256前缀>

注意：按文件名/哈希定位时，磁盘上必须还能找到原文件（要重新解析入库）。
"""

import os
import sys
import argparse
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import COLLECTION as _CFG_COLLECTION
from qdrant_line.vectorize_pipeline import (
    VectorizePipeline, PipelineConfig, canonical_path,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    stream=sys.stdout)
logger = logging.getLogger(__name__)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="修改已入库文档（删旧块→重新入库）")
    ap.add_argument("paths", nargs="*", help="文件路径")
    ap.add_argument("--name", action="append", help="按文件名（basename 包含匹配）")
    ap.add_argument("--hash", dest="hashes", action="append",
                    help="按入库时记录的文件 sha256（前缀即可）")
    ap.add_argument("--collection", default=_CFG_COLLECTION)
    ap.add_argument("--qa-augment", action="store_true")
    args = ap.parse_args(argv)

    config = PipelineConfig(qdrant_collection=args.collection,
                            qa_augment=args.qa_augment)
    pipeline = VectorizePipeline(config)

    # 三种方式统一解析成磁盘路径；名称/哈希定位直接查主库 payload
    paths = [canonical_path(p) for p in args.paths]
    if args.name or args.hashes:
        indexed = pipeline.list_indexed_files()
        for name in (args.name or []):
            hit = [p for p in indexed if name in os.path.basename(p)]
            if not hit:
                print(f"[错误] 库里找不到文件名含 '{name}' 的记录")
                return 1
            paths.extend(hit)
        for h in (args.hashes or []):
            hit = [p for p, r in indexed.items()
                   if r.get("file_hash", "").startswith(h)]
            if not hit:
                print(f"[错误] 库里找不到哈希为 {h[:16]}... 的记录")
                return 1
            paths.extend(hit)
    paths = sorted(set(paths))

    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        for p in missing:
            print(f"[错误] 磁盘上找不到: {p}（若文件已删除请用 kb_delete.py）")
        return 1

    # 先删旧块（整份文档的所有 chunk，按 source_file 精确匹配）
    pipeline.delete_by_source_files(paths)
    logger.info("旧块已删除: %d 个文件", len(paths))

    results = pipeline.run(paths, store_in_qdrant=True)
    if not results:
        print("[错误] 重新入库失败。")
        return 1

    total = sum(r["chunk_count"] for r in results)
    print(f"\n更新完成: {len(results)} 个文件, {total} chunks → {args.collection}")
    for r in results:
        print(f"  {os.path.basename(r['file'])}: {r['chunk_count']} chunks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
