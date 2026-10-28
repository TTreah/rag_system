"""
kb_delete.py — 删除已入库文档（整份文档的所有块）

按 source_file 精确匹配删除（canonical_path 规范化后比对，写库和删除
共用同一函数，不会静默失配）。

定位方式（三选一，可混合；按文件名/哈希时直接查主库 payload，无独立清单）:
    按路径    python qdrant_line/kb_delete.py /path/文件.pdf
    按文件名  python qdrant_line/kb_delete.py --name 定价标准
    按哈希    python qdrant_line/kb_delete.py --hash <sha256前缀>

默认先打印将删除的内容并需 --yes 确认（防误删）。
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="删除已入库文档的全部块")
    ap.add_argument("paths", nargs="*", help="文件路径")
    ap.add_argument("--name", action="append", help="按文件名（basename 包含匹配）")
    ap.add_argument("--hash", dest="hashes", action="append",
                    help="按入库时记录的文件 sha256（前缀即可）")
    ap.add_argument("--collection", default=_CFG_COLLECTION)
    ap.add_argument("--yes", action="store_true", help="跳过确认直接删")
    args = ap.parse_args(argv)

    config = PipelineConfig(qdrant_collection=args.collection)
    pipeline = VectorizePipeline(config)

    paths = [canonical_path(p) for p in args.paths]
    indexed = {}
    if args.name or args.hashes or True:  # 打印 chunk 数也需要
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

    if not paths:
        print("没有指定任何文件。")
        return 1

    print(f"将从 {args.collection} 删除以下 {len(paths)} 个文件的所有块：")
    for p in paths:
        rec = indexed.get(p, {})
        print(f"  {os.path.basename(p)}  （库中块数: {rec.get('chunk_count', '?')}）")

    if not args.yes:
        print("\n确认请加 --yes 重跑。未做任何改动。")
        return 0

    pipeline.delete_by_source_files(paths)
    print(f"\n已删除 {len(paths)} 个文件的全部块。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
