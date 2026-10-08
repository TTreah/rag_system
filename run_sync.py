"""
run_sync.py — 每周增量同步（共享盘 → Qdrant）

对比共享盘现状与 Qdrant 里的文件清单，只处理发生变化的文件：
    新增  → 向量化入库
    修改  → 先删该文件旧点，再重新入库
    删除  → 删掉该文件所有点
    改名  → 只改 payload 指向（可选，--detect-renames）
    没变  → 完全不碰

每次运行都会扫一遍源目录下的全部文件（这是检测的基础，不受任何参数影响）；
--batch-size / --max-files 只决定"变化的那部分怎么入库"。

用法:
    python run_sync.py                        # 正式同步（每周一次即可）
    python run_sync.py --dry-run              # 只报告差异，不改任何数据（先跑这个）
    python run_sync.py --recreate             # 全量重建（首次迁移 / 换配置后用）
    python run_sync.py --rebuild-manifest     # 清单丢了，从主库反推重建
    python run_sync.py --batch-size 500       # 每 500 个文件提交一次清单
    python run_sync.py --max-files 5000       # 首次导入量太大时：本轮只做 5000 个，
                                              # 剩下的再跑一次接着做（不必等下周）

退出码:
    0  成功（含"没有任何变化"）
    1  出错
    2  触发安全熔断（共享盘不可达 / 删除比例过高），需要人工确认
"""

import os
import sys
import logging
import argparse

# Windows 控制台统一 UTF-8，避免中文/意大利语路径乱码
if sys.stdout and getattr(sys.stdout, "encoding", "").lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ============================================================
#  可调参数（按实际环境修改）
# ============================================================
SOURCE_DIR = r"D:\pythonfile\Qdrant\PDF"     # 共享盘/本地要同步的目录
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "news_collection"          # 主 collection
MANIFEST_COLLECTION = "file_manifest"        # 清单 collection

# 分块参数 —— 改动这些值会改变 pipeline_fingerprint，
# 下次同步会把所有文件判定为"需重建"（因为旧向量已按旧参数生成）
CHUNK_STRATEGY = "recursive"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64

# 嵌入参数
EMBEDDING_MODEL = "BAAI/bge-m3"              # BGE-M3: 1024维，多语言
EMBEDDING_BATCH_SIZE = 16

# 安全阈值：待删文件超过清单总量的 30% 就中止，需 --force
MAX_DELETE_RATIO = 0.3

# ============================================================

from vectorize_pipeline import VectorizePipeline, PipelineConfig
from manifest_store import FileManifest
from sync_index import IndexSynchronizer, SafetyAbort, DEFAULT_BATCH_SIZE


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    # 第三方库的 INFO 太吵，压到 WARNING
    for noisy in ("httpx", "urllib3", "qdrant_client", "FlagEmbedding",
                  "sentence_transformers", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="每周增量同步：共享盘 → Qdrant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("用法:")[-1],
    )
    parser.add_argument("--root", action="append", metavar="DIR",
                        help=f"源目录，可重复指定（默认 {SOURCE_DIR}）")
    parser.add_argument("--dry-run", action="store_true",
                        help="只报告将要发生的变更，不写任何数据")
    parser.add_argument("--force", action="store_true",
                        help="跳过删除比例熔断（确认大批删除无误时用）")
    parser.add_argument("--recreate", action="store_true",
                        help="清空主 collection 和清单后全量重建")
    parser.add_argument("--rebuild-manifest", action="store_true",
                        help="从主 collection 反推重建清单（清单丢失时兜底）")
    parser.add_argument("--detect-renames", action="store_true",
                        help="用哈希配对识别改名/移动，复用已有点、不重新嵌入")
    # 注意：argparse 会对 help 文本再做一次 % 格式化，字面百分号必须写成 %%
    parser.add_argument("--max-delete-ratio", type=float, default=MAX_DELETE_RATIO,
                        help=f"删除比例熔断阈值，取 0~1 的小数（默认 {MAX_DELETE_RATIO * 100:.0f}%%）")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"入库分批大小，每批完成后提交一次清单"
                             f"（默认 {DEFAULT_BATCH_SIZE}；0 = 不分批一次做完）")
    parser.add_argument("--max-files", type=int, default=None,
                        help="本轮最多入库多少个文件，默认不限。"
                             "仅用于首次导入量太大时限制单次运行时长，"
                             "未处理的部分再跑一次即继续")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="输出 DEBUG 日志")
    args = parser.parse_args(argv)

    setup_logging(args.verbose)

    roots = args.root or [SOURCE_DIR]

    pipeline = VectorizePipeline(build_config(recreate=args.recreate))
    manifest = FileManifest(qdrant_url=QDRANT_URL,
                            collection=MANIFEST_COLLECTION)
    sync = IndexSynchronizer(pipeline, manifest)

    # ── 清单重建分支 ──
    if args.rebuild_manifest:
        try:
            count = sync.rebuild_manifest()
        except Exception as e:
            print(f"\n[错误] 重建清单失败: {e}")
            return 1
        print(f"\n清单已重建：{count} 个文件。")
        print("建议接着跑一次 --dry-run 核对差异。")
        return 0

    # ── --recreate：清空清单，让所有文件都被视为新增 ──
    if args.recreate:
        print("--recreate：将清空主 collection 与清单，全量重建。")
        manifest.drop_collection()

    print(f"源目录  : {', '.join(roots)}")
    print(f"主库    : {QDRANT_URL} / {COLLECTION_NAME}")
    print(f"清单    : {QDRANT_URL} / {MANIFEST_COLLECTION}")
    print(f"指纹    : {sync.fingerprint}")
    print(f"模式    : {'DRY-RUN（不写数据）' if args.dry_run else '正式同步'}")
    print()

    try:
        report = sync.run(
            roots,
            dry_run=args.dry_run,
            force=args.force,
            detect_renames=args.detect_renames,
            max_delete_ratio=args.max_delete_ratio,
            batch_size=args.batch_size,
            max_files=args.max_files,
        )
    except SafetyAbort as e:
        print("\n" + "!" * 66)
        print("安全熔断，已中止，未改动任何数据：")
        print(f"  {e}")
        print("!" * 66)
        return 2
    except Exception as e:
        logging.exception("同步失败")
        print(f"\n[错误] {e}")
        return 1

    print("\n" + report.describe())
    return 0


if __name__ == "__main__":
    sys.exit(main())
