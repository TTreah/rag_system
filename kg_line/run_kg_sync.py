"""
run_kg_sync.py — 定时增量同步：共享盘 → LightRAG 知识图谱

与 run_sync.py（Qdrant 向量线）平行的知识图谱线入口，两条线互不影响：
    run_sync.py     → Qdrant news_collection + file_manifest（向量检索）
    run_kg_sync.py  → LightRAG 知识图谱 + kg_manifest.json（实体关系查询）

对比共享盘现状与本地清单，只处理发生变化的文件：
    新增  → 上传 LightRAG，等 LLM 抽取完成后写清单
    修改  → 先删服务端旧文档，再重新上传
    删除  → 删除服务端文档
    改名  → 只更新清单（内容没变，不重复抽取；--detect-renames）
    没变  → 完全不碰

用法:
    python run_kg_sync.py                        # 正式同步（建议先 --dry-run）
    python run_kg_sync.py --dry-run              # 只报告差异，不改任何数据
    python run_kg_sync.py --recreate             # 清空图谱与清单后全量重建
    python run_kg_sync.py --rebuild-manifest     # 清单丢了，从服务端配对重建
    python run_kg_sync.py --max-files 20         # 本轮最多入库 20 个，剩下的下次继续
    python run_kg_sync.py --batch-size 5         # 每 5 个文件提交一次清单

退出码:
    0  成功（含"没有任何变化"）
    1  出错
    2  触发安全熔断（共享盘不可达 / 删除比例过高 / 服务不可用）

定时调度（macOS launchd / cron / Windows 任务计划均可），注意：
图谱入库走 LLM 抽取，比向量入库慢几个数量级——首轮全量建议用
--max-files 分批跑几个晚上，之后的周增量都很小。
"""

import os, sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
del _sys


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
from config import SOURCE_DIR, LIGHTRAG_URL  # 共享盘目录与图谱服务地址来自 .env
MANIFEST_PATH = None                          # None = 脚本旁的 kg_manifest.json

# 安全阈值：待删文档超过清单总量的 30% 就中止，需 --force
MAX_DELETE_RATIO = 0.3

# ============================================================

from kg_line.kg_docs_client import LightRAGDocsClient, KGApiError
from kg_line.kg_manifest import KgManifest, DEFAULT_MANIFEST_PATH
from kg_line.kg_sync import (
    KGSynchronizer, SafetyAbort, make_fingerprint, DEFAULT_KG_BATCH_SIZE,
)


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    for noisy in ("httpx", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


class _RunLock:
    """本地锁文件：防止定时任务与手动执行并发跑。

    图谱入库是异步的，上一轮可能还在抽取；两个同步进程同时改清单
    会互相覆盖。简单起见用 O_EXCL 创建锁文件，进程退出即释放。
    """

    def __init__(self, path: str):
        self.path = path
        self.fd = None

    def __enter__(self):
        try:
            self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(self.fd, str(os.getpid()).encode())
        except FileExistsError:
            raise SafetyAbort(
                f"已有同步进程在运行（锁文件 {self.path}）。\n"
                f"确认没有进程在跑后，删掉锁文件再试。"
            )
        return self

    def __exit__(self, *exc):
        if self.fd is not None:
            os.close(self.fd)
        try:
            os.unlink(self.path)
        except OSError:
            pass
        return False


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="定时增量同步：共享盘 → LightRAG 知识图谱",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("用法:")[-1],
    )
    parser.add_argument("--root", action="append", metavar="DIR",
                        help=f"源目录，可重复指定（默认 {SOURCE_DIR}）")
    parser.add_argument("--file", action="append", metavar="FILE",
                        help="只同步指定文件，可重复指定（部分同步："
                             "此模式下删除/改名检测关闭，只增不改")
    parser.add_argument("--dry-run", action="store_true",
                        help="只报告将要发生的变更，不写任何数据")
    parser.add_argument("--force", action="store_true",
                        help="跳过删除比例熔断（确认大批删除无误时用）")
    parser.add_argument("--recreate", action="store_true",
                        help="清空 LightRAG 全部文档和本地清单后全量重建")
    parser.add_argument("--rebuild-manifest", action="store_true",
                        help="清单丢失时，从服务端文档列表按文件名配对重建")
    parser.add_argument("--detect-renames", action="store_true",
                        help="用哈希配对识别改名/移动，只更新清单不重复抽取")
    parser.add_argument("--max-delete-ratio", type=float, default=MAX_DELETE_RATIO,
                        help=f"删除比例熔断阈值，取 0~1 的小数（默认 {MAX_DELETE_RATIO * 100:.0f}%%）")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_KG_BATCH_SIZE,
                        help=f"入库分批大小，每批完成后提交一次清单"
                             f"（默认 {DEFAULT_KG_BATCH_SIZE}；0 = 不分批一次做完）")
    parser.add_argument("--max-files", type=int, default=None,
                        help="本轮最多入库多少个文件，默认不限。"
                             "首轮全量太大时限制单次运行时长，"
                             "未处理的部分再跑一次即继续")
    parser.add_argument("--manifest", metavar="PATH", default=MANIFEST_PATH,
                        help="清单文件路径（默认脚本旁的 kg_manifest.json）")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="输出 DEBUG 日志")
    args = parser.parse_args(argv)

    setup_logging(args.verbose)

    roots = args.root or [SOURCE_DIR]
    manifest_path = args.manifest or DEFAULT_MANIFEST_PATH

    client = LightRAGDocsClient(base_url=LIGHTRAG_URL)
    manifest = KgManifest(manifest_path)

    # 服务可用性前置检查 + 取指纹（服务端换了 LLM/embedding 模型 → 全量重建）
    try:
        health = client.health()
        fingerprint = make_fingerprint(health)
    except KGApiError as e:
        print(f"\n[错误] 无法连接 LightRAG 服务: {e}")
        return 2

    sync = KGSynchronizer(client, manifest, fingerprint=fingerprint)

    print(f"源目录  : {', '.join(roots)}")
    print(f"图谱服务: {LIGHTRAG_URL} ({health.get('webui_title', '')}, "
          f"v{health.get('core_version', '?')})")
    print(f"清单    : {manifest_path}")
    print(f"指纹    : {fingerprint}")
    print()

    # ── 清单重建分支 ──
    if args.rebuild_manifest:
        try:
            count = sync.rebuild_manifest(roots)
        except (SafetyAbort, KGApiError) as e:
            print(f"\n[错误] 重建清单失败: {e}")
            return 1
        print(f"清单已重建：{count} 个文件。")
        print("建议接着跑一次 --dry-run 核对差异。")
        return 0

    # ── --recreate：清空服务端文档与清单，所有文件视为新增 ──
    if args.recreate:
        print("--recreate：将清空 LightRAG 全部文档与本地清单，全量重建。")
        try:
            client.clear_all()
        except KGApiError as e:
            print(f"\n[错误] 清空图谱失败: {e}")
            return 1
        manifest.drop()

    print(f"模式    : {'DRY-RUN（不写数据）' if args.dry_run else '正式同步'}")
    print()

    try:
        with _RunLock(manifest_path + ".lock"):
            report = sync.run(
                roots,
                dry_run=args.dry_run,
                force=args.force,
                detect_renames=args.detect_renames,
                max_delete_ratio=args.max_delete_ratio,
                batch_size=args.batch_size,
                max_files=args.max_files,
                only_files=args.file,
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
