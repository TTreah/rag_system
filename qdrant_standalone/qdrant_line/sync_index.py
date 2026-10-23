"""
sync_index.py — 扫盘与差异比对的纯函数工具库

注意：同步编排（IndexSynchronizer）与文件清单（file_manifest）已下线——
"哪些文件要新增/修改/删除"由外部扫描+数据库脚本决定，
执行走 kb_add.py / kb_update.py / kb_delete.py。
本文件只保留被复用的纯工具：scan_disk / plan_sync / SafetyAbort / _file_hash 等。

每周跑一次，把共享盘的现状同步进 Qdrant：
    新增文件      → 向量化入库
    内容变化的    → 先删掉该文件所有旧点，再重新入库
    被删除的      → 删掉该文件所有点
    未变化的      → 完全不碰（不读内容、不写库）

变更检测分两层，核心目的是别为了「看一眼有没有变」就去读 10 万个文件：
    第一层  路径 + size + mtime 比对。只读目录元数据，os.scandir 一次目录列举
            就把它们带回来了，一个字节的文件内容都不用读。
    第二层  仅对可疑文件算 sha256 确认。size 变了必然变了，无需确认；只有
            mtime 变了而 size 没变才需要哈希，用来区分「被 touch 了」和「真的改了」。

为什么 mtime 不能当权威依据：SMB 上 mtime 会因共享主机与执行机的时钟偏移、
时区差异、拷贝行为而抖动。但关键在两类信号错的方向不同——mtime 不准只造成
假阳性（多算一次哈希，慢一点），方向是安全的；真正危险的是漏掉变更，而
「内容变了但 size 和 mtime 都没变」几乎不可能发生。所以分层既省钱又安全。

状态放在 Qdrant 的 file_manifest collection 里。写清单的时机在入库成功之后，
因此任何一步崩溃，下一周都会重新识别并收敛，整个流程是幂等的。

关于"分批"：扫盘永远覆盖 roots 下的全部文件，检测的完整性不受任何参数影响；
分批只作用于"变化的那部分怎么入库"。入库按 batch_size 切批，每批
「删旧点 → 入库 → 写清单」一次做完再进下一批，所以进度是持久的：中断后
直接重跑即可，已提交的批次不会被重复嵌入。首次导入十几万文件跑不完时，
用 max_files 限定本轮处理量，剩下的下次运行继续 —— 不需要等到下周，
下周的增量照样按自己的节奏跑。
"""

import os
import hashlib
import logging
import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from qdrant_line.vectorize_pipeline import canonical_path

logger = logging.getLogger(__name__)

# 待删文件数超过清单总量的这个比例就中止，需要 --force 才继续
MAX_DELETE_RATIO = 0.3

# 入库分批大小。每批提交一次清单，所以这个值决定了"中断时最多损失多少进度"。
# 200 个文件 ≈ 几分钟的解析+嵌入，中断代价可接受；调大则提交更少、峰值内存更高。
DEFAULT_BATCH_SIZE = 200

# 目录扫描时跳过的垃圾文件
SKIP_NAME_PREFIXES = ("~$",)                      # Office 锁文件
SKIP_SUFFIXES = (".tmp", ".crdownload", ".part")  # 半截下载/临时文件
# 跳过的目录名：kb-assets 是入库时自动生成的资产目录（里面的 .md 是表格资产，
# 不是待入库文档）；#recycle 是 NAS 回收站
SKIP_DIR_NAMES = {"kb-assets", "#recycle", "@eaDir"}

_HASH_CHUNK = 1024 * 1024


def _file_hash(path: str) -> str:
    """文件内容 sha256（分块读，避免把大文件整个读进内存）。"""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(_HASH_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


# ═══════════════════════════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════════════════════════

@dataclass
class DiskEntry:
    """磁盘上一个文件的元数据（不含内容）。"""
    path: str
    size: int
    mtime: int


@dataclass
class SyncPlan:
    """一次同步的差异清单。"""
    added: List[str] = field(default_factory=list)
    modified: List[str] = field(default_factory=list)
    deleted: List[str] = field(default_factory=list)
    renamed: List[Tuple[str, str]] = field(default_factory=list)
    unreadable: List[str] = field(default_factory=list)
    unchanged: int = 0
    hashes: Dict[str, str] = field(default_factory=dict)

    @property
    def to_ingest(self) -> List[str]:
        """需要重新向量化的文件（新增 + 修改）。"""
        return sorted(set(self.added) | set(self.modified))

    @property
    def paths_to_delete(self) -> List[str]:
        """入库前必须先删点的文件（修改 + 删除）。

        改文件必须走这条路：文件变短后旧的尾部 chunk 没有新点覆盖，
        不删就会留下检索得到、内容却不存在的「幽灵文本块」。
        """
        return sorted(set(self.modified) | set(self.deleted))

    @property
    def has_changes(self) -> bool:
        return bool(self.added or self.modified or self.deleted or self.renamed)


@dataclass
class SyncReport:
    """一次同步的执行结果。"""
    plan: SyncPlan
    chunks: int = 0
    files_ingested: int = 0
    files_removed: int = 0
    batches: int = 0
    dry_run: bool = False
    skipped_by_limit: int = 0
    aborted: bool = False
    abort_reason: str = ""

    def describe(self) -> str:
        p = self.plan
        lines = [
            "=" * 66,
            f"{'增量同步结果' + ('  [DRY-RUN 未改动任何数据]' if self.dry_run else ''):^66}",
            "=" * 66,
            f"  新增      : {len(p.added):>6}",
            f"  修改      : {len(p.modified):>6}",
            f"  删除      : {len(p.deleted):>6}",
            f"  改名      : {len(p.renamed):>6}",
            f"  未变化    : {p.unchanged:>6}",
            f"  读取失败  : {len(p.unreadable):>6}",
        ]
        if self.skipped_by_limit:
            lines.append(
                f"  受 --max-files 截断: {self.skipped_by_limit:>6}"
                f"（再跑一次即继续）"
            )
        if not self.dry_run:
            lines.append("-" * 66)
            lines.append(f"  入库文件  : {self.files_ingested:>6}"
                         + (f"   （分 {self.batches} 批）" if self.batches else ""))
            lines.append(f"  入库 chunk: {self.chunks:>6}")
            if self.files_removed:
                lines.append(f"  清理删除  : {self.files_removed:>6}")
        if not p.has_changes:
            lines.append("\n  没有任何变化。")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# 目录扫描（只读元数据）
# ═══════════════════════════════════════════════════════════════

def scan_disk(
    root: str,
    allowed_exts: Optional[List[str]] = None,
    recursive: bool = True,
    follow_symlinks: bool = False,
) -> Dict[str, DiskEntry]:
    """遍历目录，收集 {规范路径: (size, mtime)}。

    用 os.scandir 而不是 Path.rglob：在 Windows/SMB 上目录列举本身就带回了
    size/mtime，DirEntry.stat() 不再产生额外网络往返。10 万文件走共享盘时，
    这个差别就是几万次 round-trip。

    follow_symlinks 默认关闭，兼作防符号链接成环。
    """
    from core.document_parser import DocumentParser

    allowed = {
        e.lower() for e in (allowed_exts or DocumentParser.supported_extensions())
    }

    found: Dict[str, DiskEntry] = {}
    stack = [str(root)]
    dir_count = 0

    while stack:
        current = stack.pop()
        dir_count += 1
        try:
            with os.scandir(current) as it:
                for entry in it:
                    try:
                        if entry.is_dir(follow_symlinks=follow_symlinks):
                            if recursive and entry.name not in SKIP_DIR_NAMES:
                                stack.append(entry.path)
                            continue
                        if not entry.is_file(follow_symlinks=follow_symlinks):
                            continue

                        name = entry.name
                        if name.startswith(SKIP_NAME_PREFIXES):
                            continue
                        if name.lower().endswith(SKIP_SUFFIXES):
                            continue
                        if os.path.splitext(name)[1].lower() not in allowed:
                            continue

                        st = entry.stat()
                        path = canonical_path(entry.path)
                        found[path] = DiskEntry(
                            path=path,
                            size=int(st.st_size),
                            mtime=int(st.st_mtime),
                        )
                    except OSError as e:
                        # 单个条目读不到（权限、被占用、断链）不影响整体
                        logger.debug("跳过 %s: %s", entry.path, e)
                        continue
        except OSError as e:
            logger.warning("无法读取目录 %s: %s", current, e)

    logger.info("Scanned %d dir(s), found %d supported file(s)",
                dir_count, len(found))
    return found


# ═══════════════════════════════════════════════════════════════
# 差异规划（纯逻辑，不碰 Qdrant，可直接单测）
# ═══════════════════════════════════════════════════════════════

def plan_sync(
    disk: Dict[str, DiskEntry],
    manifest: Dict[str, Dict[str, Any]],
    fingerprint: str,
    file_hash_fn: Callable[[str], str],
    detect_renames: bool = False,
) -> SyncPlan:
    """比对磁盘现状与清单，产出同步计划。

    Args:
        disk: scan_disk() 的结果
        manifest: 外部比对基准 {路径: payload}（由调用方的扫描+数据库脚本提供）
        fingerprint: 当前 pipeline_fingerprint()
        file_hash_fn: 计算文件内容哈希的函数（注入以便测试）
        detect_renames: 是否用哈希配对识别「改名 / 移动」
    """
    cache: Dict[str, Optional[str]] = {}

    def file_hash(path: str) -> Optional[str]:
        if path not in cache:
            try:
                cache[path] = file_hash_fn(path)
            except OSError as e:
                # 读不到（被占用 / 权限 / 半截拷贝）→ 本轮跳过该文件，
                # 既不删它的旧点也不入库。这是安全方向：宁可不动，不要动错。
                logger.warning("读取失败，本轮跳过: %s (%s)", path, e)
                cache[path] = None
        return cache[path]

    added: List[str] = []
    modified: List[str] = []
    suspect: List[str] = []

    for path, entry in disk.items():
        record = manifest.get(path)

        if record is None:
            added.append(path)
            continue

        # 配置变了 → 已入库的向量已过期，即使文件没变也必须重建
        if record.get("pipeline_fingerprint") != fingerprint:
            modified.append(path)
        elif int(record.get("size", -1)) != entry.size:
            modified.append(path)      # size 变了必然变了，不必再算哈希
        elif int(record.get("mtime", -1)) != entry.mtime:
            suspect.append(path)       # 只有 mtime 变了 → 疑似 touch，需哈希确认

    # 第二层：哈希确认（只为疑似文件读内容）
    unreadable: List[str] = []
    for path in suspect:
        actual = file_hash(path)
        if actual is None:
            # 核对不了就维持原状（不重建），但要记下来，否则这个文件会被
            # 静默跳过、谁也不知道它这周没被检查过
            unreadable.append(path)
        elif actual != manifest[path].get("file_hash"):
            modified.append(path)

    deleted = [p for p in manifest if p not in disk]

    # 要入库的文件必须拿到哈希（写进 chunk payload 和清单），读不到的剔除
    hashes: Dict[str, str] = {}
    for path in added + modified:
        h = file_hash(path)
        if h:
            hashes[path] = h

    unreadable += [p for p in added + modified if p not in hashes]
    added = [p for p in added if p in hashes]
    modified = [p for p in modified if p in hashes]

    # 可选：用哈希把「删掉的旧文件」和「新增的文件」配对成改名，省掉重新嵌入
    renamed: List[Tuple[str, str]] = []
    if detect_renames and deleted and added:
        by_hash: Dict[str, List[str]] = {}
        for path in deleted:
            h = manifest[path].get("file_hash")
            if h:
                by_hash.setdefault(h, []).append(path)

        kept_added = []
        for path in added:
            bucket = by_hash.get(hashes.get(path, ""))
            if bucket:
                renamed.append((bucket.pop(0), path))
            else:
                kept_added.append(path)

        renamed_old = {old for old, _ in renamed}
        added = kept_added
        deleted = [p for p in deleted if p not in renamed_old]

    return SyncPlan(
        added=added,
        modified=modified,
        deleted=deleted,
        renamed=renamed,
        unreadable=unreadable,
        unchanged=len(disk) - len(added) - len(modified) - len(renamed),
        hashes=hashes,
    )


# ═══════════════════════════════════════════════════════════════
# 同步执行
# ═══════════════════════════════════════════════════════════════

class SafetyAbort(RuntimeError):
    """触发了保护性熔断（需要人工确认，不是程序错误）。"""
