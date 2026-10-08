"""
sync_index.py — 增量同步：共享盘目录 ↔ Qdrant 索引

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

from vectorize_pipeline import (
    VectorizePipeline,
    canonical_path,
    pipeline_fingerprint,
)
from manifest_store import FileManifest

logger = logging.getLogger(__name__)

# 待删文件数超过清单总量的这个比例就中止，需要 --force 才继续
MAX_DELETE_RATIO = 0.3

# 入库分批大小。每批提交一次清单，所以这个值决定了"中断时最多损失多少进度"。
# 200 个文件 ≈ 几分钟的解析+嵌入，中断代价可接受；调大则提交更少、峰值内存更高。
DEFAULT_BATCH_SIZE = 200

# 目录扫描时跳过的垃圾文件
SKIP_NAME_PREFIXES = ("~$",)                      # Office 锁文件
SKIP_SUFFIXES = (".tmp", ".crdownload", ".part")  # 半截下载/临时文件

_HASH_CHUNK = 1024 * 1024


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
    from document_parser import DocumentParser

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
                            if recursive:
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
        manifest: FileManifest.load() 的结果 {路径: payload}
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


class IndexSynchronizer:
    """把共享盘目录同步进 Qdrant，增量更新文件清单。"""

    def __init__(
        self,
        pipeline: VectorizePipeline,
        manifest: FileManifest,
        fingerprint: Optional[str] = None,
    ):
        self.pipeline = pipeline
        self.manifest = manifest
        self.fingerprint = fingerprint or pipeline_fingerprint(pipeline.config)

    # ── 入口 ─────────────────────────────────────────────

    def run(
        self,
        roots: List[str],
        dry_run: bool = False,
        force: bool = False,
        detect_renames: bool = False,
        max_delete_ratio: float = MAX_DELETE_RATIO,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_files: Optional[int] = None,
    ) -> SyncReport:
        """执行一次增量同步。

        无论有没有变化，扫盘总是覆盖 roots 下的全部文件；batch_size / max_files
        只影响"变化的那部分怎么入库"，不影响检测的完整性。

        Args:
            batch_size: 入库分批大小。每批入库成功后立刻写清单，因此进度是
                持久化的：中途中断只需重跑，已完成的部分不会被重复嵌入。
                0 或负数表示不分批，一次全部入库。
            max_files: 本轮最多入库多少个文件（默认不限）。这是为"首次导入
                十几万文件、一次跑不完"准备的，剩下的在**下次运行**继续，
                不是等下周——下周的增量照样正常跑。

        Raises:
            SafetyAbort: 触发了共享盘可达性 / 删除比例熔断
        """
        # ── 阶段 0：共享盘可达性 ──
        # 掉线时目录遍历返回空集，若不拦住会被当成「所有文件都被删了」，
        # 从而删光整个索引。这是整套流程里破坏性最大的一种失败模式。
        for root in roots:
            if not os.path.isdir(root):
                raise SafetyAbort(
                    f"源目录不可访问: {root}\n"
                    f"共享盘掉线 / 未挂载 / 凭据过期时继续执行会判定"
                    f"「全部文件已删除」并清空索引，因此中止。"
                )

        # ── 阶段 1：扫盘（只读元数据）──
        cfg = self.pipeline.config
        disk: Dict[str, DiskEntry] = {}
        for root in roots:
            disk.update(scan_disk(
                root,
                allowed_exts=cfg.supported_extensions,
                recursive=cfg.recursive,
            ))
        if not disk:
            raise SafetyAbort(
                f"在 {roots} 下没有扫描到任何受支持的文件，中止以免误判为全删。\n"
                f"如果共享盘确实已清空，请手工重建索引（--recreate）。"
            )

        # ── 阶段 2：读清单 ──
        manifest_records = self.manifest.load()

        # ── 阶段 3：规划差异 ──
        plan = plan_sync(
            disk, manifest_records, self.fingerprint,
            file_hash_fn=self._file_hash,
            detect_renames=detect_renames,
        )

        # ── 阶段 4：删除比例熔断 ──
        if manifest_records and not force:
            budget = max(1, int(len(manifest_records) * max_delete_ratio))
            if len(plan.deleted) > budget:
                raise SafetyAbort(
                    f"待删文件数 {len(plan.deleted)} 超过清单总量 "
                    f"{len(manifest_records)} 的 {max_delete_ratio:.0%}（{budget} 个）。\n"
                    f"常见原因：共享盘内容不完整、路径形式变了（映射盘符 vs UNC）。\n"
                    f"确认这批删除无误后，加 --force 重跑。"
                )

        # ── 阶段 5：--max-files 截断（必须在删除之前决定）──
        # 被截断的文件本轮完全不处理：不动它的旧点，也不入库，
        # 下次运行仍会被判为变化并继续。
        to_ingest = plan.to_ingest
        skipped = 0
        if max_files is not None and max_files < len(to_ingest):
            skipped = len(to_ingest) - max_files
            keep = set(to_ingest[:max_files])
            plan.added = [p for p in plan.added if p in keep]
            plan.modified = [p for p in plan.modified if p in keep]
            plan.hashes = {p: h for p, h in plan.hashes.items() if p in keep}
            to_ingest = plan.to_ingest
            logger.info("--max-files %d：本轮入库 %d 个，其余 %d 个下次运行继续",
                        max_files, len(to_ingest), skipped)

        report = SyncReport(plan=plan, dry_run=dry_run, skipped_by_limit=skipped)

        if dry_run:
            logger.info("DRY-RUN：未改动任何数据")
            return report

        if not plan.has_changes:
            logger.info("没有任何变化，跳过全部写操作")
            return report

        now = datetime.datetime.now().isoformat()

        # ── 阶段 6：磁盘上已消失的文件 ──
        # 与入库无关，一次做完。顺序上要放在批次之前：先让库里不再引用
        # 已经不存在的文件，再去忙新增和修改。
        if plan.deleted:
            logger.info("Step 1/4: 清理 %d 个已删除文件 ...", len(plan.deleted))
            self.pipeline.delete_by_source_files(plan.deleted)
            self.manifest.delete_paths(plan.deleted)
            report.files_removed = len(plan.deleted)

        # ── 阶段 7：改名（只改 payload，不重新嵌入）──
        if plan.renamed:
            logger.info("Step 2/4: 改名 %d 项（复用已有点）...", len(plan.renamed))
            for old, new in plan.renamed:
                self.pipeline.repoint_source_file(old, new)
            # manifest_records 是阶段 2 的快照，改名要继承的 file_hash /
            # chunk_count 从这里取 —— 此刻旧记录可能已被删掉，只能靠快照
            self.manifest.delete_paths([old for old, _ in plan.renamed])
            self.manifest.upsert_many(
                self._renamed_records(plan, disk, manifest_records, now)
            )

        # ── 阶段 8：分批次入库 ──
        # 每批"删旧点 → 入库 → 写清单"一次做完再进下一批。这样进度是持久的：
        # 中断后重跑，已提交的批次不会被重复嵌入。若把清单统一留到最后写，
        # 中途失败就等于白跑，而这正是首次导入十几万文件时最可能发生的事。
        if to_ingest:
            batches = self._split_batches(to_ingest, batch_size)
            report.batches = len(batches)
            logger.info("Step 3/4: 入库 %d 个文件，分 %d 批（每批 %s 个）...",
                        len(to_ingest), len(batches),
                        batch_size if batch_size > 0 else "全部")
            modified_set = set(plan.modified)
            for bi, batch in enumerate(batches, 1):
                logger.info("  批次 %d/%d：%d 个文件", bi, len(batches), len(batch))

                # 只对"修改过"的文件清旧点：新增文件在库里本来就没有点，
                # 首次导入十万文件时白跑这些删除调用是纯浪费。
                stale = [p for p in batch if p in modified_set]
                if stale:
                    self.pipeline.delete_by_source_files(stale)

                file_meta = {p: {"file_hash": plan.hashes[p]} for p in batch}
                results = self.pipeline.run(
                    batch, store_in_qdrant=True, file_meta=file_meta,
                )

                chunk_counts: Dict[str, int] = {}
                manifestable = set()
                for r in results:
                    chunk_counts[r["file"]] = r["chunk_count"]
                    manifestable.add(r["file"])

                report.files_ingested += len(results)
                report.chunks += sum(r["chunk_count"] for r in results)

                # 解析失败的文件（加密 PDF、损坏的 docx、解析器抛异常）不会出现在
                # results 里，绝不能写进清单 —— 一旦写进去，下次磁盘与清单一致，
                # 就会被永久判定为"未变化"，这个文件从索引里彻底消失且无人察觉。
                # 不写清单，它下次仍是"新增"，自动重试。
                failed = [p for p in batch if p not in manifestable]
                if failed:
                    logger.warning(
                        "  本批 %d 个文件解析失败，不写入清单（下次自动重试）: %s",
                        len(failed), [os.path.basename(p) for p in failed[:5]],
                    )

                # 本批提交：清单落盘，进度到此为止都是持久的
                self.manifest.upsert_many([
                    self._record(path=p,
                                 file_hash=plan.hashes.get(p, ""),
                                 entry=disk.get(p),
                                 chunk_count=chunk_counts.get(p, 0),
                                 ingested_at=now)
                    for p in batch if p in manifestable
                ])

        # ── 阶段 9：收尾核对 ──
        logger.info("Step 4/4: 完成。本轮入库 %d 个文件 / %d 个 chunk",
                    report.files_ingested, report.chunks)
        self._warn_if_manifest_looks_lost(manifest_records, plan)

        return report

    @staticmethod
    def _split_batches(items: List[str], batch_size: int) -> List[List[str]]:
        """按 batch_size 切分；batch_size <= 0 表示不分批。"""
        if batch_size is None or batch_size <= 0:
            return [list(items)]
        return [list(items[i:i + batch_size])
                for i in range(0, len(items), batch_size)]

    def _warn_if_manifest_looks_lost(
        self, manifest_records: Dict[str, Any], plan: SyncPlan
    ) -> None:
        """清单为空但库里有点 → 很可能是清单丢了。

        这种状态下所有文件都会被判为"新增"，于是全部重新嵌入（浪费但不丢数据）。
        不过更该警惕的是它背后的原因。提醒一句，让用户知道 --rebuild-manifest
        才是对的修法。
        """
        if manifest_records or not plan.added:
            return
        try:
            from qdrant_client import QdrantClient
            cfg = self.pipeline.config
            info = QdrantClient(url=cfg.qdrant_url).get_collection(
                cfg.qdrant_collection
            )
            points = getattr(info, "points_count", 0) or 0
        except Exception:
            return
        if points > 0:
            logger.warning(
                "清单是空的，但主库 '%s' 里有 %d 个点 —— 清单可能丢了。"
                "本次会把所有文件当作新增重新嵌入（结果正确但很慢）。"
                "如果确认是清单丢失，用 --rebuild-manifest 重建会快得多。",
                cfg.qdrant_collection, points,
            )

    # ── 清单写入 ─────────────────────────────────────────

    def _renamed_records(
        self,
        plan: SyncPlan,
        disk: Dict[str, DiskEntry],
        old_records: Dict[str, Dict[str, Any]],
        ingested_at: str,
    ) -> List[Dict[str, Any]]:
        """改名项不重新嵌入，其 file_hash / chunk_count 从旧路径继承。

        old_records 必须传阶段 2 读到的快照 —— 旧记录在这条路径上会被删掉，
        若等删完再读清单，继承到的是空值，改名后的文件就丢了哈希，
        下次同步会误判为"内容变了"而白跑一次嵌入。
        """
        records = []
        for old, new in plan.renamed:
            source = old_records.get(old, {})
            records.append(self._record(
                path=new,
                file_hash=source.get("file_hash", ""),
                entry=disk.get(new),
                chunk_count=source.get("chunk_count", 0),
                ingested_at=ingested_at,
            ))
        return records

    def _record(
        self,
        path: str,
        file_hash: str,
        entry: Optional[DiskEntry],
        chunk_count: int,
        ingested_at: str,
    ) -> Dict[str, Any]:
        mtime = entry.mtime if entry else 0
        return {
            "source_file": path,
            "file_hash": file_hash,
            "size": entry.size if entry else 0,
            "mtime": mtime,
            "mtime_iso": (
                datetime.datetime.fromtimestamp(mtime).isoformat() if mtime else ""
            ),
            "chunk_count": chunk_count,
            "pipeline_fingerprint": self.fingerprint,
            "ingested_at": ingested_at,
        }

    # ── 清单重建（兜底）───────────────────────────────────

    def rebuild_manifest(self) -> int:
        """从主 collection 的 payload 反推重建清单。

        清单和数据是「分家」的，这是兜底：清单丢了、或怀疑它和实际库漂移了，
        用它重建。前提是主 collection 里存了 file_hash —— 也就是本改动之后
        入库的数据。
        """
        from qdrant_client import QdrantClient

        cfg = self.pipeline.config
        client = QdrantClient(url=cfg.qdrant_url)
        collection = cfg.qdrant_collection

        if not client.collection_exists(collection):
            raise RuntimeError(f"主 collection '{collection}' 不存在，无法重建清单。")

        by_file: Dict[str, Dict[str, Any]] = {}
        missing_meta = 0
        offset = None
        while True:
            points, offset = client.scroll(
                collection_name=collection,
                limit=2000,
                offset=offset,
                with_payload=["source_file", "file_hash", "pipeline_fingerprint",
                              "file_size", "file_mtime_epoch"],
                with_vectors=False,
            )
            for p in points:
                payload = p.payload or {}
                path = payload.get("source_file")
                if not path:
                    continue
                if "file_size" not in payload or "file_mtime_epoch" not in payload:
                    missing_meta += 1
                rec = by_file.setdefault(path, {
                    "source_file": path,
                    "file_hash": payload.get("file_hash", ""),
                    "size": int(payload.get("file_size", -1)),
                    "mtime": int(payload.get("file_mtime_epoch", -1)),
                    "mtime_iso": payload.get("file_mtime", ""),
                    "pipeline_fingerprint": payload.get("pipeline_fingerprint", ""),
                    "ingested_at": "",
                    "chunk_count": 0,
                })
                rec["chunk_count"] += 1
            if offset is None:
                break

        # 旧数据（本次改动之前入库的）payload 里没有 size/mtime，重建出来的记录
        # 会被判定为"与磁盘不符"，下一轮同步就变成全量重嵌入。与其让用户
        # 困惑于"为什么重建完还在全量跑"，不如在这里直接说清楚。
        if missing_meta:
            raise RuntimeError(
                f"主库中有 {missing_meta} 个点的 payload 没有 file_size / "
                f"file_mtime_epoch 字段（是本次同步功能上线前入库的旧数据），"
                f"无法据此重建清单。\n"
                f"请改用 --recreate 做一次干净的全量重建。"
            )

        self.manifest.drop_collection()
        self.manifest.ensure_collection()
        self.manifest.upsert_many(list(by_file.values()))
        logger.info("清单已从主库重建：%d 个文件", len(by_file))
        return len(by_file)

    # ── 工具 ─────────────────────────────────────────────

    @staticmethod
    def _file_hash(path: str) -> str:
        """文件内容 sha256（分块读，避免把大文件整个读进内存）。"""
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(_HASH_CHUNK), b""):
                digest.update(block)
        return digest.hexdigest()
