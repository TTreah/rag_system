"""
kg_sync.py — 知识图谱线的增量同步：共享盘目录 → LightRAG 知识图谱

与 Qdrant 线（sync_index.py）同构，差异规划完全复用：
    scan_disk / plan_sync / SafetyAbort / _file_hash  ←  直接从 sync_index 导入
变的只是「执行」这一层：
    Qdrant 线：删旧点 → 解析/分块/嵌入 → upsert（同步、便宜、秒级）
    图谱线  ：删旧文档 → upload → 等 LLM 抽取完成（异步、贵、分钟级）

图谱线特有的三件事：
    1. 上传一次一个文件，服务端只认文件名 → 撞名要加前缀（_alloc_upload_names）
    2. 同名冲突是 HTTP 409 → 修改过的文件必须先删旧文档、等删除完成、再传
    3. 抽取是异步的 → 每批上传后轮询 track_status，processed 才写清单；
       failed 不写清单，下次运行自动重试（与 Qdrant 线同一原则）

清单是本地 JSON（kg_manifest.py），每个批次提交一次，中断重跑不重复抽取。
服务端 LLM/embedding 模型变化会让 fingerprint 变化 → 自动触发全量重建。
"""

import os
import time
import hashlib
import logging
import datetime
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field

from qdrant_line.sync_index import (
    DiskEntry,
    SyncPlan,
    SafetyAbort,
    scan_disk,
    plan_sync,
    MAX_DELETE_RATIO,
)
from qdrant_line.vectorize_pipeline import canonical_path

from qdrant_line.sync_index import _file_hash  # noqa: E402  与 Qdrant 线共用同一实现
from kg_line.kg_docs_client import (
    LightRAGDocsClient,
    KGApiError,
    TERMINAL_OK,
    TERMINAL_BAD,
    TERMINAL_STATES,
)
from kg_line.kg_manifest import KgManifest
from core.document_parser import DocumentParser
from kg_line.kg_preprocess import enhance_numeric_clauses

logger = logging.getLogger(__name__)


def _maybe_convert_doc(path: str) -> "tuple[str, Optional[str]]":
    """.doc → 临时 .docx（LightRAG 服务端不支持 .doc）。

    返回 (上传用路径, 临时文件路径或 None)。转换只发生在内存中的上传环节，
    原文件不动；清单仍以原始 .doc 路径为键，hash/size/mtime 都记原文件的。
    """
    if not path.lower().endswith(".doc"):
        return path, None
    tmp = DocumentParser._convert_doc_to_docx(path)
    return tmp, tmp

# 图谱线解析/抽取逻辑变更时手动 +1，触发全量重建（对应 Qdrant 线的 PARSER_VERSION）
KG_PIPELINE_VERSION = "1"

# 入库分批大小。LLM 抽取是瓶颈（一个文件几十秒到几分钟），
# 这个值同时是「中断时最多损失多少已上传未确认的进度」。
DEFAULT_KG_BATCH_SIZE = 10

# 轮询节奏与上限
POLL_INTERVAL = 5          # 秒
BATCH_TIMEOUT = 3600       # 一批文件等抽取完成的最长等待
DELETE_TIMEOUT = 300       # 等删除完成的最长等待


# ═══════════════════════════════════════════════════════════════
# 报告
# ═══════════════════════════════════════════════════════════════

@dataclass
class KGSyncReport:
    plan: SyncPlan
    uploaded: int = 0
    deleted: int = 0
    failed: List[str] = field(default_factory=list)
    batches: int = 0
    dry_run: bool = False
    skipped_by_limit: int = 0

    def describe(self) -> str:
        p = self.plan
        lines = [
            "=" * 66,
            f"{'知识图谱增量同步' + ('  [DRY-RUN 未改动任何数据]' if self.dry_run else ''):^66}",
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
                f"  受 --max-files 截断: {self.skipped_by_limit:>6}（再跑一次即继续）"
            )
        if not self.dry_run:
            lines.append("-" * 66)
            lines.append(f"  入库文档  : {self.uploaded:>6}"
                         + (f"   （分 {self.batches} 批）" if self.batches else ""))
            if self.deleted:
                lines.append(f"  清理删除  : {self.deleted:>6}")
            if self.failed:
                lines.append(f"  抽取失败  : {len(self.failed):>6}（未写清单，下次自动重试）")
        if not p.has_changes:
            lines.append("\n  没有任何变化。")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# 同步执行
# ═══════════════════════════════════════════════════════════════

class KGSynchronizer:
    """把共享盘目录同步进 LightRAG 知识图谱，增量维护本地清单。"""

    def __init__(
        self,
        client: LightRAGDocsClient,
        manifest: KgManifest,
        fingerprint: Optional[str] = None,
    ):
        self.client = client
        self.manifest = manifest
        # fingerprint 默认由调用方从 /health 的服务端配置生成（见 make_fingerprint）
        self.fingerprint = fingerprint or KG_PIPELINE_VERSION

    # ── 入口 ─────────────────────────────────────────────

    def run(
        self,
        roots: List[str],
        dry_run: bool = False,
        force: bool = False,
        detect_renames: bool = False,
        max_delete_ratio: float = MAX_DELETE_RATIO,
        batch_size: int = DEFAULT_KG_BATCH_SIZE,
        max_files: Optional[int] = None,
        file_hash_fn: Callable[[str], str] = _file_hash,
        only_files: Optional[List[str]] = None,
    ) -> KGSyncReport:
        """执行一次增量同步。阶段划分与 sync_index.IndexSynchronizer.run 一致，
        安全熔断（共享盘可达性 / 删除比例）原样保留。

        Args:
            only_files: 只同步这几个文件（部分同步）。注意：看不到全集就无法
                判断"谁被删了"，因此此模式下删除与改名检测整体关闭——
                清单里其它文件的旧点一律保留。

        Raises:
            SafetyAbort: 触发保护性熔断，未改动任何数据
        """
        # ── 阶段 0：共享盘可达性（掉线时扫到空集会误判为"全部已删"）──
        for root in roots:
            if not os.path.isdir(root):
                raise SafetyAbort(
                    f"源目录不可访问: {root}\n"
                    f"共享盘掉线 / 未挂载时继续执行会判定「全部文件已删除」"
                    f"并清空图谱，因此中止。"
                )

        # ── 阶段 1：扫盘（只读元数据）──
        disk: Dict[str, DiskEntry] = {}
        for root in roots:
            disk.update(scan_disk(root))
        if not disk:
            raise SafetyAbort(
                f"在 {roots} 下没有扫描到任何受支持的文件，中止以免误判为全删。"
            )

        # ── 阶段 1.5：部分同步过滤 ──
        if only_files:
            wanted = {canonical_path(f) for f in only_files}
            missing = wanted - set(disk)
            if missing:
                logger.warning("以下指定文件未扫到（不存在或类型不支持）: %s",
                               [os.path.basename(p) for p in sorted(missing)])
            disk = {p: e for p, e in disk.items() if p in wanted}
            if not disk:
                raise SafetyAbort("指定的文件一个都没扫到，中止。")

        # ── 阶段 2：读清单 ──
        records = self.manifest.load()

        # ── 阶段 3：差异规划（纯逻辑，与 Qdrant 线共用）──
        plan = plan_sync(
            disk, records, self.fingerprint,
            file_hash_fn=file_hash_fn,
            detect_renames=detect_renames,
        )

        # 部分同步看不到全集，"谁被删了"无从判断——删除与改名整体关闭
        if only_files and (plan.deleted or plan.renamed):
            logger.info("部分同步模式：跳过 %d 个删除、%d 个改名的判定",
                        len(plan.deleted), len(plan.renamed))
            plan.deleted = []
            plan.renamed = []

        # ── 阶段 4：删除比例熔断 ──
        if records and not force:
            budget = max(1, int(len(records) * max_delete_ratio))
            if len(plan.deleted) > budget:
                raise SafetyAbort(
                    f"待删文档数 {len(plan.deleted)} 超过清单总量 "
                    f"{len(records)} 的 {max_delete_ratio:.0%}（{budget} 个）。\n"
                    f"确认这批删除无误后，加 --force 重跑。"
                )

        # ── 阶段 5：--max-files 截断（被截断的下次运行继续）──
        to_ingest = plan.to_ingest
        skipped = 0
        if max_files is not None and max_files < len(to_ingest):
            skipped = len(to_ingest) - max_files
            keep = set(to_ingest[:max_files])
            plan.added = [p for p in plan.added if p in keep]
            plan.modified = [p for p in plan.modified if p in keep]
            plan.hashes = {p: h for p, h in plan.hashes.items() if p in keep}
            to_ingest = plan.to_ingest
            logger.info("--max-files %d：本轮入库 %d 个，其余 %d 个下次继续",
                        max_files, len(to_ingest), skipped)

        report = KGSyncReport(plan=plan, dry_run=dry_run, skipped_by_limit=skipped)

        if dry_run:
            logger.info("DRY-RUN：未改动任何数据")
            return report
        if not plan.has_changes:
            logger.info("没有任何变化，跳过全部写操作")
            return report

        now = datetime.datetime.now().isoformat()

        # 服务端文档列表只拉一次：删除要按它补 doc_id，撞名检测也要用
        server_docs = self.client.list_documents()

        # ── 阶段 6：磁盘上已消失的文件 ──
        if plan.deleted:
            logger.info("Step 1/3: 删除 %d 个已消失文档 ...", len(plan.deleted))
            self._delete_paths(plan.deleted, records, server_docs)
            self.manifest.delete_paths(plan.deleted)
            self.manifest.save()
            report.deleted = len(plan.deleted)

        # ── 阶段 7：改名（内容没变，doc_id 不变，只挪清单记录）──
        # 服务端 file_path 仍是旧名字——只是展示元数据，不影响检索，不值得
        # 为它重跑一遍 LLM 抽取。
        if plan.renamed:
            logger.info("Step 2/3: 改名 %d 项（仅更新清单）...", len(plan.renamed))
            for old, new in plan.renamed:
                self.manifest.rename_path(old, new)
            self.manifest.save()

        # ── 阶段 8：分批上传 + 等抽取完成 ──
        if to_ingest:
            upload_names = self._alloc_upload_names(to_ingest, plan, records, server_docs)
            batches = self._split_batches(to_ingest, batch_size)
            report.batches = len(batches)
            logger.info("Step 3/3: 上传 %d 个文件，分 %d 批（每批 %s 个）...",
                        len(to_ingest), len(batches),
                        batch_size if batch_size > 0 else "全部")

            for bi, batch in enumerate(batches, 1):
                logger.info("  批次 %d/%d：%d 个文件", bi, len(batches), len(batch))
                outcome = self._ingest_batch(batch, plan, upload_names, records)

                for path, res in outcome.items():
                    if res["status"] == TERMINAL_OK:
                        entry = disk.get(path)
                        self.manifest.upsert(path, self._record(
                            path=path,
                            file_hash=plan.hashes.get(path, ""),
                            entry=entry,
                            doc_id=res.get("doc_id", ""),
                            upload_name=upload_names[path],
                            chunks_count=res.get("chunks_count") or 0,
                            ingested_at=now,
                        ))
                        report.uploaded += 1
                    else:
                        # failed / timeout：不写清单，下次运行仍是"新增"，自动重试
                        report.failed.append(path)
                        logger.warning("  抽取未成功 [%s]: %s — %s",
                                       os.path.basename(path),
                                       res["status"], res.get("error_msg", ""))

                # 本批提交：清单落盘，进度到此为止是持久的
                self.manifest.save()

        self._warn_if_drift()
        return report

    # ── 一批文件的上传与确认 ───────────────────────────────

    def _ingest_batch(
        self,
        batch: List[str],
        plan: SyncPlan,
        upload_names: Dict[str, str],
        records: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """上传一批文件并等它们到达终态。

        Returns:
            {path: {"status", "doc_id", "error_msg", "chunks_count"}}
        """
        modified = set(plan.modified)
        pending: Dict[str, Dict[str, Any]] = {}

        for path in batch:
            # 修改过的文件：服务端还有同名旧文档，必须先删等删完，否则 409
            if path in modified:
                self._delete_paths([path], records, None)

            # 优先本地解析 + 数值条款预处理后走文本直传：
            # 1) .doc 由本地 parser 转换处理，不再需要为上传单独转 docx
            # 2) 预处理标记数值条款，提高 LLM 抽取时的数字保留率
            # 本地解析失败才退回原始文件上传
            text_payload = None
            try:
                parsed = DocumentParser().parse(path)
                if parsed.content.strip():
                    text_payload = enhance_numeric_clauses(
                        parsed.content, doc_title=os.path.basename(path))
            except Exception as e:
                logger.debug("本地解析失败，退回文件上传: %s (%s)", path, e)

            upload_path, tmp = (None, None) if text_payload \
                else _maybe_convert_doc(path)
            try:
                try:
                    resp = self._do_upload(text_payload, upload_path,
                                           upload_names[path])
                except KGApiError as e:
                    if e.is_conflict:
                        # 上次运行留下的同名残档（failed / 半截）：删掉重传一次
                        logger.info("  同名冲突，删除服务端残档后重传: %s",
                                    upload_names[path])
                        self._delete_by_upload_name(upload_names[path])
                        resp = self._do_upload(text_payload, upload_path,
                                               upload_names[path])
                    else:
                        raise
            except KGApiError as e:
                pending[path] = {"status": "upload_failed", "error_msg": str(e)}
                continue
            finally:
                if tmp:
                    try:
                        os.unlink(tmp)
                    except OSError:
                        pass

            pending[path] = {"track_id": resp.get("track_id", ""),
                             "upload_name": upload_names[path]}

        return self._await_terminal(pending)

    def _do_upload(self, text_payload, upload_path, upload_name):
        """统一上传入口：有预处理文本走文本直传，否则文件上传。"""
        if text_payload is not None:
            return self.client.insert_text(text_payload, file_source=upload_name)
        return self.client.upload(upload_path, upload_name=upload_name)

    def _await_terminal(
        self, pending: Dict[str, Dict[str, Any]], timeout: int = BATCH_TIMEOUT
    ) -> Dict[str, Dict[str, Any]]:
        """轮询直到每个文件都到 processed/failed（或超时）。

        主路径用 track_id 查；track 查不到时（服务重启丢跟踪等）回退到
        按 upload_name 在文档列表里找。
        """
        deadline = time.time() + timeout
        unfinished = {p: dict(info) for p, info in pending.items()
                      if info.get("track_id")}
        done = {p: info for p, info in pending.items() if not info.get("track_id")}

        while unfinished and time.time() < deadline:
            time.sleep(POLL_INTERVAL)
            for path, info in list(unfinished.items()):
                doc = self._find_doc(info["track_id"], info["upload_name"])
                if doc is None:
                    continue
                status = doc.get("status", "")
                if status in TERMINAL_STATES:
                    done[path] = {
                        "status": status,
                        "doc_id": doc.get("id", ""),
                        "error_msg": doc.get("error_msg") or "",
                        "chunks_count": doc.get("chunks_count") or 0,
                    }
                    del unfinished[path]

            if unfinished:
                logger.info("  等待抽取完成：还剩 %d 个 ...", len(unfinished))

        for path in unfinished:
            done[path] = {"status": "timeout",
                          "error_msg": f"超过 {timeout}s 未到终态"}
        return done

    def _find_doc(self, track_id: str, upload_name: str) -> Optional[Dict[str, Any]]:
        """按 track_id 找到该文件对应的文档记录；找不到返回 None（继续等）。"""
        try:
            tracked = self.client.track_status(track_id)
            docs = tracked.get("documents", [])
        except KGApiError:
            docs = []
        if not docs:
            # 回退：按上传名在服务端列表里找
            for d in self.client.list_documents():
                if d.get("file_path") == upload_name:
                    return d
            return None
        return docs[0]

    # ── 删除 ─────────────────────────────────────────────

    def _delete_paths(
        self,
        paths: List[str],
        records: Dict[str, Dict[str, Any]],
        server_docs: Optional[List[Dict[str, Any]]],
    ) -> None:
        """按清单里的 doc_id 删除；清单缺 doc_id 时按 upload_name 从服务端补。"""
        doc_ids = []
        if server_docs is None:
            server_docs = self.client.list_documents()
        by_name = {d.get("file_path"): d.get("id") for d in server_docs}

        for path in paths:
            rec = records.get(path, {})
            doc_id = rec.get("doc_id") or by_name.get(rec.get("upload_name", ""))
            if doc_id:
                doc_ids.append(doc_id)
            else:
                logger.warning("  找不到服务端文档，跳过删除: %s", path)

        if not doc_ids:
            return
        results = self.client.delete_documents(doc_ids)
        for r in results:
            if r.get("status") == "busy":
                raise SafetyAbort(
                    "LightRAG 管线正忙，删除请求被拒。等后台任务跑完再重试。"
                )
        self._wait_deleted(doc_ids)

    def _delete_by_upload_name(self, upload_name: str) -> None:
        docs = self.client.list_documents()
        ids = [d["id"] for d in docs
               if d.get("file_path") == upload_name and d.get("id")]
        if ids:
            self.client.delete_documents(ids)
            self._wait_deleted(ids)

    def _wait_deleted(self, doc_ids: List[str], timeout: int = DELETE_TIMEOUT) -> None:
        """删除是异步的，同名重传前必须确认文档真的没了。"""
        deadline = time.time() + timeout
        remaining = set(doc_ids)
        while remaining and time.time() < deadline:
            time.sleep(POLL_INTERVAL)
            existing = {d.get("id") for d in self.client.list_documents()}
            remaining &= existing
        if remaining:
            raise KGApiError(
                f"删除 {len(remaining)} 个文档超过 {timeout}s 仍未完成，"
                f"请检查 LightRAG 服务状态后重跑。"
            )

    # ── 上传名分配（撞名加前缀）─────────────────────────────

    @staticmethod
    def _alloc_upload_names(
        to_ingest: List[str],
        plan: SyncPlan,
        records: Dict[str, Dict[str, Any]],
        server_docs: List[Dict[str, Any]],
    ) -> Dict[str, str]:
        """为每个待入库文件分配服务端文件名。

        服务端只认文件名，两个不同目录下的同名文件会 409。默认用 basename；
        与「本次不会删掉的既有文档」或「同批其它文件」撞名时，
        加路径哈希前缀（如 a1b2c3d4__报告.pdf），稳定且可读。
        """
        going_away = set(plan.to_ingest) | set(plan.deleted)
        taken = set()
        for d in server_docs:
            name = d.get("file_path", "")
            if name:
                taken.add(name)
        # 本次要删/要重建的文档占的名字会释放出来
        for path, rec in records.items():
            if path in going_away and rec.get("upload_name") in taken:
                taken.discard(rec["upload_name"])

        names: Dict[str, str] = {}
        for path in sorted(to_ingest):
            base = os.path.basename(path)
            # .doc 会以转换后的 .docx 上传，名字按服务端实际看到的来分配
            if path.lower().endswith(".doc"):
                base = os.path.splitext(base)[0] + ".docx"
            if base not in taken:
                names[path] = base
            else:
                h = hashlib.md5(path.encode()).hexdigest()[:8]
                names[path] = f"{h}__{base}"
                logger.info("  撞名改名上传: %s → %s", base, names[path])
            taken.add(names[path])
        return names

    # ── 清单重建（兜底）───────────────────────────────────

    def rebuild_manifest(
        self,
        roots: List[str],
        file_hash_fn: Callable[[str], str] = _file_hash,
    ) -> int:
        """清单丢了，从服务端文档列表按文件名配对磁盘文件，重建本地清单。

        只有「服务端文件名 ↔ 磁盘路径」一一对应的才敢配对（默认不撞名时
        服务端文件名就是 basename）。配不上的跳过并告警——那些文件下次
        同步会按"新增"重新抽取，安全但费 token。
        """
        disk: Dict[str, DiskEntry] = {}
        for root in roots:
            if not os.path.isdir(root):
                raise SafetyAbort(f"源目录不可访问: {root}")
            disk.update(scan_disk(root))

        docs = [d for d in self.client.list_documents()
                if d.get("status") == TERMINAL_OK]

        # basename → 磁盘路径（有歧义的整组跳过）
        by_base: Dict[str, List[str]] = {}
        for path in disk:
            by_base.setdefault(os.path.basename(path), []).append(path)
        # 补充：.doc 上传时服务端记的是转换后的 .docx 名，反向补一条映射
        for path in disk:
            if path.lower().endswith(".doc"):
                stem = os.path.splitext(os.path.basename(path))[0]
                by_base.setdefault(stem + ".docx", []).append(path)

        now = datetime.datetime.now().isoformat()
        adopted = 0
        for d in docs:
            name = d.get("file_path", "")
            candidates = by_base.get(os.path.basename(name), [])
            if len(candidates) != 1:
                logger.warning("  无法唯一配对，跳过: %s（磁盘候选 %d 个）",
                               name, len(candidates))
                continue
            path = candidates[0]
            entry = disk[path]
            self.manifest.upsert(path, self._record(
                path=path,
                file_hash=file_hash_fn(path),
                entry=entry,
                doc_id=d.get("id", ""),
                upload_name=name,
                chunks_count=d.get("chunks_count") or 0,
                ingested_at=now,
            ))
            adopted += 1

        self.manifest.save()
        logger.info("KG 清单已重建：%d 个文件", adopted)
        return adopted

    # ── 收尾核对 ─────────────────────────────────────────

    def _warn_if_drift(self) -> None:
        """服务端 processed 数与清单数不一致 → 提示可能漂移。"""
        try:
            counts = self.client.status_counts()
        except KGApiError:
            return
        server_processed = counts.get(TERMINAL_OK, 0)
        manifest_count = len(self.manifest.load())
        if server_processed != manifest_count:
            logger.warning(
                "服务端 processed 文档 %d 个，本地清单 %d 个，二者不一致。"
                "若服务端被其它途径改过，用 --rebuild-manifest 重建清单。",
                server_processed, manifest_count,
            )

    # ── 工具 ─────────────────────────────────────────────

    def _record(self, path: str, file_hash: str, entry: Optional[DiskEntry],
                doc_id: str, upload_name: str, chunks_count: int,
                ingested_at: str) -> Dict[str, Any]:
        mtime = entry.mtime if entry else 0
        return {
            "source_file": path,
            "file_hash": file_hash,
            "size": entry.size if entry else 0,
            "mtime": mtime,
            "mtime_iso": (
                datetime.datetime.fromtimestamp(mtime).isoformat() if mtime else ""
            ),
            "doc_id": doc_id,
            "upload_name": upload_name,
            "chunks_count": chunks_count,
            "pipeline_fingerprint": self.fingerprint,
            "ingested_at": ingested_at,
        }

    @staticmethod
    def _split_batches(items: List[str], batch_size: int) -> List[List[str]]:
        if batch_size is None or batch_size <= 0:
            return [list(items)]
        return [list(items[i:i + batch_size])
                for i in range(0, len(items), batch_size)]


def make_fingerprint(health: Dict[str, Any]) -> str:
    """由服务端配置生成指纹。

    图谱是 LLM 抽出来的，换个模型等于换了"解析器"：旧图谱的实体/关系
    和新模型产出的不可比。把模型名编进指纹，服务端换模型后下次同步
    自动全量重建，不用人记。
    """
    cfg = health.get("configuration", {})
    parts = [
        "kg", KG_PIPELINE_VERSION,
        str(cfg.get("llm_model", "")),
        str(cfg.get("embedding_model", "")),
    ]
    return hashlib.md5("|".join(parts).encode()).hexdigest()[:16]
