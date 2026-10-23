"""
kg_manifest.py — 知识图谱线的本地文件清单（JSON）

对应 Qdrant 线的 manifest_store.py，但存在本地 JSON 文件而不是 Qdrant collection。
为什么这里敢用本地文件而 Qdrant 线不敢：
    图谱线的文档量小（LLM 抽取成本决定的，通常是几百到几千个文件），
    一个 JSON 完全装得下；而且丢了可以 --rebuild-manifest 从服务端文档列表
    按文件名配对重建。Qdrant 线动辄十万文件、每周全量比对，才需要放服务端。

清单记录结构（与 plan_sync 的输入契约对齐，缺字段会被误判为"已修改"）：
    source_file           规范路径（canonical_path）
    file_hash             文件内容 sha256
    size / mtime          廉价预筛
    mtime_iso             给人看
    doc_id                LightRAG 文档 ID（删除时要用）
    upload_name           上传到服务端的文件名（默认是 basename，撞名时加前缀）
    chunks_count          服务端报告的 chunk 数，便于核对
    pipeline_fingerprint  图谱线配置指纹（KG_PIPELINE_VERSION）
    ingested_at           入库时间

写入一律原子替换（tmp + os.replace）：定时任务中途被杀不会留下半个 JSON。
"""

import os
import json
import logging
import tempfile
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

DEFAULT_MANIFEST_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "kg_manifest.json"
)


class KgManifest:
    """本地 JSON 清单：{规范路径: 记录}。"""

    def __init__(self, path: str = DEFAULT_MANIFEST_PATH):
        self.path = path
        self._records: Dict[str, Dict[str, Any]] = None

    # ── 读写 ─────────────────────────────────────────────

    def load(self) -> Dict[str, Dict[str, Any]]:
        """读入全部记录。文件不存在视为空清单（首次运行）。"""
        if self._records is not None:
            return self._records
        if not os.path.exists(self.path):
            self._records = {}
            return self._records
        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._records = data.get("files", {})
        logger.info("KG 清单已加载: %d 个文件 (%s)", len(self._records), self.path)
        return self._records

    def save(self) -> None:
        """原子写回。每个批次提交一次，中断后已提交的进度不丢。"""
        if self._records is None:
            return
        payload = {"version": 1, "files": self._records}
        fd, tmp = tempfile.mkstemp(
            dir=os.path.dirname(self.path) or ".",
            prefix=".kg_manifest_", suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=1)
            os.replace(tmp, self.path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    # ── 便捷操作（都基于 load() 出来的内存副本）─────────────

    def upsert(self, path: str, record: Dict[str, Any]) -> None:
        self.load()[path] = record

    def delete_paths(self, paths: List[str]) -> None:
        records = self.load()
        for p in paths:
            records.pop(p, None)

    def rename_path(self, old: str, new: str) -> None:
        """改名：整条记录挪到新路径下（doc_id / file_hash 原样继承）。"""
        records = self.load()
        if old in records:
            rec = records.pop(old)
            rec["source_file"] = new
            records[new] = rec

    def drop(self) -> None:
        """删除清单文件（--recreate 用）。"""
        self._records = {}
        if os.path.exists(self.path):
            os.unlink(self.path)
            logger.info("KG 清单已删除: %s", self.path)
