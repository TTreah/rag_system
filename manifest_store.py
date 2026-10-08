"""
manifest_store.py — 文件清单 collection

在同一个 Qdrant 实例里维护一个「一个文件一个点」的 collection，记录每个文件
当前在库中的版本，作为每周增量同步的比对基准。

为什么不直接拿主 collection 当清单：
    主 collection 是一个 chunk 一个点，10 万文件对应百万级 chunk。每周为了反推
    "哪些文件变了" 要 scroll 上百万个点。文件级清单一遍只有 10 万个点，快一个
    量级——这是它值得单独存在的唯一理由。

清单里存什么（全部由脚本维护，不需要人工编辑）：
    source_file          规范路径，点 ID 也由它派生
    file_hash            文件内容 sha256 —— 权威变更依据
    size / mtime         廉价预筛，用来避免为"没变过的文件"去读内容
    mtime_iso            同上，给人和报表看
    chunk_count          入库 chunk 数，便于核对
    pipeline_fingerprint 分块/嵌入/解析配置指纹，变化即全量重建
    ingested_at          入库时间

关于 last_seen：有意不维护。每周给它刷一遍意味着要把 10 万条清单全量重写，
和"只写变化的文件"直接矛盾。共享盘掉线的误判改用同步流程里的可达性检查 +
删除比例熔断来防（见 sync_index.py）。
"""

import logging
from typing import Any, Dict, List, Optional

from vectorize_pipeline import VectorizePipeline

logger = logging.getLogger(__name__)

DEFAULT_MANIFEST_COLLECTION = "file_manifest"
_SCROLL_PAGE = 1000
_WRITE_BATCH = 500


class FileManifest:
    """文件清单 collection 的读写封装（惰性连接 Qdrant）。"""

    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        collection: str = DEFAULT_MANIFEST_COLLECTION,
    ):
        self.qdrant_url = qdrant_url
        self.collection = collection
        self._client = None

    # ── 生命周期 ─────────────────────────────────────────

    def _ensure_client(self):
        if self._client is None:
            from qdrant_client import QdrantClient
            self._client = QdrantClient(url=self.qdrant_url)
        return self._client

    def ensure_collection(self) -> None:
        """不存在则创建。清单不需要向量，但各版本 Qdrant 对「无向量 collection」
        支持不一，这里统一配一个 1 维哑向量：所有版本都能建，成本可忽略。
        """
        from qdrant_client.models import VectorParams, Distance

        client = self._ensure_client()
        if client.collection_exists(self.collection):
            return
        client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=1, distance=Distance.DOT),
        )
        logger.info("Created manifest collection '%s'", self.collection)

    def drop_collection(self) -> None:
        """删除清单（--recreate 时用）。"""
        client = self._ensure_client()
        if client.collection_exists(self.collection):
            client.delete_collection(self.collection)
            logger.info("Dropped manifest collection '%s'", self.collection)

    # ── 读 ───────────────────────────────────────────────

    def load(self) -> Dict[str, Dict[str, Any]]:
        """全量读取清单 → {规范路径: payload}。"""
        client = self._ensure_client()
        self.ensure_collection()

        records: Dict[str, Dict[str, Any]] = {}
        offset = None
        while True:
            points, offset = client.scroll(
                collection_name=self.collection,
                limit=_SCROLL_PAGE,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for p in points:
                payload = p.payload or {}
                source_file = payload.get("source_file")
                if source_file:
                    records[source_file] = payload
            if offset is None:
                break

        logger.info("Manifest loaded: %d file(s)", len(records))
        return records

    def count(self) -> int:
        client = self._ensure_client()
        if not client.collection_exists(self.collection):
            return 0
        info = client.get_collection(self.collection)
        return getattr(info, "points_count", 0) or 0

    # ── 写 ───────────────────────────────────────────────

    def upsert_many(self, records: List[Dict[str, Any]]) -> None:
        """按 source_file 派生的稳定 ID 写入/覆盖清单记录。"""
        from qdrant_client.models import PointStruct

        if not records:
            return
        client = self._ensure_client()
        self.ensure_collection()

        for i in range(0, len(records), _WRITE_BATCH):
            batch = records[i:i + _WRITE_BATCH]
            points = [
                PointStruct(
                    id=self._point_id(rec["source_file"]),
                    vector=[0.0],
                    payload=rec,
                )
                for rec in batch
            ]
            client.upsert(
                collection_name=self.collection, points=points, wait=True,
            )
        logger.info("Manifest upserted: %d record(s)", len(records))

    def delete_paths(self, source_files: List[str]) -> None:
        """按稳定 ID 直接删除清单记录。

        清单的 ID 就是 hash(路径)，所以这里能精确按 ID 删，不需要过滤、也不需要
        任何 payload 索引。
        """
        from qdrant_client.models import PointIdsList

        if not source_files:
            return
        client = self._ensure_client()
        self.ensure_collection()

        ids = [self._point_id(p) for p in source_files]
        for i in range(0, len(ids), _WRITE_BATCH):
            client.delete(
                collection_name=self.collection,
                points_selector=PointIdsList(points=ids[i:i + _WRITE_BATCH]),
                wait=True,
            )
        logger.info("Manifest deleted: %d record(s)", len(source_files))

    # ── 工具 ─────────────────────────────────────────────

    @staticmethod
    def _point_id(source_file: str) -> str:
        """与主 collection 的 ID 派生方式保持一致，避免两套规则漂移。"""
        return VectorizePipeline._hash_text(f"manifest|{source_file}")
