"""
vectorize_pipeline.py — 文档向量化流水线
编排 解析 → 分块 → 向量化 → 入库(Qdrant) 的完整流程。

使用方式:
    python vectorize_pipeline.py <file_or_dir> [<file_or_dir> ...]

    或作为模块使用:
    from vectorize_pipeline import VectorizePipeline
    pipeline = VectorizePipeline()
    pipeline.run(["doc1.pdf", "doc2.docx"])

流程:
    1. DocumentParser  →  识别文件类型，提取文本，输出 ParsedDocument
    2. TextChunker     →  按策略切分文本，输出 Chunk 列表
    3. TextEmbedder    →  向量化 chunk，输出 (n, dim) 矩阵
    4. QdrantClient    →  存入 Qdrant collection，支持后续相似检索
"""

import os
import uuid
import hashlib
import logging
import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import numpy as np

from document_parser import DocumentParser, ParsedDocument
from text_chunker import TextChunker, Chunk
from embedder import TextEmbedder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# 路径与配置指纹
# ═══════════════════════════════════════════════════════════════

# 解析逻辑变更时手动 +1，使 pipeline_fingerprint 变化、触发全量重建
PARSER_VERSION = "1"


def canonical_path(p) -> str:
    """路径的规范形式，入库与删除必须共用这一个函数。

    Qdrant 的 KEYWORD MatchValue 是精确字符串匹配：写库和删除时路径形式只要有
    一点差异（大小写、正反斜杠、残留的 ..），删除就会静默失效——删掉 0 个点，
    而且不报错。这是最隐蔽的一类故障。

    刻意不用 Path.resolve()：它会做 realpath 系统调用。10 万文件走 SMB 就是
    10 万次额外网络往返。abspath + normpath 是纯字符串处理，不碰文件系统。

    注意 normcase 在 Windows 上会小写化：大小写不同但实际是同一个文件（Windows
    共享盘的常态）会被正确归并。代价是基于 Linux 的、区分大小写的 SMB 共享上，
    两个仅大小写不同的文件会被当成同一个。
    """
    return os.path.normcase(os.path.normpath(os.path.abspath(str(p))))


def pipeline_fingerprint(config: "PipelineConfig") -> str:
    """分块 / 嵌入 / 解析配置的指纹。

    任一环节变化都会让已入库的向量过期，但文件本身一个字节没变，靠文件哈希
    比对查不出来。把它写进 payload，检测到不一致就按全量重建处理，避免新旧
    两种切法的 chunk 混在同一个 collection 里、检索质量静默下降。
    """
    parts = [
        "parser", PARSER_VERSION,
        "chunk", str(config.chunk_strategy), str(config.chunk_size),
        str(config.chunk_overlap), str(config.min_chunk_size),
        "embed", str(config.embedding_model), str(config.embedding_use_flag),
    ]
    return hashlib.md5("|".join(parts).encode()).hexdigest()[:16]


# ═══════════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════════

@dataclass
class PipelineConfig:
    """流水线可调参数"""
    # ── 分块 ──
    chunk_strategy: str = "recursive"   # fixed / recursive / sentence / markdown
    chunk_size: int = 512
    chunk_overlap: int = 64
    min_chunk_size: int = 200

    # ── 嵌入 ──
    embedding_model: str = "BAAI/bge-m3"
    embedding_batch_size: int = 32
    embedding_normalize: bool = True
    embedding_device: Optional[str] = None  # None = auto
    embedding_use_flag: bool = False        # 使用 FlagEmbedding 后端（BGE-M3）

    # ── Qdrant ──
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "documents"
    # 稠密维度始终由 embedder 实际输出决定（dense_vectors.shape[1]），无需配置
    qdrant_force_recreate: bool = False

    # ── 通用 ──
    recursive: bool = True             # 是否递归扫描目录
    supported_extensions: Optional[List[str]] = None  # None = 全部支持类型


# ═══════════════════════════════════════════════════════════════
# 流水线
# ═══════════════════════════════════════════════════════════════

class VectorizePipeline:
    """
    end-to-end 文档向量化流水线。

    使用示例:
        config = PipelineConfig(chunk_size=512, chunk_strategy="recursive")
        pipeline = VectorizePipeline(config)

        # 本地模式（不连接 Qdrant，仅返回 chunks + vectors）
        results = pipeline.run_local(["docs/"])
        for doc in results:
            print(f"{doc['file']}: {len(doc['chunks'])} chunks, "
                  f"vectors shape={doc['vectors'].shape}")

        # 完整模式（写入 Qdrant）
        pipeline.run(["docs/"])

        # 查询 Qdrant
        results = pipeline.search("你的问题", top_k=5)
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self._parser: Optional[DocumentParser] = None
        self._chunker: Optional[TextChunker] = None
        self._embedder: Optional[TextEmbedder] = None
        self._qdrant: Optional[Any] = None

    # ── 完整流程 ─────────────────────────────────────────

    def run(
        self,
        inputs: List[str],
        store_in_qdrant: bool = True,
        file_meta: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        完整流程：扫描文件 → 解析 → 分块 → 嵌入 → (可选)写入 Qdrant

        Args:
            inputs: 文件或目录路径列表
            store_in_qdrant: 是否写入 Qdrant
            file_meta: 按规范路径提供的额外元信息，会合并进该文件所有 chunk 的
                metadata 并写入 payload。增量同步用它注入 file_hash。

        Returns:
            [{"file": ..., "chunks": ..., "vectors": ...}, ...]
        """
        # Step 0: 收集文件
        files = self._collect_files(inputs)
        if not files:
            logger.warning("No files found to process.")
            return []
        logger.info(f"Found {len(files)} file(s) to process")

        # Step 1: 解析文档
        logger.info("=" * 50)
        logger.info("Step 1/4: Parsing documents ...")
        parsed_docs = self._parse_documents(files)
        logger.info("Parsed %d document(s), %d total sections",
                     len(parsed_docs),
                     sum(len(d.sections) for d in parsed_docs))

        # Step 2: 分块
        logger.info("=" * 50)
        logger.info("Step 2/4: Chunking texts ...")
        all_chunks, chunk_doc_map = self._chunk_documents(parsed_docs, file_meta)
        logger.info("Created %d chunks", len(all_chunks))

        # Step 3: 向量化
        logger.info("=" * 50)
        logger.info("Step 3/4: Embedding chunks (dense + sparse) ...")
        dense_vectors, sparse_vectors = self._embed_chunks(all_chunks)
        logger.info("Dense embedded to shape %s, %d sparse vectors",
                     dense_vectors.shape, len(sparse_vectors))

        # Step 4: 入库 Qdrant (optional)
        if store_in_qdrant:
            logger.info("=" * 50)
            logger.info("Step 4/4: Storing to Qdrant ...")
            self._store_to_qdrant(all_chunks, dense_vectors, sparse_vectors, chunk_doc_map)
            logger.info("Stored %d vectors to Qdrant collection '%s'",
                         len(all_chunks), self.config.qdrant_collection)

        # 组装结果
        return self._build_results(parsed_docs, all_chunks, dense_vectors, chunk_doc_map)

    def run_local(self, inputs: List[str]) -> List[Dict[str, Any]]:
        """仅本地处理，不连 Qdrant — 获取 chunks + vectors"""
        return self.run(inputs, store_in_qdrant=False)

    # ── 检索（需要 Qdrant 中已有数据）─────────────────────

    def search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        在已入库的 Qdrant collection 中混合检索（dense + sparse → RRF fusion）。

        Returns:
            [{"text": ..., "score": ..., "metadata": {...}}, ...]
        """
        from qdrant_client.models import Prefetch, FusionQuery, Fusion, SparseVector

        self._ensure_qdrant()
        embedder = self._ensure_embedder()

        # 一次编码同时拿到 dense + sparse（查询侧专用路径，带检索指令）
        dense, sparse_map = embedder.embed_query(query)

        # 构建 prefetch 列表
        prefetch = [
            Prefetch(query=dense.tolist(), using="dense", limit=top_k * 2),
        ]
        if sparse_map:
            prefetch.append(
                Prefetch(
                    query=SparseVector(
                        indices=list(sparse_map.keys()),
                        values=list(sparse_map.values()),
                    ),
                    using="sparse",
                    limit=top_k * 2,
                ),
            )

        response = self._qdrant.query_points(
            collection_name=self.config.qdrant_collection,
            prefetch=prefetch,
            query=FusionQuery(fusion=Fusion.RRF),
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=True,
        )

        results = []
        for point in (response.points or []):
            payload = point.payload or {}
            results.append({
                "id": point.id,
                "score": point.score,
                "text": payload.get("text", ""),
                "metadata": payload.get("metadata", {}),
                "source_file": payload.get("source_file", ""),
            })
        return results

    # ── 文件收集 ─────────────────────────────────────────

    def _collect_files(self, inputs: List[str]) -> List[str]:
        """展开目录，按后缀过滤"""
        parser_cls = DocumentParser
        allowed = set(self.config.supported_extensions or parser_cls.supported_extensions())

        files = []
        for inp in inputs:
            p = Path(inp)
            if not p.exists():
                logger.warning("Path not found: %s", inp)
                continue

            if p.is_file():
                if p.suffix.lower() in allowed:
                    files.append(canonical_path(p))
                else:
                    logger.debug("Skipped unsupported: %s", p.name)
            elif p.is_dir():
                pattern = "*" if self.config.recursive else "*.*"
                glob_method = p.rglob if self.config.recursive else p.glob
                for f in glob_method(pattern):
                    if f.is_file() and f.suffix.lower() in allowed:
                        files.append(canonical_path(f))
            else:
                logger.warning("Skipped: %s", inp)

        return sorted(set(files))  # 去重排序

    # ── 解析 ─────────────────────────────────────────────

    def _parse_documents(self, files: List[str]) -> List[ParsedDocument]:
        self._ensure_parser()
        docs = []
        for fp in files:
            try:
                doc = self._parser.parse(fp)
                docs.append(doc)
                logger.debug("  Parsed: %s → %d sections", fp, len(doc.sections))
            except Exception as e:
                logger.error("Failed to parse %s: %s", fp, e)
        return docs

    # ── 分块 ─────────────────────────────────────────────

    def _chunk_documents(
        self,
        docs: List[ParsedDocument],
        file_meta: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> tuple:  # (all_chunks, chunk_doc_map)
        self._ensure_chunker()
        all_chunks = []
        chunk_doc_map = {}  # chunk_index → doc index

        for di, doc in enumerate(docs):
            # 一次 stat 同时拿到 size 和 mtime。这两个值会随 chunk 一起写进 payload：
            # size/mtime 是 --rebuild-manifest 从主库里反推清单的唯一依据，
            # 缺了就重建不出可比对的记录（会退化成全量重嵌入）。
            st = os.stat(doc.file_path)
            doc_meta = {
                "source_file": doc.file_path,
                "file_type": doc.file_type,
                "file_name": Path(doc.file_path).name,
                "file_size": int(st.st_size),
                "file_mtime_epoch": int(st.st_mtime),
                "file_mtime": datetime.datetime.fromtimestamp(st.st_mtime).isoformat(),
                "pipeline_fingerprint": pipeline_fingerprint(self.config),
            }
            if file_meta:
                doc_meta.update(file_meta.get(doc.file_path, {}))

            if doc.sections:
                chunks = self._chunker.split_from_sections(doc.sections, doc_meta)
            else:
                chunks = self._chunker.split(doc.content, doc_meta)

            for c in chunks:
                # 确保 chunk 有唯一 ID
                c.metadata.setdefault("doc_id", self._hash_path(doc.file_path))
                c.metadata.setdefault("chunk_id", str(uuid.uuid4()))
                all_chunks.append(c)
                chunk_doc_map[len(all_chunks) - 1] = di

            logger.debug("  Chunked [%s] → %d chunks", Path(doc.file_path).name, len(chunks))

        return all_chunks, chunk_doc_map

    # ── 嵌入 ─────────────────────────────────────────────

    def _embed_chunks(self, chunks: List[Chunk]):
        """返回 (dense_vectors, sparse_vectors) 元组"""
        embedder = self._ensure_embedder()
        return embedder.embed_chunks_hybrid(chunks, batch_size=self.config.embedding_batch_size)

    # ── Qdrant 存储 ──────────────────────────────────────

    def _store_to_qdrant(
        self,
        chunks: List[Chunk],
        dense_vectors: "np.ndarray",
        sparse_vectors: list,
        chunk_doc_map: dict,
    ):
        self._ensure_qdrant()

        collection = self.config.qdrant_collection
        vec_size = dense_vectors.shape[1]

        # 只在真的产出稀疏向量时才建稀疏索引，避免留下无用索引
        with_sparse = any(sp for sp in sparse_vectors)

        # 检查或创建 collection
        if self._qdrant.collection_exists(collection):
            info = self._qdrant.get_collection(collection)
            existing = getattr(info, 'points_count', 0)
            logger.info("Collection '%s' exists, points: %d", collection, existing)

            if self.config.qdrant_force_recreate:
                logger.info("Force-recreating collection '%s' ...", collection)
                self._qdrant.delete_collection(collection)
                self._create_collection(collection, vec_size, with_sparse)
            else:
                self._assert_dim_matches(collection, info, vec_size)
        else:
            logger.info("Creating collection '%s' (dense=%d, sparse=%s) ...",
                        collection, vec_size, with_sparse)
            self._create_collection(collection, vec_size, with_sparse)

        # 已有 collection 也要补建索引（新建的已在 _create_collection 里建好）
        self.ensure_payload_indexes(collection)

        # 批量上传
        from qdrant_client.models import PointStruct, SparseVector

        batch_size = 100
        total = len(chunks)
        ingested_at = datetime.datetime.now().isoformat()

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            points = []

            for i in range(start, end):
                chunk = chunks[i]
                dense_vec = dense_vectors[i].tolist()
                sp = sparse_vectors[i] if i < len(sparse_vectors) else {}

                # 空稀疏向量不写入：Qdrant 允许命名向量按点缺失，
                # 传空 SparseVector 反而可能被拒
                vector: Dict[str, Any] = {"dense": dense_vec}
                if sp:
                    vector["sparse"] = SparseVector(
                        indices=list(sp.keys()),
                        values=list(sp.values()),
                    )

                points.append(PointStruct(
                    id=self._point_id(chunk),
                    vector=vector,
                    payload={
                        "text": chunk.text,
                        "source_file": chunk.metadata.get("source_file", ""),
                        "file_type": chunk.metadata.get("file_type", ""),
                        "chunk_index": chunk.chunk_index,
                        "section_label": chunk.metadata.get("section_label", ""),
                        "file_mtime": chunk.metadata.get("file_mtime", ""),
                        # 供 --rebuild-manifest 从主库反推清单
                        "file_hash": chunk.metadata.get("file_hash", ""),
                        "pipeline_fingerprint": chunk.metadata.get("pipeline_fingerprint", ""),
                        "ingested_at": ingested_at,
                        "metadata": chunk.metadata,
                    },
                ))

            self._qdrant.upsert(
                collection_name=collection,
                points=points,
                wait=True,
            )

            if total > batch_size:
                logger.info("  Upserted %d/%d chunks ...", end, total)

    def _create_collection(self, name: str, vec_size: int, with_sparse: bool = True):
        from qdrant_client.models import VectorParams, Distance, SparseVectorParams
        kwargs: Dict[str, Any] = {
            "collection_name": name,
            "vectors_config": {
                "dense": VectorParams(size=vec_size, distance=Distance.COSINE),
            },
        }
        if with_sparse:
            kwargs["sparse_vectors_config"] = {"sparse": SparseVectorParams()}
        self._qdrant.create_collection(**kwargs)
        self.ensure_payload_indexes(name)

    def ensure_payload_indexes(self, collection: Optional[str] = None) -> None:
        """确保 source_file 上有 KEYWORD 索引（幂等）。

        增量同步的"按文件删除"依赖 source_file 过滤。没有索引时 Qdrant 会退化为
        全表扫描，10 万文件级会慢到不可用。
        """
        from qdrant_client.models import PayloadSchemaType
        self._ensure_qdrant()
        self._qdrant.create_payload_index(
            collection_name=collection or self.config.qdrant_collection,
            field_name="source_file",
            field_schema=PayloadSchemaType.KEYWORD,
            wait=True,
        )

    def delete_by_source_files(
        self, source_files: List[str], batch_size: int = 500
    ) -> int:
        """按 source_file 精确删除主 collection 中的点，返回删除的文件数。

        改文件必须先调用它再重新入库：文件改短之后，旧的尾部 chunk 没有新点去
        覆盖（点 ID 只到新的最大 chunk_index），不删就会残留检索得到、内容却
        已经不存在的"幽灵文本块"。
        """
        from qdrant_client.models import (
            Filter, FieldCondition, MatchAny, FilterSelector,
        )
        self._ensure_qdrant()
        if not source_files:
            return 0
        self.ensure_payload_indexes()

        for i in range(0, len(source_files), batch_size):
            batch = source_files[i:i + batch_size]
            self._qdrant.delete(
                collection_name=self.config.qdrant_collection,
                points_selector=FilterSelector(filter=Filter(must=[
                    FieldCondition(key="source_file", match=MatchAny(any=batch)),
                ])),
                wait=True,
            )
        logger.info("Deleted points for %d file(s)", len(source_files))
        return len(source_files)

    def repoint_source_file(self, old: str, new: str) -> None:
        """把某文件的点整体改挂到新路径（改名优化，不重新嵌入）。

        注意：点 ID 仍是按旧路径生成的，之后不会被按 ID 查到——同步流程一律用
        source_file 过滤定位，不依赖 ID，所以无影响。
        """
        from qdrant_client.models import (
            Filter, FieldCondition, MatchValue, FilterSelector,
        )
        self._ensure_qdrant()
        self.ensure_payload_indexes()
        self._qdrant.set_payload(
            collection_name=self.config.qdrant_collection,
            payload={"source_file": new},
            points=FilterSelector(filter=Filter(must=[
                FieldCondition(key="source_file", match=MatchValue(value=old)),
            ])),
            wait=True,
        )

    def _assert_dim_matches(self, collection: str, info: Any, vec_size: int) -> None:
        """入库前校验已有 collection 的稠密维度。

        不做校验的话，维度不符要等到 upsert 才抛 Qdrant 的晦涩报错。
        典型翻车场景：用默认 bge-small(384) 入库，再用 rag_search 的 bge-m3(1024) 检索。
        """
        params = getattr(getattr(info, "config", None), "params", None)
        vectors = getattr(params, "vectors", None) if params else None
        existing_size = getattr(vectors, "size", None)
        if isinstance(vectors, dict):  # 命名向量配置
            existing_size = getattr(vectors.get("dense"), "size", None)
        if existing_size and existing_size != vec_size:
            raise ValueError(
                f"Collection '{collection}' 的稠密维度是 {existing_size}，"
                f"当前 embedder 产出 {vec_size} 维，二者不匹配。"
                f"请改用一致的 embedding_model，或设 qdrant_force_recreate=True 重建。"
            )

    # ── 结果组装 ─────────────────────────────────────────

    def _build_results(
        self,
        docs: List[ParsedDocument],
        chunks: List[Chunk],
        vectors: "np.ndarray",
        chunk_doc_map: dict,
    ) -> List[Dict[str, Any]]:
        """按文档分组返回（单趟分组，避免 docs×chunks 的双重循环）"""
        grouped: Dict[int, List[int]] = {di: [] for di in range(len(docs))}
        for ci in range(len(chunks)):
            di = chunk_doc_map.get(ci)
            if di is not None:
                grouped[di].append(ci)

        results = []
        for di, doc in enumerate(docs):
            idxs = grouped[di]
            results.append({
                "file": doc.file_path,
                "file_type": doc.file_type,
                "metadata": doc.metadata,
                "full_content": doc.content,
                "chunks": [{
                    "text": chunks[ci].text,
                    "chunk_index": chunks[ci].chunk_index,
                    "metadata": chunks[ci].metadata,
                } for ci in idxs],
                "vectors": np.array([vectors[ci] for ci in idxs]) if idxs else np.array([]),
                "chunk_count": len(idxs),
            })

        return results

    # ── 懒加载 ───────────────────────────────────────────

    def _ensure_parser(self) -> DocumentParser:
        if self._parser is None:
            self._parser = DocumentParser()
        return self._parser

    def _ensure_chunker(self) -> TextChunker:
        if self._chunker is None:
            self._chunker = TextChunker(
                strategy=self.config.chunk_strategy,
                chunk_size=self.config.chunk_size,
                overlap=self.config.chunk_overlap,
                min_chunk_size=self.config.min_chunk_size,
            )
        return self._chunker

    def _ensure_embedder(self) -> TextEmbedder:
        if self._embedder is None:
            self._embedder = TextEmbedder(
                model_name=self.config.embedding_model,
                batch_size=self.config.embedding_batch_size,
                normalize=self.config.embedding_normalize,
                device=self.config.embedding_device,
                use_flag_embedding=self.config.embedding_use_flag,
            )
        return self._embedder

    def _ensure_qdrant(self) -> Any:
        if self._qdrant is None:
            from qdrant_client import QdrantClient
            self._qdrant = QdrantClient(url=self.config.qdrant_url)
            logger.info("Connected to Qdrant at %s", self.config.qdrant_url)
        return self._qdrant

    # ── 工具 ─────────────────────────────────────────────

    @staticmethod
    def _hash_path(path: str) -> str:
        return hashlib.md5(path.encode()).hexdigest()[:12]

    @staticmethod
    def _hash_text(text: str) -> str:
        """生成确定性 UUID（相同输入 → 相同 ID）。"""
        h = hashlib.md5(text.encode()).hexdigest()
        return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"

    @staticmethod
    def _point_id(chunk: Chunk) -> str:
        """由 (来源文件, 块内序号) 生成稳定点 ID。

        刻意不用全局下标：那样新增一份文件会平移其后所有 chunk 的位置，
        导致整库 ID 变化、重复写入，增量更新也就无从谈起。
        以文件为单位的块序号与全局顺序无关，因此：
          - 重跑同一份文件      → ID 不变，原地覆盖，幂等
          - 文件内容被修改      → 同 ID 覆盖为最新内容
          - 新增其它文件        → 既有文件的 ID 全部不受影响
        """
        key = f"{chunk.metadata.get('source_file', '')}|{chunk.chunk_index}"
        return VectorizePipeline._hash_text(key)


# ═══════════════════════════════════════════════════════════════
# 自检
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # 纯库模块，无 CLI。使用方式见 run_news.py / rag_search.py
    print("vectorize_pipeline.py — library module, no CLI.")
    print("Usage:")
    print("  from vectorize_pipeline import VectorizePipeline, PipelineConfig")
    print("  pipeline = VectorizePipeline(PipelineConfig(...))")
    print("  pipeline.run(['/path/to/docs/'])")
