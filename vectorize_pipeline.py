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
    embedding_model: str = "bge-small"
    embedding_batch_size: int = 32
    embedding_normalize: bool = True
    embedding_device: Optional[str] = None  # None = auto
    embedding_use_flag: bool = False        # 使用 FlagEmbedding 后端（BGE-M3）

    # ── Qdrant ──
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "documents"
    qdrant_vector_size: int = 0       # 0 = 自动从模型获取
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
    ) -> List[Dict[str, Any]]:
        """
        完整流程：扫描文件 → 解析 → 分块 → 嵌入 → (可选)写入 Qdrant

        Args:
            inputs: 文件或目录路径列表
            store_in_qdrant: 是否写入 Qdrant

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
        all_chunks, chunk_doc_map = self._chunk_documents(parsed_docs)
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

        # 获取 dense + sparse 查询向量
        query_result = embedder.embed(query)
        if isinstance(query_result, dict):
            dense_vec = query_result["dense_vecs"]
        else:
            dense_vec = query_result
        dense_vec = dense_vec.tolist() if hasattr(dense_vec, 'tolist') else dense_vec

        # 尝试获取稀疏查询向量（需要 FlagEmbedding 重新 encode）
        sparse_vec = None
        if embedder.is_flag:
            model = embedder._model
            q_result = model.encode_queries(query, return_dense=True, return_sparse=True)
            if isinstance(q_result, dict):
                lw = q_result.get("lexical_weights")
                if lw is not None:
                    # 单条查询: lw 直接是 defaultdict，非列表
                    weights = lw if isinstance(lw, dict) else (lw[0] if lw else {})
                    if weights:
                        sparse_vec = SparseVector(
                            indices=[int(k) for k in weights.keys()],
                            values=[float(v) for v in weights.values()],
                        )

        # 构建 prefetch 列表
        prefetch = [
            Prefetch(query=dense_vec, using="dense", limit=top_k * 2),
        ]
        if sparse_vec is not None:
            prefetch.append(
                Prefetch(query=sparse_vec, using="sparse", limit=top_k * 2),
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
                    files.append(str(p.resolve()))
                else:
                    logger.debug("Skipped unsupported: %s", p.name)
            elif p.is_dir():
                pattern = "*" if self.config.recursive else "*.*"
                glob_method = p.rglob if self.config.recursive else p.glob
                for f in glob_method(pattern):
                    if f.is_file() and f.suffix.lower() in allowed:
                        files.append(str(f.resolve()))
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
        self, docs: List[ParsedDocument]
    ) -> tuple:  # (all_chunks, chunk_doc_map)
        self._ensure_chunker()
        all_chunks = []
        chunk_doc_map = {}  # chunk_index → doc index

        for di, doc in enumerate(docs):
            doc_meta = {
                "source_file": doc.file_path,
                "file_type": doc.file_type,
                "file_name": Path(doc.file_path).name,
                "file_mtime": datetime.datetime.fromtimestamp(
                    os.path.getmtime(doc.file_path)
                ).isoformat(),
            }

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

        # 检查或创建 collection
        if self._qdrant.collection_exists(collection):
            info = self._qdrant.get_collection(collection)
            existing = getattr(info, 'points_count', 0)
            logger.info("Collection '%s' exists, points: %d", collection, existing)

            if self.config.qdrant_force_recreate:
                logger.info("Force-recreating collection '%s' ...", collection)
                self._qdrant.delete_collection(collection)
                self._create_collection(collection, vec_size)
        else:
            logger.info("Creating collection '%s' (dense=%d + sparse) ...", collection, vec_size)
            self._create_collection(collection, vec_size)

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
                sparse_vec = SparseVector(
                    indices=list(sp.keys()),
                    values=list(sp.values()),
                ) if sp else SparseVector(indices=[], values=[])

                points.append(PointStruct(
                    id=self._hash_text(chunk.text + str(i)),
                    vector={
                        "dense": dense_vec,
                        "sparse": sparse_vec,
                    },
                    payload={
                        "text": chunk.text,
                        "source_file": chunk.metadata.get("source_file", ""),
                        "file_type": chunk.metadata.get("file_type", ""),
                        "chunk_index": chunk.chunk_index,
                        "section_label": chunk.metadata.get("section_label", ""),
                        "file_mtime": chunk.metadata.get("file_mtime", ""),
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

    def _create_collection(self, name: str, vec_size: int):
        from qdrant_client.models import VectorParams, Distance, SparseVectorParams
        self._qdrant.create_collection(
            collection_name=name,
            vectors_config={
                "dense": VectorParams(size=vec_size, distance=Distance.COSINE),
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(),
            },
        )

    # ── 结果组装 ─────────────────────────────────────────

    def _build_results(
        self,
        docs: List[ParsedDocument],
        chunks: List[Chunk],
        vectors: "np.ndarray",
        chunk_doc_map: dict,
    ) -> List[Dict[str, Any]]:
        """按文档分组返回"""
        results = []
        for di, doc in enumerate(docs):
            doc_chunks = []
            doc_vectors = []

            for ci, c in enumerate(chunks):
                if chunk_doc_map.get(ci) == di:
                    doc_chunks.append({
                        "text": c.text,
                        "chunk_index": c.chunk_index,
                        "metadata": c.metadata,
                    })
                    doc_vectors.append(vectors[ci])

            import numpy as np
            results.append({
                "file": doc.file_path,
                "file_type": doc.file_type,
                "metadata": doc.metadata,
                "full_content": doc.content,
                "chunks": doc_chunks,
                "vectors": np.array(doc_vectors) if doc_vectors else np.array([]),
                "chunk_count": len(doc_chunks),
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
        """生成确定性 UUID（相同文本 → 相同 ID，重复 upsert 不产生重复数据）"""
        h = hashlib.md5(text.encode()).hexdigest()
        return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


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
