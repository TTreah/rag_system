"""
vectorize_pipeline.py — 文档向量化流水线
编排 解析 → 分块 → 向量化 → 入库(Qdrant) 的完整流程。

使用方式:
    python vectorize_pipeline.py <file_or_dir> [<file_or_dir> ...]

    或作为模块使用:
    from qdrant_line.vectorize_pipeline import VectorizePipeline
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

from core.document_parser import DocumentParser, ParsedDocument
from core.text_chunker import TextChunker, Chunk
from core.embedder import TextEmbedder, resolve_model_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# 路径与配置指纹
# ═══════════════════════════════════════════════════════════════

# 解析逻辑变更时手动 +1，使 pipeline_fingerprint 变化、触发全量重建
PARSER_VERSION = "2"   # v2: 解析层加 Unicode 归一化
CHUNKER_VERSION = "2"  # v2: 标题路径 + 表格按行拆分


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

    embedding_model 先经 resolve_model_path 归一：模型名和本地缓存绝对路径
    指向同一个模型，必须产生同一个指纹，否则换种写法会被误判为"模型换了"
    而全量重建。
    """
    parts = [
        "parser", PARSER_VERSION,
        "chunker", CHUNKER_VERSION,
        "chunk", str(config.chunk_strategy), str(config.chunk_size),
        str(config.chunk_overlap), str(config.min_chunk_size),
        str(config.chunk_context_header), str(config.qa_augment),
        str(config.extract_assets), str(config.table_chunk_max),
        str(getattr(config, "use_mineru", False)),
        "embed", resolve_model_path(str(config.embedding_model)),
        # 兼容旧指纹：已移除的 BGE/Ollama 开关历史上恒为 False
        "False", "False",
        str(config.embedding_use_qwen3_vl),
    ]
    return hashlib.md5("|".join(parts).encode()).hexdigest()[:16]


# ═══════════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════════

@dataclass
class PipelineConfig:
    """流水线可调参数"""
    # ── 分块 ──
    # 768/100：制度类文档的 bullet 列表较长，512 会把"问题情境"和"答案数字"
    # 切进两个块（评测中价格系数类用例失败即此原因）；放大窗口让列表整块保留
    chunk_strategy: str = "recursive"   # fixed / recursive / sentence / markdown
    chunk_size: int = 768
    chunk_overlap: int = 100
    min_chunk_size: int = 200
    # 每个 chunk 文本前置【文件名｜章节】上下文头：
    # 口语化查询与正文字面重合少，上下文头补上文档级关键词，
    # 同时让单条结果可溯源、脱离上下文也可读
    chunk_context_header: bool = True
    # QA 对增强：入库时为每个 chunk 生成若干口语化假设问题，与原文拼接后
    # 一起向量化（payload 里仍存原文）。口语查询直接命中假设问题。
    # 成本：每 chunk 一次小 LLM 调用，仅入库时发生
    qa_augment: bool = False
    qa_questions_per_chunk: int = 3

    # ── 资产提取（图片/表格）──
    # 把文档里的图片、表格另存为 <文档目录>/kb-assets/<文档名>/ 下的文件，
    # 生成描述文字入库并关联文件路径；检索命中描述时返回资产路径
    extract_assets: bool = True
    # 表格单块上限：实测 1500 拆行会损失行间关联，6000 整表效果更好；
    # 仅超过 6000 字才按行拆（原实现是直接截断）
    table_chunk_max: int = 6000

    # PDF 默认走 MinerU（逐页文本层/OCR 自动切换，输出 Markdown）；
    # MinerU 不可用时自动回退到 pymupdf 原生解析
    use_mineru: bool = True
    mineru_url: str = "http://localhost:8000/file_parse"

    # ── 嵌入 ──
    embedding_model: str = "Qwen/Qwen3-VL-Embedding-2B"
    embedding_batch_size: int = 32
    embedding_normalize: bool = True
    embedding_device: Optional[str] = None  # None = auto
    embedding_use_qwen3_vl: bool = True     # 使用 Qwen3-VL 多模态嵌入（唯一后端）

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
        self._asset_root: Optional[str] = None

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

        # 资产统一放「入库根目录」的 kb-assets/ 下，而不是每个子目录各自一份：
        # 根目录取输入里的目录（或文件的公共父目录）
        dirs = [i for i in inputs if os.path.isdir(i)]
        if dirs:
            self._asset_root = os.path.join(dirs[0], self.ASSET_DIR_NAME)
        else:
            self._asset_root = os.path.join(
                os.path.commonpath([os.path.dirname(f) for f in files]),
                self.ASSET_DIR_NAME)

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
                # 跳过自动生成的资产目录：里面的 .md 是表格资产，
                # 不是待入库文档（否则二次入库会把它们当文档重复入库）
                from qdrant_line.sync_index import SKIP_DIR_NAMES
                glob_method = p.rglob if self.config.recursive else p.glob
                for f in glob_method("*"):
                    if not f.is_file():
                        continue
                    if any(part in SKIP_DIR_NAMES for part in f.parts):
                        continue
                    if f.suffix.lower() in allowed:
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
                chunks = self._ensure_chunker().split_from_sections(doc.sections, doc_meta)
            elif doc.metadata.get("parser") == "mineru":
                chunks = self._ensure_chunker("markdown").split(doc.content, doc_meta)
            else:
                chunks = self._ensure_chunker().split(doc.content, doc_meta)

            # 统一重编号：合并碎块后 index 可能有空洞，资产块从 len(chunks)
            # 起编号会与旧 index 碰撞（点 ID 由 文件+index 派生，会互相覆盖）
            for idx, c in enumerate(chunks):
                c.chunk_index = idx

            for c in chunks:
                # 上下文头：让 chunk 脱离原文也能被独立理解和命中。
                # 注意此后 c.text 不再是原文的纯切片，char_start/char_end
                # 仍指向原文位置（不含头部），仅用于溯源展示
                if self.config.chunk_context_header:
                    sec_label = (c.metadata.get("section_label")
                                 or c.metadata.get("heading_path")
                                 or c.metadata.get("heading")
                                 or "正文")
                    header = f"【{doc_meta['file_name']}｜{sec_label}】\n"
                    c.text = header + c.text
                # 确保 chunk 有唯一 ID
                c.metadata.setdefault("doc_id", self._hash_path(doc.file_path))
                c.metadata.setdefault("chunk_id", str(uuid.uuid4()))
                all_chunks.append(c)
                chunk_doc_map[len(all_chunks) - 1] = di

            # 资产（图片/表格）：落盘 + 描述文字 chunk
            asset_chunks = self._build_asset_chunks(
                doc, doc_meta, start_index=len(chunks))
            for c in asset_chunks:
                c.metadata.setdefault("doc_id", self._hash_path(doc.file_path))
                c.metadata.setdefault("chunk_id", str(uuid.uuid4()))
                all_chunks.append(c)
                chunk_doc_map[len(all_chunks) - 1] = di

            logger.debug("  Chunked [%s] → %d chunks", Path(doc.file_path).name, len(chunks))

        return all_chunks, chunk_doc_map

    # ── 嵌入 ─────────────────────────────────────────────

    # 资产落盘的目录名（在文档所在目录下创建）
    ASSET_DIR_NAME = "kb-assets"

    @staticmethod
    def _split_table_md(md: str, max_chars: int) -> List[str]:
        """长表格按行拆块，每块重复表头（前两行：表头 + 分隔线）。"""
        lines = md.strip().splitlines()
        if len(md) <= max_chars or len(lines) < 3:
            return [md]
        header, rows = lines[:2], lines[2:]
        parts, cur = [], list(header)
        cur_len = sum(len(x) + 1 for x in cur)
        for row in rows:
            if cur_len + len(row) + 1 > max_chars and len(cur) > 2:
                parts.append("\n".join(cur))
                cur, cur_len = list(header), sum(len(x) + 1 for x in header)
            cur.append(row)
            cur_len += len(row) + 1
        if len(cur) > 2:
            parts.append("\n".join(cur))
        return parts or [md]

    def _build_asset_chunks(
        self,
        doc: ParsedDocument,
        doc_meta: Dict[str, Any],
        start_index: int,
    ) -> List[Chunk]:
        """图片/表格资产：落盘保存 + LLM 生成描述 → 描述 chunk 入库。

        chunk.text = 描述文字（可检索）；chunk.metadata.asset_path 关联文件。
        """
        if not self.config.extract_assets or not doc.assets:
            return []

        from augment.asset_describer import describe_table, describe_image

        stem = Path(doc.file_path).stem
        # 文件名里的空格/斜杠会导致路径问题，做个简单清理
        safe_stem = "".join(c if c.isalnum() or c in "-_（）()" else "_"
                            for c in stem)[:60]
        out_dir = os.path.join(
            self._asset_root or os.path.join(os.path.dirname(doc.file_path),
                                             self.ASSET_DIR_NAME),
            safe_stem)
        os.makedirs(out_dir, exist_ok=True)

        title = doc_meta.get("file_name", stem)
        chunks = []
        for i, asset in enumerate(doc.assets):
            ext = asset.get("ext", "png")
            fname = f"{asset['kind']}_{i + 1:02d}.{ext}"
            fpath = os.path.join(out_dir, fname)
            try:
                if asset["kind"] == "image":
                    with open(fpath, "wb") as f:
                        f.write(asset["data"])
                else:  # table：存 markdown 文本（errors 兜底防漏网代理字符）
                    with open(fpath, "w", encoding="utf-8",
                              errors="replace") as f:
                        f.write(str(asset["data"]))
            except OSError as e:
                logger.warning("资产落盘失败 %s: %s", fpath, e)
                continue

            if asset["kind"] == "image":
                desc = describe_image(title, asset.get("label", ""),
                                      asset.get("context", ""))
            else:
                desc = describe_table(title, asset.get("label", ""),
                                      asset.get("context", ""),
                                      str(asset["data"]))

            label = asset.get("label", "")
            if asset["kind"] == "table":
                parts = self._split_table_md(str(asset["data"]),
                                             self.config.table_chunk_max)
                texts = []
                for pi, part in enumerate(parts):
                    lbl = label if len(parts) == 1 else f"{label}（{pi+1}/{len(parts)}）"
                    texts.append((lbl, f"【{title}｜{lbl}】\n[表格] {desc}\n{part}"))
            else:
                texts = [(label, f"【{title}｜{label}】\n[图片] {desc}")]

            for lbl, text in texts:
                chunks.append(Chunk(
                    text=text,
                    metadata={**doc_meta,
                              "chunk_method": "asset_desc",
                              "asset_type": asset["kind"],
                              "asset_path": fpath,
                              "section_label": lbl},
                    chunk_index=start_index + len(chunks),
                ))
        if chunks:
            logger.info("  资产: %s → %d 个（%s）", title, len(chunks), out_dir)
        return chunks

    def _embed_chunks(self, chunks: List[Chunk]):
        """返回 (dense_vectors, sparse_vectors) 元组。

        qa_augment 开启时：为每个 chunk 生成口语化假设问题，拼在原文后
        一起向量化；问题同时记入 chunk.metadata["qa_questions"] 备查。
        payload 里的 text 仍是原文，检索展示不受影响。
        """
        embedder = self._ensure_embedder()

        texts = None
        if self.config.qa_augment:
            from augment.qa_augment import generate_questions
            from concurrent.futures import ThreadPoolExecutor

            def _gen(c):
                qs = generate_questions(
                    c.text, n=self.config.qa_questions_per_chunk)
                c.metadata["qa_questions"] = qs
                if qs:
                    return c.text + "\n\n相关问法：" + "；".join(qs)
                return c.text

            # 每条一次 LLM 调用，串行太慢（3500 chunks ≈ 5h），8 路并发
            with ThreadPoolExecutor(max_workers=8) as pool:
                texts = list(pool.map(_gen, chunks))
            logger.info("QA 增强完成：%d chunks", len(texts))

        return embedder.embed_chunks_hybrid(
            chunks, batch_size=self.config.embedding_batch_size, texts=texts)

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

    def list_indexed_files(self) -> Dict[str, Dict[str, Any]]:
        """列出库中已入库的文件清单：{规范路径: {file_hash, chunk_count}}。

        从主 collection 的 payload 聚合（按 source_file 去重计数）。
        供 kb_add/kb_update/kb_delete 做名称/哈希定位与"是否已入库"判断。
        """
        self._ensure_qdrant()
        # collection 还不存在（首次入库前的查询）视为空库
        if not self._qdrant.collection_exists(self.config.qdrant_collection):
            return {}
        by_file: Dict[str, Dict[str, Any]] = {}
        offset = None
        while True:
            points, offset = self._qdrant.scroll(
                collection_name=self.config.qdrant_collection,
                limit=2000, offset=offset,
                with_payload=["source_file", "file_hash"],
                with_vectors=False,
            )
            for p in points:
                payload = p.payload or {}
                path = payload.get("source_file")
                if not path:
                    continue
                rec = by_file.setdefault(path, {
                    "file_hash": payload.get("file_hash", ""),
                    "chunk_count": 0,
                })
                rec["chunk_count"] += 1
            if offset is None:
                break
        return by_file

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
        典型翻车场景：入库与检索使用了不同的嵌入模型/维度。
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
            self._parser = DocumentParser(
                use_mineru=self.config.use_mineru,
                mineru_url=self.config.mineru_url,
            )
        return self._parser

    def _ensure_chunker(self, strategy: Optional[str] = None) -> TextChunker:
        if strategy and strategy != self.config.chunk_strategy:
            return TextChunker(
                strategy=strategy,
                chunk_size=self.config.chunk_size,
                overlap=self.config.chunk_overlap,
                min_chunk_size=self.config.min_chunk_size,
            )
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
                use_qwen3_vl=self.config.embedding_use_qwen3_vl,
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
    print("  from qdrant_line.vectorize_pipeline import VectorizePipeline, PipelineConfig")
    print("  pipeline = VectorizePipeline(PipelineConfig(...))")
    print("  pipeline.run(['/path/to/docs/'])")
