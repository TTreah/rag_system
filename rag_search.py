"""
rag_search.py — Qdrant RAG 检索工具
供 agent 调用的轻量检索接口：连接 Qdrant → BGE-M3 向量化查询 → 返回 top-k 结果。

使用方式:
    from rag_search import RAGSearcher

    searcher = RAGSearcher()
    results = searcher.search("Lucia Bramati World Cup", top_k=5)
    # → [{"score": 0.58, "text": "...", "source_file": "...", ...}, ...]

    # 或命令行:
    python rag_search.py                             # 交互式输入查询
    python rag_search.py "Lucia Bramati World Cup"   # 直接传 query

查询例子：Lucia Bramati 在 2025 年世界杯短道赛里最好的名次是第几？
"""

import os
import sys
from typing import List, Dict, Any, Optional
from qdrant_client.models import Prefetch, FusionQuery, Fusion, SparseVector

# ============================================================
#  连接配置（按需修改）
# ============================================================
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "news_collection"
MODEL_NAME = "BAAI/bge-m3"
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages:"
TOP_K = 10

# ============================================================


class RAGSearcher:
    """
    RAG 检索器（惰性加载模型 + Qdrant 连接）。

    每次 search() 调用:
        1. BGE-M3 encode_queries() 向量化查询
        2. Qdrant query_points() 检索
        3. 返回带 score / text / source_file 的结构化结果
    """

    def __init__(
        self,
        qdrant_url: str = QDRANT_URL,
        collection: str = COLLECTION_NAME,
        model_name: str = MODEL_NAME,

    ):
        self._qdrant_url = qdrant_url
        self._collection = collection
        self._model_name = model_name
        self._model = None
        self._client = None

    # ── 检索（唯一公开方法）────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = TOP_K,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Args:
            query: 自然语言查询
            top_k: 返回结果数
            score_threshold: 最低相似度阈值（None=不过滤）

        Returns:
            [{"id": str, "score": float, "text": str,
              "source_file": str, "section_label": str,
              "metadata": dict}, ...]
        """
        model = self._get_model()
        client = self._get_client()

        # 1. 混合查询向量化（dense + sparse）
        result = model.encode_queries(query, return_dense=True, return_sparse=True)
        if isinstance(result, dict):
            dense_vec = result["dense_vecs"].tolist()
            lw = result.get("lexical_weights")
        else:
            dense_vec = result.tolist()
            lw = None

        # 2. 混合检索（dense + sparse → RRF fusion）
        prefetch = [
            Prefetch(query=dense_vec, using="dense", limit=top_k * 2),
        ]
        if lw is not None and len(lw) > 0:
            sparse_vec = SparseVector(
                indices=[int(k) for k in lw.keys()],
                values=[float(v) for v in lw.values()],
            )
            prefetch.append(
                Prefetch(query=sparse_vec, using="sparse", limit=top_k * 2),
            )

        response = client.query_points(
            collection_name=self._collection,
            prefetch=prefetch,
            query=FusionQuery(fusion=Fusion.RRF),
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=True,
        )

        # 3. 整理结果
        results = []
        for point in (response.points or []):
            payload = point.payload or {}
            results.append({
                "id": point.id,
                "score": point.score,
                "text": payload.get("text", ""),
                "source_file": payload.get("source_file", ""),
                "file_name": os.path.basename(payload.get("source_file", "")),
                "file_type": payload.get("file_type", ""),
                "section_label": payload.get("section_label", ""),
                "chunk_index": payload.get("chunk_index", -1),
                "file_mtime": payload.get("file_mtime", ""),
                "ingested_at": payload.get("ingested_at", ""),
                "metadata": payload.get("metadata", {}),
            })

        return results

    @property
    def collection_info(self) -> dict:
        """返回 collection 基本信息（点数、维度等）"""
        client = self._get_client()
        if not client.collection_exists(self._collection):
            return {"exists": False}
        info = client.get_collection(self._collection)
        config = getattr(info, "config", None)
        params = getattr(config, "params", None) if config else None
        vectors = getattr(params, "vectors", None) if params else None

        # 兼容单向量和命名向量两种配置
        result = {"exists": True, "points": getattr(info, "points_count", 0)}
        if vectors:
            if isinstance(vectors, dict):
                for name, vc in vectors.items():
                    result[name] = {
                        "dim": getattr(vc, "size", None),
                        "distance": str(getattr(vc, "distance", "")),
                    }
            else:
                result["dense"] = {
                    "dim": getattr(vectors, "size", None),
                    "distance": str(getattr(vectors, "distance", "")),
                }
        return result

    # ── 内部懒加载 ─────────────────────────────────────────

    def _get_model(self):
        if self._model is None:
            # 强制离线，从缓存加载
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            from FlagEmbedding import FlagAutoModel
            self._model = FlagAutoModel.from_finetuned(
                self._model_name,
                query_instruction_for_retrieval=QUERY_INSTRUCTION,
                use_fp16=True,
            )
        return self._model

    def _get_client(self):
        if self._client is None:
            from qdrant_client import QdrantClient
            self._client = QdrantClient(url=self._qdrant_url)
        return self._client


# ============================================================
#  交互式检索入口
# ============================================================
if __name__ == "__main__":
    searcher = RAGSearcher()

    def _safe(s: str) -> str:
        """GBK 控制台安全输出（意大利语文件名含 NBSP 等特殊字符）"""
        return s.encode("gbk", errors="replace").decode("gbk")

    def _print_results(results):
        if not results:
            print("  (无结果)")
            return
        for i, r in enumerate(results):
            print(f"\n  [{i + 1}] score={r['score']:.4f}")
            print(f"      file: {_safe(r['file_name'])}")
            print(f"      text: {_safe(r['text'][:150])}")

    # 命令行传参 → 单次检索；否则进入交互式循环
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        _print_results(searcher.search(query, top_k=5))
    else:
        print("知识库检索 (输入空行退出)")
        print(f"Collection: {COLLECTION_NAME}")
        while True:
            query = input("\n输入查询: ").strip()
            if not query:
                break
            _print_results(searcher.search(query, top_k=5))
