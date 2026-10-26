"""
rag_search.py — Qdrant RAG 检索工具
供 agent 调用的轻量检索接口：连接 Qdrant → BGE-M3 向量化查询 → 返回 top-k 结果。

使用方式:
    from qdrant_line.rag_search import RAGSearcher

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

# 允许直接以脚本方式运行（python qdrant_line/rag_search.py "..."），
# 把包根（core/ augment/ 等所在目录）加进 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Dict, Any, Optional
from qdrant_client.models import Prefetch, FusionQuery, Fusion, SparseVector

from core.embedder import TextEmbedder
from core.reranker import Reranker, RERANK_THRESHOLD

# ============================================================
#  连接配置（按需修改）
# ============================================================
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "news_collection"
# 模型名即可：TextEmbedder 会自动解析到 ModelScope / HuggingFace 本地缓存
MODEL_NAME = "Qwen/Qwen3-VL-Embedding-2B"
TOP_K = 5

# ============================================================


class RAGSearcher:
    """
    RAG 检索器（惰性加载模型 + Qdrant 连接）。

    查询向量化复用 TextEmbedder（embed_query 走查询侧专用编码路径，
    带检索指令），不在此处重复实现，避免与入库侧漂移。

    每次 search() 调用:
        1. TextEmbedder.embed_query() 向量化查询（dense + sparse）
        2. Qdrant query_points() 混合检索（RRF 融合）
        3. 返回带 score / text / source_file 的结构化结果
    """

    def __init__(
        self,
        qdrant_url: str = QDRANT_URL,
        collection: str = COLLECTION_NAME,
        model_name: str = MODEL_NAME,
        use_rerank: bool = True,
        rerank_threshold: float = RERANK_THRESHOLD,
        max_images: int = 2,
    ):
        self._qdrant_url = qdrant_url
        self._collection = collection
        self._model_name = model_name
        self._use_rerank = use_rerank
        self._rerank_threshold = rerank_threshold
        self._max_images = max_images
        self._embedder = None
        self._reranker = None
        self._client = None

    # ── 检索（唯一公开方法）────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = TOP_K,
        score_threshold: Optional[float] = None,
        use_rewrite: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Args:
            query: 自然语言查询
            top_k: 返回结果数
            score_threshold: 最低相似度阈值（None=不过滤）
            use_rewrite: 口语→书面语改写（默认开）。是"原查询 + 改写查询"
                双路检索 + RRF 融合，而不是只用改写结果——改写偶尔会丢关键词
                （实测改写曾把"导购"丢掉导致原本能命中的题失手），双路
                融合能同时吃到两边的命中

        Returns:
            [{"id": str, "score": float, "text": str,
              "source_file": str, "section_label": str,
              "metadata": dict}, ...]
        """
        queries = [query]
        if use_rewrite:
            from augment.query_rewriter import rewrite_query
            rewritten = rewrite_query(query)
            if rewritten != query:
                queries.append(rewritten)

        # 逐路检索（多路时每路取 top_k*2，给二次融合留足候选，
        # 否则一路第 5 名的正确点会被交错挤出最终 top-k），
        # 两路结果再做一次 RRF 融合（按排名而非分数合并）。
        # 单路时保持 Qdrant 原始分数，不做二次融合。
        per_query_k = top_k * 2 if len(queries) > 1 else top_k
        if self._use_rerank:
            per_query_k = max(per_query_k, 12)  # 给重排留足候选池
        ranked_lists = [
            self._search_once(q, top_k=per_query_k,
                              score_threshold=score_threshold)
            for q in queries
        ]
        if len(ranked_lists) == 1:
            candidates = ranked_lists[0]
        else:
            fused: Dict[Any, Dict[str, Any]] = {}
            for lst in ranked_lists:
                for rank, r in enumerate(lst, 1):
                    rrf = 1.0 / (60 + rank)
                    if r["id"] not in fused:
                        fused[r["id"]] = {**r, "score": rrf}
                    else:
                        fused[r["id"]]["score"] += rrf
            candidates = sorted(fused.values(), key=lambda r: r["score"],
                                reverse=True)

        # 重排 + 相关性闸门：reranker 的分是真实相关度（0~1），
        # 低于阈值的结果全部丢弃——无关查询在这里被整单拒答。
        # 双路查询时每个候选分别用原查询和改写查询打分取 max：
        # 改写后的书面查询更贴近文档措辞，口语化弱命中不会被误杀。
        # 重排输入包含 chunk 的假设问题（qa_augment 产物）：
        # 问题由 chunk 生成，命中问题即命中答案；无关查询的得分
        # 不会被抬高（假设问题全部在自行车域内）
        if self._use_rerank and candidates:
            reranker = self._get_reranker()
            texts = []
            for r in candidates:
                t = r["text"]
                qa = (r.get("metadata") or {}).get("qa_questions")
                if qa:
                    t = t + "\n" + "；".join(qa)
                texts.append(t)
            best = [0.0] * len(candidates)
            for q in queries:
                for i, s in enumerate(reranker.score(q, texts)):
                    best[i] = max(best[i], s)
            for r, s in zip(candidates, best):
                r["rerank_score"] = s
            candidates = [r for r in candidates
                          if r["rerank_score"] >= self._rerank_threshold]
            candidates.sort(key=lambda r: r["rerank_score"], reverse=True)

        # 去重：图片点与描述点共享 (source_file, chunk_index)，保留分数高的
        seen = set()
        deduped = []
        for r in candidates:
            key = (r.get("source_file"), r.get("chunk_index"))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(r)

        # 多样性约束：图片最多占结果的 max_images 个，其余位置留给文本块。
        # 否则同一主题的 PPT/宣传图会挤满 top-k，用户拿到一堆图却没有正文。
        picked, n_img = [], 0
        for r in deduped:
            is_img = (r.get("metadata") or {}).get("asset_type") == "image"
            if is_img and n_img >= self._max_images:
                continue
            picked.append(r)
            n_img += is_img
            if len(picked) >= top_k:
                break
        return picked

    def _search_once(
        self,
        query: str,
        top_k: int,
        score_threshold: Optional[float],
    ) -> List[Dict[str, Any]]:
        """单条查询的完整检索：向量化 → Qdrant 混合检索 → 结构化结果"""
        embedder = self._get_embedder()
        client = self._get_client()

        # 1. 混合查询向量化（dense + sparse，查询侧编码路径）
        dense, sparse_map = embedder.embed_query(query)

        # 2. 混合检索（dense + sparse → RRF fusion）
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
                # 资产关联：命中图片/表格描述时给出文件路径，供调用方取用素材
                "asset_type": (payload.get("metadata") or {}).get("asset_type"),
                "asset_path": (payload.get("metadata") or {}).get("asset_path"),
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

    def _get_embedder(self) -> TextEmbedder:
        if self._embedder is None:
            self._embedder = TextEmbedder(
                model_name=self._model_name,
                use_qwen3_vl=True,
            )
        return self._embedder

    def _get_reranker(self) -> Reranker:
        if self._reranker is None:
            self._reranker = Reranker()
        return self._reranker

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
            if r.get("asset_path"):
                print(f"      资产: {_safe(r['asset_path'])}")
            print(f"      text: {_safe(r['text'][:150])}")

    # 命令行传参 → 单次检索；否则进入交互式循环
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        _print_results(searcher.search(query, top_k=TOP_K))
    else:
        print("知识库检索 (输入空行退出)")
        print(f"Collection: {COLLECTION_NAME}")
        while True:
            query = input("\n输入查询: ").strip()
            if not query:
                break
            _print_results(searcher.search(query, top_k=TOP_K))
