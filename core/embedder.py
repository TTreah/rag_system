"""
embedder.py — 文本向量化嵌入模块
封装 sentence-transformers 模型，提供单条/批量嵌入。
当 HuggingFace 不可用时（国内常见情况），自动降级为本地轻量嵌入。

支持的模型来源:
    - HuggingFace 本地模型 (sentence-transformers)
    - ModelScope 镜像 (国内可用)
    - 内置轻量降级方案 (无需下载，开箱即用)

使用方式:
    embedder = TextEmbedder(model_name="bge-small")
    vectors = embedder.embed(["文本1", "文本2", ...])
    # vectors.shape → (2, 384)
"""

import os
import sys
import time
import hashlib
import logging
from typing import Any, List, Union, Optional
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def resolve_model_path(model_name: str) -> str:
    """把模型名解析为本地缓存路径。

    解析顺序：已是目录 → ModelScope 缓存 → HuggingFace 缓存 → 原样返回
    （交给下游在线加载或报错）。入口脚本因此只需写模型名（"BAAI/bge-m3"），
    不用关心模型实际是从哪个镜像站下的、缓存目录叫什么。

    ModelScope 缓存结构: ~/.cache/modelscope/models/BAAI--bge-m3/snapshots/<版本>/
    HuggingFace 缓存结构: ~/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/<hash>/
    """
    if os.path.isdir(model_name):
        return model_name

    flat = model_name.replace("/", "--")
    caches = [
        Path.home() / ".cache" / "modelscope" / "models" / flat,
        Path.home() / ".cache" / "huggingface" / "hub" / ("models--" + flat),
    ]
    for cache in caches:
        snapshots = cache / "snapshots"
        if not snapshots.exists():
            continue
        versions = sorted(d for d in snapshots.iterdir() if d.is_dir())
        for v in reversed(versions):          # 取最新版本
            if (v / "config.json").exists():  # 确认是完整模型而非半截下载
                return str(v)
    return model_name


# ═══════════════════════════════════════════════════════════════
# 轻量级后备嵌入器（无需外部模型，开箱即用）
# ═══════════════════════════════════════════════════════════════

class FallbackEmbedder:
    """
    基于字符 n-gram 哈希的轻量嵌入器。
    无需下载任何模型，适合网络受限的环境测试使用。

    原理:
        提取 1~3 gram 字符特征 → 哈希 → 稀疏向量 → 归一化
    维度: 256 (可配)
    """

    def __init__(self, dim: int = 256):
        self.dim = dim
        self._ngram_range = (1, 3)

    def encode(
        self,
        texts: List[str],
        batch_size: int = 64,
        show_progress_bar: bool = False,
        normalize_embeddings: bool = True,
    ) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)

        for i, text in enumerate(texts):
            vec = self._text_to_vec(text)
            vectors[i] = vec

        if normalize_embeddings:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            vectors = vectors / norms

        return vectors

    def _text_to_vec(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        if not text:
            return vec

        # 字符 n-gram 哈希
        chars = list(text)
        for n in range(self._ngram_range[0], self._ngram_range[1] + 1):
            for start in range(len(chars) - n + 1):
                gram = "".join(chars[start:start + n])
                h = int(hashlib.md5(gram.encode("utf-8")).hexdigest(), 16)
                idx = h % self.dim
                vec[idx] += 1.0

        # 也加入一些词级特征（简单空格分词）
        words = text.replace("\n", " ").split()
        for w in words:
            h = int(hashlib.md5(w.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dim
            vec[idx] += 0.5

        return vec

    def get_sentence_embedding_dimension(self) -> int:
        return self.dim

    @property
    def device(self):
        return "cpu"


# ═══════════════════════════════════════════════════════════════
# Ollama 后端（本机已有 Ollama 跑 bge-m3 时，免下载 2GB 模型）
# ═══════════════════════════════════════════════════════════════

class OllamaEmbedder:
    """
    通过 Ollama 的 /api/embed 做向量化。

    与 FlagEmbedding 后端的差别：只有 dense 向量，没有 sparse
    （Ollama 不暴露 lexical weights）。入库的 collection 将只建 dense 索引，
    检索退化为纯向量相似度，不做 RRF 混合。
    """

    def __init__(self, model: str = "bge-m3",
                 base_url: str = "http://localhost:11434", timeout: int = 120):
        import urllib.request
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._urlopen = urllib.request.urlopen
        self._dim = None

    def _post(self, payload: dict) -> dict:
        import json as _json
        import urllib.request
        req = urllib.request.Request(
            f"{self.base_url}/api/embed",
            data=_json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with self._urlopen(req, timeout=self.timeout) as resp:
            return _json.loads(resp.read().decode("utf-8"))

    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        normalize_embeddings: bool = True,
    ) -> np.ndarray:
        vectors = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            resp = self._post({"model": self.model, "input": batch})
            vectors.extend(resp["embeddings"])
        arr = np.asarray(vectors, dtype=np.float32)

        if normalize_embeddings:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            arr = arr / norms
        return arr

    def get_sentence_embedding_dimension(self) -> int:
        if self._dim is None:
            resp = self._post({"model": self.model, "input": ["dim probe"]})
            self._dim = len(resp["embeddings"][0])
        return self._dim

    @property
    def device(self):
        return "ollama"


# ═══════════════════════════════════════════════════════════════
# Qwen3-VL 后端（多模态嵌入：文本 + 图片）
# ═══════════════════════════════════════════════════════════════

class Qwen3VLEmbedderAdapter:
    """
    Qwen3-VL-Embedding 的 sentence-transformers 式接口适配。

    为什么不用 Ollama 跑它：Ollama/llama.cpp 尚不支持 VL 架构的嵌入
    （实测 /api/embed 直接报错），这里用 ModelScope 原版权重 + transformers，
    模型仓库自带的官方脚本（scripts/qwen3_vl_embedding.py）做加载。

    与 BGE-M3(FlagEmbedding) 的差别：只有 dense，没有 sparse 输出；
    支持图片输入（本适配暂不暴露，留出接口）。
    """

    def __init__(self, model_path: str,
                 query_instruction: str = "Given a search query, retrieve "
                                          "relevant passages that answer the query.",
                 prefer_mps: bool = True):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "qwen3_vl_embedding",
            os.path.join(model_path, "scripts", "qwen3_vl_embedding.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self._emb = mod.Qwen3VLEmbedder(model_name_or_path=model_path)
        # 官方脚本只认 cuda/cpu；Mac 上挪到 MPS 提速。
        # 并发服务场景必须关掉：MPS 非线程安全，多请求并发会触发
        # Metal 断言直接崩进程（实测：8 路并发嵌入时崩溃）
        if prefer_mps:
            try:
                import torch
                if torch.backends.mps.is_available():
                    self._emb.model.to("mps")
            except Exception:
                pass
        self.query_instruction = query_instruction

    def encode(
        self,
        texts: List[str],
        batch_size: int = 8,
        show_progress_bar: bool = False,
        normalize_embeddings: bool = True,
        is_query: bool = False,
    ) -> np.ndarray:
        vectors = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            inputs = []
            for t in batch:
                item = {"text": t}
                # 查询侧带检索指令，文档侧不带（非对称检索）
                if is_query:
                    item["instruction"] = self.query_instruction
                inputs.append(item)
            emb = self._emb.process(inputs, normalize=normalize_embeddings)
            # bf16 → float32 再转 numpy（numpy 不认 bf16）
            vectors.append(emb.float().cpu().numpy())
        return np.vstack(vectors) if vectors else np.zeros((0, 2048))

    def get_sentence_embedding_dimension(self) -> int:
        # Qwen3VLConfig 是嵌套结构，没有顶层 hidden_size，实测一条最准
        vec = self._emb.process([{"text": "dim probe"}], normalize=False)
        return int(vec.shape[-1])

    @property
    def device(self):
        return str(self._emb.model.device)


# ═══════════════════════════════════════════════════════════════
# 主嵌入器
# ═══════════════════════════════════════════════════════════════

class TextEmbedder:
    """
    文本向量化封装。

    特性:
        - 惰性加载模型（首次调用 embed 时加载）
        - 批量处理 + 进度显示
        - 自动 GPU/CPU 选择
        - HuggingFace 不可用时自动降级为本地轻量嵌入
        - BGE-M3 通过 FlagEmbedding 后端加载（返回 dict 自动提取 dense_vecs）
    """

    # 常用中文模型
    PRESET_MODELS = {
        "bge-small":     "BAAI/bge-small-zh-v1.5",      # 384维，轻量快速
        "bge-base":      "BAAI/bge-base-zh-v1.5",        # 768维
        "bge-large":     "BAAI/bge-large-zh-v1.5",       # 1024维，最强中文
        "bge-small-en":  "BAAI/bge-small-en-v1.5",       # 英文
        "bge-m3":        "BAAI/bge-m3",                  # 1024维，多语言+稀疏+ColBERT
        "m3e-base":      "moka-ai/m3e-base",             # 768维，中文
        "text2vec":      "shibing624/text2vec-base-chinese", # 768维
        "multilingual":  "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        # 国内 ModelScope 镜像
        "bge-small-ms":  "BAAI/bge-small-zh-v1.5",       # 配合 modelscope_dir 使用
    }

    def __init__(
        self,
        model_name: str = "bge-small",
        device: Optional[str] = None,
        batch_size: int = 32,
        normalize: bool = True,
        show_progress: bool = True,
        cache_dir: Optional[str] = None,
        local_files_only: bool = False,
        fallback_dim: int = 256,
        use_fallback: bool = False,       # 强制使用轻量嵌入（跳过 HF 下载）
        use_flag_embedding: bool = False, # 使用 FlagEmbedding 后端（BGE-M3 等）
        use_ollama: bool = False,         # 使用本机 Ollama 后端（免下载模型）
        use_qwen3_vl: bool = False,       # 使用 Qwen3-VL 多模态嵌入（本地权重）
        ollama_url: str = "http://localhost:11434",
        modelscope_dir: Optional[str] = None,  # ModelScope 本地模型目录
    ):
        """
        Args:
            model_name: 模型名或 PRESET_MODELS 中的简称
            device: 设备 (None=auto)
            batch_size: 批量嵌入大小
            normalize: 是否归一化向量（L2，推荐）
            show_progress: 显示进度条
            cache_dir: HuggingFace 缓存目录
            local_files_only: 仅使用本地文件（不联网）
            fallback_dim: 降级嵌入器维度 (256)
            use_fallback: 强制使用轻量降级嵌入
            use_flag_embedding: 使用 FlagEmbedding 后端（BGE-M3 等返回 dict 的模型）
            modelscope_dir: ModelScope 下载的模型本地路径
        """
        self.model_name = self.PRESET_MODELS.get(model_name, model_name)
        self._device = device
        self.batch_size = batch_size
        self.normalize = normalize
        self.show_progress = show_progress
        self.cache_dir = cache_dir
        self.local_files_only = local_files_only
        self.fallback_dim = fallback_dim
        self.use_fallback = use_fallback
        self.use_flag_embedding = use_flag_embedding
        self.use_ollama = use_ollama
        self.use_qwen3_vl = use_qwen3_vl
        self.ollama_url = ollama_url
        self.modelscope_dir = modelscope_dir

        self._model = None
        self._dim = None
        self._using_fallback = False
        self._using_flag = False
        self._using_qwen3vl = False

    # ── 公共 API ──────────────────────────────────────────

    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """向量化文本。

        Returns:
            np.ndarray, shape (n, dim) 或 (dim,) for single input
        """
        single = isinstance(texts, str)
        if single:
            texts = [texts]

        self._ensure_model()
        start = time.time()

        # FlagEmbedding (BGE-M3) 只接受 batch_size, max_length 等核心参数
        # show_progress_bar / normalize_embeddings 是 sentence-transformers 的
        if self._using_flag:
            vectors = self._model.encode(texts, batch_size=self.batch_size)
        else:
            vectors = self._model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress and len(texts) > 10,
                normalize_embeddings=self.normalize,
            )

        # FlagEmbedding (BGE-M3) 返回 dict，提取稠密向量
        if isinstance(vectors, dict):
            vectors = vectors['dense_vecs']

        elapsed = time.time() - start
        tag = "[fallback]" if self._using_fallback else ("[flag]" if self._using_flag else "")
        logger.info(
            "%s Embedded %d texts in %.1fs (%.0f texts/s), shape=%s",
            tag, len(texts), elapsed, len(texts) / elapsed if elapsed > 0 else 0,
            vectors.shape,
        )

        if single:
            return vectors[0]
        return vectors

    def embed_chunks(
        self,
        chunks: List[Any],
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """对 text_chunker.Chunk 列表向量化，回写维度到 chunk.metadata。

        Returns:
            np.ndarray shape (n, dim)
        """
        texts = [c.text for c in chunks]
        bs = batch_size or self.batch_size
        self._ensure_model()

        if self._using_flag:
            vectors = self._model.encode(texts, batch_size=bs)
        else:
            vectors = self._model.encode(
                texts,
                batch_size=bs,
                show_progress_bar=self.show_progress and len(texts) > 10,
                normalize_embeddings=self.normalize,
            )

        # FlagEmbedding (BGE-M3) 返回 dict，提取稠密向量
        if isinstance(vectors, dict):
            vectors = vectors['dense_vecs']
        for chunk, vec in zip(chunks, vectors):
            chunk.metadata["embedding_dim"] = len(vec)
        return vectors

    def embed_chunks_hybrid(
        self,
        chunks: List[Any],
        batch_size: Optional[int] = None,
        texts: Optional[List[str]] = None,
    ):
        """对 chunks 向量化，返回 (dense, sparse) 元组。

        sparse 为 dict 列表，key 已转为 int，可直接用于 Qdrant SparseVector。

        Args:
            texts: 实际用于向量化的文本（默认 chunk.text）。QA 增强时传入
                "原文+假设问题"的拼接文本，payload 里存的仍是原文。

        Returns:
            (dense: np.ndarray shape (n, 1024),
             sparse: list[dict[int, float]])
        """
        texts = texts if texts is not None else [c.text for c in chunks]
        bs = batch_size or self.batch_size
        self._ensure_model()

        if not self._using_flag:
            # 非 FlagEmbedding → 仅返回空稀疏向量
            vectors = self._model.encode(
                texts,
                batch_size=bs,
                show_progress_bar=self.show_progress and len(texts) > 10,
                normalize_embeddings=self.normalize,
            )
            for chunk, vec in zip(chunks, vectors):
                chunk.metadata["embedding_dim"] = len(vec)
            return vectors, [{}] * len(chunks)

        # FlagEmbedding: 同时返回 dense + sparse
        result = self._model.encode(
            texts, batch_size=bs,
            return_dense=True, return_sparse=True,
        )
        dense = result['dense_vecs']
        sparse = result.get('lexical_weights')
        if sparse is None:
            sparse = [{}] * len(chunks)

        # 转换 key: str → int (Qdrant SparseVector 要求)
        sparse = [
            {int(k): float(v) for k, v in (s.items() if s else [])}
            for s in sparse
        ]

        for chunk, vec in zip(chunks, dense):
            chunk.metadata["embedding_dim"] = len(vec)
        return dense, sparse

    def embed_query(self, query: str):
        """向量化单条查询，返回 (dense, sparse)。

        必须与文档侧走不同的编码路径：BGE-M3 这类模型在 query 侧要附加检索指令
        （query_instruction_for_retrieval），直接用文档路径 encode() 会得到偏移的向量。

        Returns:
            (dense: np.ndarray shape (dim,), sparse: dict[int, float])
            非 FlagEmbedding 后端时 sparse 为空 dict。
        """
        self._ensure_model()

        if self._using_qwen3vl:
            # Qwen3-VL：查询侧带检索指令（非对称检索）
            vec = self._model.encode(
                [query], batch_size=1,
                normalize_embeddings=self.normalize, is_query=True,
            )
            return np.asarray(vec[0]), {}

        if not self._using_flag:
            vec = self._model.encode(
                [query],
                batch_size=1,
                show_progress_bar=False,
                normalize_embeddings=self.normalize,
            )
            return np.asarray(vec[0]), {}

        result = self._model.encode_queries(
            query, return_dense=True, return_sparse=True
        )
        if not isinstance(result, dict):
            return np.asarray(result), {}

        dense = np.asarray(result["dense_vecs"])
        if dense.ndim > 1:
            dense = dense[0]

        # 单条查询时 lexical_weights 可能是 dict，也可能是长度为 1 的 list
        lw = result.get("lexical_weights")
        if isinstance(lw, list):
            lw = lw[0] if lw else None
        sparse = {int(k): float(v) for k, v in lw.items()} if lw else {}
        return dense, sparse

    @property
    def dim(self) -> int:
        self._ensure_model()
        return self._dim

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def is_fallback(self) -> bool:
        """是否使用了降级嵌入器"""
        return self._using_fallback

    @property
    def is_flag(self) -> bool:
        """是否使用了 FlagEmbedding 后端"""
        return self._using_flag

    def unload(self):
        self._model = None
        self._dim = None

    # ── 内部 ─────────────────────────────────────────────

    def _ensure_model(self):
        if self._model is not None:
            return

        # 强制降级模式
        if self.use_fallback:
            self._init_fallback()
            return

        # Ollama 本机后端（dense-only）
        if self.use_ollama:
            self._init_ollama()
            return

        # Qwen3-VL 多模态后端（dense-only）
        if self.use_qwen3_vl:
            self._init_qwen3vl()
            return

        # FlagEmbedding 后端（BGE-M3 等返回多向量 dict 的模型）
        if self.use_flag_embedding:
            self._init_flag_model()
            return

        # 尝试加载真实模型
        print(f"[Embedder] Loading model: {self.model_name} ...", file=sys.stderr)

        # 尝试顺序: ModelScope本地 → HuggingFace本地 → HuggingFace在线 → 降级
        model_path = None

        # 1) ModelScope 本地目录
        if self.modelscope_dir:
            candidate = Path(self.modelscope_dir)
            if candidate.exists():
                model_path = str(candidate)
                print(f"[Embedder] Using ModelScope local model: {model_path}", file=sys.stderr)

        # 2) HuggingFace 缓存查找
        if not model_path:
            model_path = self._find_local_model()

        # 3) 尝试连接 HuggingFace
        if not model_path and not self.local_files_only:
            try:
                self._init_sentence_transformer(self.model_name)
                return
            except Exception as e:
                print(f"[Embedder] HF download failed: {e}", file=sys.stderr)
                print("[Embedder] Trying local fallback...", file=sys.stderr)

        # 4) 本地模型加载
        if model_path:
            try:
                self._init_sentence_transformer(model_path)
                return
            except Exception as e:
                print(f"[Embedder] Local model load failed: {e}", file=sys.stderr)

        # 5) 最终降级
        self._init_fallback()

    def _init_sentence_transformer(self, model_path: str):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(
            model_path,
            cache_folder=self.cache_dir,
            device=self._device,
            local_files_only=self.local_files_only,
        )
        self._dim = self._model.get_sentence_embedding_dimension()
        self._using_fallback = False
        print(
            f"[Embedder] Model loaded. Dim={self._dim}, "
            f"Device={self._model.device}",
            file=sys.stderr,
        )

    def _init_flag_model(self):
        """通过 FlagEmbedding 加载 BGE-M3 等多向量模型

        默认离线模式（HF_HUB_OFFLINE=1），从本地缓存加载，不联网。
        如需更新模型版本，在调用前设置环境变量:
            os.environ["HF_HUB_OFFLINE"] = "0"
            os.environ["TRANSFORMERS_OFFLINE"] = "0"
        """
        # 强制离线：禁止底层库发起网络请求，直接从缓存读取
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        from FlagEmbedding import FlagAutoModel
        model_path = resolve_model_path(self.model_name)
        if model_path != self.model_name:
            print(f"[Embedder] Resolved local model: {model_path}", file=sys.stderr)
        print(f"[Embedder] Loading FlagEmbedding model from cache: {model_path} ...",
              file=sys.stderr)
        try:
            self._model = FlagAutoModel.from_finetuned(
                model_path,
                query_instruction_for_retrieval=(
                    "Represent this sentence for searching relevant passages:"
                ),
                use_fp16=True,
            )
        except ValueError:
            # FlagAutoModel 靠路径 basename 识别模型类别，本地缓存目录名
            # （如 ModelScope 的 snapshots/master）识别不了 → 回退到
            # BGEM3FlagModel，它直接接受任意本地路径
            from FlagEmbedding import BGEM3FlagModel
            self._model = BGEM3FlagModel(
                model_path,
                query_instruction_for_retrieval=(
                    "Represent this sentence for searching relevant passages:"
                ),
                use_fp16=True,
            )
        self._dim = 1024  # BGE-M3 固定 1024 维
        self._using_fallback = False
        self._using_flag = True
        device = getattr(self._model, 'device', 'auto')
        print(
            f"[Embedder] FlagEmbedding model loaded. Dim={self._dim}, "
            f"Device={device}",
            file=sys.stderr,
        )

    def _init_qwen3vl(self):
        model_path = resolve_model_path(self.model_name)
        print(f"[Embedder] Loading Qwen3-VL embedding: {model_path} ...",
              file=sys.stderr)
        self._model = Qwen3VLEmbedderAdapter(model_path)
        self._dim = self._model.get_sentence_embedding_dimension()
        self._using_qwen3vl = True
        print(f"[Embedder] Qwen3-VL loaded. Dim={self._dim}, "
              f"Device={self._model.device}", file=sys.stderr)

    def _init_ollama(self):
        # Ollama 模型名不带组织前缀："BAAI/bge-m3" → "bge-m3"
        model = self.model_name.split("/")[-1]
        self._model = OllamaEmbedder(model=model, base_url=self.ollama_url)
        self._dim = self._model.get_sentence_embedding_dimension()
        print(f"[Embedder] Using Ollama backend: {model} "
              f"(dim={self._dim}, {self.ollama_url})", file=sys.stderr)

    def _init_fallback(self):
        print(
            f"[Embedder] Using fallback embedder (dim={self.fallback_dim}). "
            f"Set use_fallback=False to retry HF download.",
            file=sys.stderr,
        )
        self._model = FallbackEmbedder(dim=self.fallback_dim)
        self._dim = self.fallback_dim
        self._using_fallback = True

    def _find_local_model(self) -> Optional[str]:
        """在常见缓存路径中查找本地模型"""
        # HuggingFace 默认缓存
        candidates = []
        if self.cache_dir:
            candidates.append(Path(self.cache_dir) / "models--" + self.model_name.replace("/", "--"))
        candidates.append(
            Path.home() / ".cache" / "huggingface" / "hub" /
            ("models--" + self.model_name.replace("/", "--"))
        )
        # sentence-transformers 缓存
        candidates.append(
            Path.home() / ".cache" / "torch" / "sentence_transformers" /
            self.model_name.replace("/", "_")
        )

        for c in candidates:
            if c.exists():
                # 找到 snapshots 子目录中最新的
                snapshots = c / "snapshots"
                if snapshots.exists():
                    subdirs = sorted([d for d in snapshots.iterdir() if d.is_dir()])
                    if subdirs:
                        return str(subdirs[-1])
                return str(c)
        return None
