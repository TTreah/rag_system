"""
embedder.py — 文本/图片向量化（Qwen3-VL-Embedding）

当前唯一后端：Qwen3-VL-Embedding-2B（本地权重，ModelScope 缓存，MPS 加速）。
无模型时退化为字符 n-gram 哈希嵌入（仅供无网络环境冒烟测试）。

BGE / FlagEmbedding / Ollama 后端已全部移除；如需历史行为请查看 git 记录。
"""

import os
import sys
import time
import hashlib
import logging
from pathlib import Path
from typing import Any, List, Union, Optional

import numpy as np

from config import EMBEDDING_MODEL as _CFG_EMBEDDING_MODEL

logger = logging.getLogger(__name__)

DEFAULT_MODEL = _CFG_EMBEDDING_MODEL


def resolve_model_path(model_name: str) -> str:
    """把模型名解析为本地缓存路径。

    解析顺序：已是目录 → ModelScope 缓存 → HuggingFace 缓存 → 原样返回。
    ModelScope 缓存结构: ~/.cache/modelscope/models/Qwen--Qwen3-VL-Embedding-2B/snapshots/<版本>/
    HuggingFace 缓存结构: ~/.cache/huggingface/hub/models--Qwen--Qwen3-VL-Embedding-2B/snapshots/<hash>/
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
            if (v / "config.json").exists():  # 确认是完整模型
                return str(v)
    return model_name


# ═══════════════════════════════════════════════════════════════
# 无模型降级（仅冒烟测试用）
# ═══════════════════════════════════════════════════════════════

class FallbackEmbedder:
    """字符 n-gram 哈希嵌入，维度 256。无模型环境下保证流程可跑通。"""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def encode(self, texts: List[str], batch_size: int = 64,
               show_progress_bar: bool = False,
               normalize_embeddings: bool = True) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            vec = np.zeros(self.dim, dtype=np.float32)
            chars = list(text or "")
            for n in (1, 2, 3):
                for start in range(len(chars) - n + 1):
                    gram = "".join(chars[start:start + n])
                    h = int(hashlib.md5(gram.encode("utf-8")).hexdigest(), 16)
                    vec[h % self.dim] += 1.0
            vectors[i] = vec
        if normalize_embeddings:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            vectors = vectors / norms
        return vectors

    def get_sentence_embedding_dimension(self) -> int:
        return self.dim

    @property
    def device(self):
        return "cpu"


# ═══════════════════════════════════════════════════════════════
# Qwen3-VL-Embedding（当前唯一正式后端）
# ═══════════════════════════════════════════════════════════════

class Qwen3VLEmbedderAdapter:
    """加载模型仓库自带的官方脚本（scripts/qwen3_vl_embedding.py）。

    支持文本、图片、以及文本+图片混合输入，统一映射到 2048 维空间。
    """

    def __init__(self, model_path: str,
                 query_instruction: str = "Given a search query, retrieve "
                                          "relevant passages that answer the query.",
                 device: Optional[str] = None):
        import importlib.util
        script = os.path.join(model_path, "scripts", "qwen3_vl_embedding.py")
        if not os.path.exists(script):
            raise FileNotFoundError(f"缺少官方加载脚本: {script}")
        spec = importlib.util.spec_from_file_location("qwen3_vl_embedding", script)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self._emb = mod.Qwen3VLEmbedder(model_name_or_path=model_path)
        self.query_instruction = query_instruction

        # 官方脚本只认 cuda/cpu；Mac 上挪到 MPS 提速（device="cpu" 可强制 CPU）
        try:
            import torch
            if device != "cpu" and torch.backends.mps.is_available():
                self._emb.model.to("mps")
        except Exception:
            pass

    def encode(self, texts: List[str], batch_size: int = 8,
               show_progress_bar: bool = False,
               normalize_embeddings: bool = True,
               is_query: bool = False) -> np.ndarray:
        vectors = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            inputs = []
            for t in batch:
                item = {"text": t}
                if is_query:
                    item["instruction"] = self.query_instruction
                inputs.append(item)
            emb = self._emb.process(inputs, normalize=normalize_embeddings)
            vectors.append(emb.float().cpu().numpy())
        return np.vstack(vectors) if vectors else np.zeros((0, 2048))

    def encode_images(self, image_paths: List[str], batch_size: int = 4,
                      normalize_embeddings: bool = True) -> np.ndarray:
        """图片本体向量化（与文本同一空间，支持跨模态检索）。"""
        vectors = []
        for start in range(0, len(image_paths), batch_size):
            batch = image_paths[start:start + batch_size]
            inputs = [{"image": p} for p in batch]
            emb = self._emb.process(inputs, normalize=normalize_embeddings)
            vectors.append(emb.float().cpu().numpy())
        return np.vstack(vectors) if vectors else np.zeros((0, 2048))

    def get_sentence_embedding_dimension(self) -> int:
        vec = self._emb.process([{"text": "dim probe"}], normalize=False)
        return int(vec.shape[-1])

    @property
    def device(self):
        return str(self._emb.model.device)


# ═══════════════════════════════════════════════════════════════
# 主嵌入器
# ═══════════════════════════════════════════════════════════════

class TextEmbedder:
    """Qwen3-VL 文本/图片嵌入封装（惰性加载、批量、自动 MPS）。

    对外 API 与历史版本保持兼容：embed / embed_chunks / embed_chunks_hybrid /
    embed_query / dim / is_fallback / unload。
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        batch_size: int = 32,
        normalize: bool = True,
        show_progress: bool = True,
        cache_dir: Optional[str] = None,
        local_files_only: bool = False,
        fallback_dim: int = 256,
        use_fallback: bool = False,
        use_qwen3_vl: bool = True,
        modelscope_dir: Optional[str] = None,
    ):
        self.model_name = model_name
        self._device = device
        self.batch_size = batch_size
        self.normalize = normalize
        self.show_progress = show_progress
        self.cache_dir = cache_dir
        self.local_files_only = local_files_only
        self.fallback_dim = fallback_dim
        self.use_fallback = use_fallback
        self.use_qwen3_vl = use_qwen3_vl
        self.modelscope_dir = modelscope_dir

        self._model = None
        self._dim = None
        self._using_fallback = False

    # ── 公共 API ──────────────────────────────────────────

    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        single = isinstance(texts, str)
        if single:
            texts = [texts]
        self._ensure_model()
        start = time.time()
        vectors = self._model.encode(
            texts, batch_size=self.batch_size,
            show_progress_bar=self.show_progress and len(texts) > 10,
            normalize_embeddings=self.normalize,
        )
        elapsed = time.time() - start
        logger.info("Embedded %d texts in %.1fs, shape=%s",
                    len(texts), elapsed, vectors.shape)
        if single:
            return vectors[0]
        return vectors

    def embed_chunks(self, chunks: List[Any],
                     batch_size: Optional[int] = None) -> np.ndarray:
        texts = [c.text for c in chunks]
        bs = batch_size or self.batch_size
        self._ensure_model()
        vectors = self._model.encode(
            texts, batch_size=bs,
            show_progress_bar=self.show_progress and len(texts) > 10,
            normalize_embeddings=self.normalize,
        )
        for chunk, vec in zip(chunks, vectors):
            chunk.metadata["embedding_dim"] = len(vec)
        return vectors

    def embed_chunks_hybrid(self, chunks: List[Any],
                            batch_size: Optional[int] = None,
                            texts: Optional[List[str]] = None):
        """保留历史签名。Qwen3-VL 只有 dense，sparse 恒为空。"""
        texts = texts if texts is not None else [c.text for c in chunks]
        bs = batch_size or self.batch_size
        self._ensure_model()
        vectors = self._model.encode(
            texts, batch_size=bs,
            show_progress_bar=self.show_progress and len(texts) > 10,
            normalize_embeddings=self.normalize,
        )
        for chunk, vec in zip(chunks, vectors):
            chunk.metadata["embedding_dim"] = len(vec)
        return vectors, [{}] * len(chunks)

    def embed_query(self, query: str):
        """查询侧编码（带检索指令），返回 (dense, sparse={})。"""
        self._ensure_model()
        vec = self._model.encode(
            [query], batch_size=1,
            normalize_embeddings=self.normalize, is_query=True,
        )
        return np.asarray(vec[0]), {}

    @property
    def dim(self) -> int:
        self._ensure_model()
        return self._dim

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def is_fallback(self) -> bool:
        return self._using_fallback

    def unload(self):
        self._model = None
        self._dim = None

    # ── 内部 ─────────────────────────────────────────────

    def _ensure_model(self):
        if self._model is not None:
            return
        if self.use_fallback:
            self._init_fallback()
            return

        path = self.modelscope_dir or resolve_model_path(self.model_name)
        try:
            print(f"[Embedder] Loading Qwen3-VL: {path} ...", file=sys.stderr)
            self._model = Qwen3VLEmbedderAdapter(path, device=self._device)
            self._dim = self._model.get_sentence_embedding_dimension()
            self._using_fallback = False
            print(f"[Embedder] Loaded. Dim={self._dim}, Device={self._model.device}",
                  file=sys.stderr)
        except Exception as e:
            print(f"[Embedder] Qwen3-VL 加载失败，降级为哈希嵌入: {e}",
                  file=sys.stderr)
            self._init_fallback()

    def _init_fallback(self):
        print(f"[Embedder] Using fallback embedder (dim={self.fallback_dim}).",
              file=sys.stderr)
        self._model = FallbackEmbedder(dim=self.fallback_dim)
        self._dim = self.fallback_dim
        self._using_fallback = True
