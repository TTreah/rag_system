"""
reranker.py — Qwen3-VL-Reranker-2B 重排序 + 相关性闸门

打分方式：模型是 decoder-only，读取 "yes"/"no" 两个 token 的 logits，
输出 0~1 相关性分（模型仓库自带的官方脚本封装）。

BGE-Reranker / FlagEmbedding 后端已移除；脚本内不再依赖 FlagEmbedding。
"""

import os
import logging
from typing import Optional

from config import RERANK_MODEL as _CFG_RERANK_MODEL

logger = logging.getLogger(__name__)

DEFAULT_RERANK_MODEL = _CFG_RERANK_MODEL

# 相关性闸门阈值（Qwen3-VL 分数尺度，基于 CEO-files 300 条实测标定）：
#   无关候选最高 ≈0.45，相关弱命中 ≥0.50；取 0.50。
# 语料扩大后可用 eval/calibrate_threshold.py 重新标定。
RERANK_THRESHOLD = 0.50


class Reranker:
    """惰性加载的 Qwen3-VL 重排器。"""

    def __init__(self, model_name: str = DEFAULT_RERANK_MODEL,
                 use_fp16: bool = True):
        self.model_name = model_name
        self.use_fp16 = use_fp16
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            from core.embedder import resolve_model_path
            path = resolve_model_path(self.model_name)
            script = os.path.join(path, "scripts", "qwen3_vl_reranker.py")
            if not os.path.exists(script):
                raise FileNotFoundError(f"缺少官方加载脚本: {script}")
            print(f"[Reranker] Loading: {path} ...", flush=True)
            import importlib.util
            spec = importlib.util.spec_from_file_location("qwen3_vl_reranker", script)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            self._model = mod.Qwen3VLReranker(model_name_or_path=path)
            try:
                import torch
                if torch.backends.mps.is_available():
                    # 主干和二分类头都要挪；脚本默认只认 cuda/cpu
                    self._model.model.to("mps")
                    self._model.score_linear.to("mps")
                    self._model.device = torch.device("mps")
            except Exception:
                pass
        return self._model

    def score(self, query: str, texts: list) -> list:
        """对 (query, text) 对打分，返回 0~1 相关性分数列表。"""
        if not texts:
            return []
        model = self._ensure_model()
        inputs = {
            "instruction": "Given a search query, retrieve relevant "
                           "candidates that answer the query.",
            "query": {"text": query},
            "documents": [{"text": t} for t in texts],
        }
        scores = model.process(inputs)
        return [float(s) for s in scores]
