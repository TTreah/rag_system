"""
reranker.py — BGE-Reranker-v2-m3 交叉编码器重排序 + 相关性闸门

为什么需要它（40 条评测暴露的两个问题都靠它解决）：
    1. 词汇鸿沟排序：dense+sparse 召回后，"想下架"命中"物料下架"这类
       一词多义错排，向量召回无法分辨；交叉编码器把 (查询, 文本) 拼在一起
       过模型，输出真实相关性分，能压掉这种假命中
    2. 相关性闸门：RRF 分数按排名计算，相关/无关查询的分数区间重叠
       （实测无关 top-1 也有 0.75），无法设阈值拒答；reranker 的输出
       经 sigmoid 归一是 0~1 的真实相关度，可以设阈值把无关查询整单拒掉

注意：Ollama 无法提供该模型的重排序能力（它的 GGUF 是给 llama.cpp 的
reranking 模式用的，Ollama 没有对应端点，实测 /api/embed 崩溃、
/api/generate 返回空、/api/rerank 404），所以走 FlagEmbedding + 本地
PyTorch 权重（ModelScope 缓存）。

使用方式:
    from core.reranker import Reranker
    rr = Reranker()
    scores = rr.score("查询", ["候选文本1", "候选文本2"])   # 0~1，越高越相关
"""

import os
import logging

logger = logging.getLogger(__name__)

DEFAULT_RERANK_MODEL = "Qwen/Qwen3-VL-Reranker-2B"

# 相关性闸门阈值。注意：两个后端的分数尺度完全不同，阈值分两套——
#
# Qwen3-VL-Reranker（当前默认，100 条用例实测标定，重排输入含 QA 假设问题）：
#   无关候选最高 0.264（感冒发烧/Excel）；口语化弱相关最低 0.377；
#   正式相关 0.5~0.99。QA 增强后重标定（60 条抽样）：无关最高 0.4512，相关最低 top 分 0.5156，
#   间隙 [0.45, 0.52]，取 0.50。语料扩大后需用 eval/calibrate_threshold.py 重新标定。
#   对比 BGE 时代：重叠区消失，模糊题得分整体抬升一个量级
#   （"想下架"题 0.0055 → 0.418）。
#
# BGE-Reranker-v2-m3（历史标定）：
#   无关 ≈0.000~0.024，弱相关 0.014~0.066，阈值 0.02。
#   换回 BGE 后端时阈值要同步换回 0.02。
QWEN3VL_RERANK_THRESHOLD = 0.50
BGE_RERANK_THRESHOLD = 0.02
RERANK_THRESHOLD = QWEN3VL_RERANK_THRESHOLD


class Reranker:
    """惰性加载的重排器，支持两种后端：

    - BGE-Reranker-v2-m3（FlagEmbedding，分类头打分）
    - Qwen3-VL-Reranker（transformers + 官方脚本，yes/no logit 打分，
      支持多模态；Ollama 的 VL 版 GGUF 转换质量不可用，故用原版权重）
    """

    def __init__(self, model_name: str = DEFAULT_RERANK_MODEL,
                 use_fp16: bool = True):
        self.model_name = model_name
        self.use_fp16 = use_fp16
        self._model = None
        self._is_qwen3vl = "Qwen3-VL" in model_name

    def _ensure_model(self):
        if self._model is None:
            from core.embedder import resolve_model_path
            path = resolve_model_path(self.model_name)
            print(f"[Reranker] Loading: {path} ...", flush=True)
            if self._is_qwen3vl:
                import importlib.util
                spec = importlib.util.spec_from_file_location(
                    "qwen3_vl_reranker",
                    os.path.join(path, "scripts", "qwen3_vl_reranker.py"),
                )
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                self._model = mod.Qwen3VLReranker(model_name_or_path=path)
                try:
                    import torch
                    if torch.backends.mps.is_available():
                        # 主干和二分类头都要挪，脚本默认只认 cuda/cpu
                        self._model.model.to("mps")
                        self._model.score_linear.to("mps")
                        self._model.device = torch.device("mps")
                except Exception:
                    pass
            else:
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
                os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
                from FlagEmbedding import FlagReranker
                self._model = FlagReranker(path, use_fp16=self.use_fp16)
        return self._model

    def score(self, query: str, texts: list) -> list:
        """对 (query, text) 对打分，返回 0~1 相关性分数列表。"""
        if not texts:
            return []
        model = self._ensure_model()
        if self._is_qwen3vl:
            inputs = {
                "instruction": "Given a search query, retrieve relevant "
                               "candidates that answer the query.",
                "query": {"text": query},
                "documents": [{"text": t} for t in texts],
            }
            scores = model.process(inputs)
            return [float(s) for s in scores]
        pairs = [[query, t] for t in texts]
        scores = model.compute_score(pairs, normalize=True)
        if isinstance(scores, (int, float)):
            scores = [scores]
        return [float(s) for s in scores]
