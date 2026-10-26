"""
test_embedder_scenarios.py — Qwen3-VL 嵌入器冒烟测试

覆盖：
  1. 从本地 ModelScope 缓存加载 Qwen3-VL-Embedding-2B（维度 2048）
  2. 文本相似度合理性（相关 > 无关）
  3. 查询侧嵌入（带检索指令）
  4. 图片向量化（若提供 --image 路径）
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import core  # noqa: F401

import numpy as np
from core.embedder import TextEmbedder


def hr(title):
    print(f"\n{'=' * 65}\n  {title}\n{'=' * 65}")


def main():
    hr("Qwen3-VL-Embedding 冒烟测试")
    embedder = TextEmbedder()  # 默认 Qwen/Qwen3-VL-Embedding-2B
    vec = embedder.embed(["你好世界 Hello world"])
    print(f"  Dim:         {embedder.dim}")
    print(f"  is_fallback: {embedder.is_fallback}")
    print(f"  Shape:       {vec.shape}")

    ok = (not embedder.is_fallback) and embedder.dim == 2048

    a = embedder.embed(["千里达运动产品开发控制程序"])[0]
    b = embedder.embed(["千里达新产品定价标准作业流程"])[0]
    c = embedder.embed(["红烧肉的做法"])[0]
    sim_rel = float(a @ b)
    sim_irr = float(a @ c)
    print(f"  相关文本相似度: {sim_rel:.4f}")
    print(f"  无关文本相似度: {sim_irr:.4f}")
    ok = ok and sim_rel > sim_irr

    q, sparse = embedder.embed_query("M985 的目标用户是谁？")
    print(f"  查询向量: {q.shape} | sparse: {sparse}")
    ok = ok and q.shape == (2048,)

    img = None
    for i, arg in enumerate(sys.argv):
        if arg == "--image" and i + 1 < len(sys.argv):
            img = sys.argv[i + 1]
    if img and os.path.exists(img):
        from core.embedder import Qwen3VLEmbedderAdapter, resolve_model_path
        adapter = Qwen3VLEmbedderAdapter(resolve_model_path(embedder.model_name))
        iv = adapter.encode_images([img])[0]
        print(f"  图片向量: {iv.shape}")
        ok = ok and iv.shape == (2048,)

    hr("SUMMARY")
    print(f"  {'[PASS]' if ok else '[FAIL]'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
