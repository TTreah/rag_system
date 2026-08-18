"""
test_embedder_scenarios.py — 测试 embedder 两种加载方案

方案一 (离线):  use_flag + 默认离线变量    → 从缓存加载 BGE-M3，不联网
方案二 (在线):  use_flag + 允许联网        → BGE-M3，允许检查 HF 更新
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from embedder import TextEmbedder


def hr(title):
    print(f"\n{'=' * 65}")
    print(f"  {title}")
    print(f"{'=' * 65}")


# ─── 方案一：离线模式（默认，缓存中加载 BGE-M3）─────────────

def test_offline():
    hr("TEST 1/2 — Offline mode (default, cached BGE-M3)")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    t0 = time.time()
    embedder = TextEmbedder(model_name="bge-m3", use_flag_embedding=True)
    vec = embedder.embed(["你好世界 Hello world"])
    elapsed = time.time() - t0

    ok = embedder.is_flag and not embedder.is_fallback and embedder.dim == 1024
    print(f"  Dim: {embedder.dim}")
    print(f"  is_fallback: {embedder.is_fallback}")
    print(f"  is_flag:     {embedder.is_flag}")
    print(f"  Shape:       {vec.shape}")
    print(f"  Time:        {elapsed:.2f}s")

    if ok:
        vec2 = embedder.embed(["完全不同的另一段文本"])
        cosine = float((vec[0] @ vec2[0]) /
                       (__import__('numpy').linalg.norm(vec[0]) *
                        __import__('numpy').linalg.norm(vec2[0])))
        print(f"  Cosine sim (diff texts): {cosine:.4f} (should be < 0.95)")
    print(f"  Result:      {'[PASS]' if ok else '[FAIL]'}")
    return ok


# ─── 方案二：在线模式（允许联网检查更新）────────────────────

def test_online():
    hr("TEST 2/2 — Online mode (allow HF version check)")
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"

    t0 = time.time()
    embedder = TextEmbedder(model_name="bge-m3", use_flag_embedding=True)
    vec = embedder.embed(["你好世界 Hello world"])
    elapsed = time.time() - t0

    ok = embedder.is_flag and not embedder.is_fallback and embedder.dim == 1024
    print(f"  Dim: {embedder.dim}")
    print(f"  is_fallback: {embedder.is_fallback}")
    print(f"  is_flag:     {embedder.is_flag}")
    print(f"  Shape:       {vec.shape}")
    print(f"  Time:        {elapsed:.2f}s")
    print(f"  Result:      {'[PASS]' if ok else '[FAIL]'}")
    return ok


# ─── Main ───────────────────────────────────────────────────

if __name__ == "__main__":
    print("embedder test — offline / online")
    print(f"Python:  {sys.executable}")

    results = {}
    results["offline"] = test_offline()
    results["online"]  = test_online()

    hr("SUMMARY")
    all_pass = True
    for name, ok in results.items():
        print(f"  {name:12s}: {'[PASS]' if ok else '[FAIL]'}")
        all_pass = all_pass and ok
    print(f"\n  {'Overall: ALL PASS' if all_pass else 'Overall: SOME FAILED'}")
    sys.exit(0 if all_pass else 1)
