"""
update_image_assets.py — 图片资产升级：真实描述 + 图片本体向量

背景：原先图片只嵌入「上下文猜的描述文本」，多模态模型形同虚设。
本脚本分两阶段（避免两个 2B 模型同时驻留内存）：

    阶段 1  Qwen3-VL-2B-Instruct 看图生成描述 → 缓存到 JSON
    阶段 2  Qwen3-VL-Embedding 对图片本体向量化 → 原地更新 Qdrant 点的
            dense 向量，并把 payload 的 text 换成真实描述

点 ID 不变，只更新向量与描述，文本块完全不动。

用法:
    python qdrant_line/update_image_assets.py --limit 20   # 先试 20 张
    python qdrant_line/update_image_assets.py              # 全量
"""

import os
import sys
import json
import time
import argparse
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import core  # noqa: F401

from core.embedder import Qwen3VLEmbedderAdapter, resolve_model_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

COLLECTION = "CEO-files"
CAPTION_CACHE = "/tmp/image_captions.json"
VLM_MODEL = "Qwen/Qwen3-VL-2B-Instruct"
EMB_MODEL = "Qwen/Qwen3-VL-Embedding-2B"


def collect_image_points(client):
    points, offset = [], None
    while True:
        pts, offset = client.scroll(COLLECTION, limit=300, offset=offset,
                                    with_payload=True, with_vectors=False)
        for p in pts:
            md = p.payload.get("metadata") or {}
            if (md.get("asset_type") == "image" and md.get("asset_path")
                    and md.get("point_kind") != "image_caption"):
                points.append({
                    "id": p.id,
                    "path": md["asset_path"],
                    "text": p.payload.get("text", ""),
                    "metadata": md,
                    "source_file": p.payload.get("source_file", ""),
                    "chunk_index": p.payload.get("chunk_index", -1),
                })
        if offset is None:
            break
    return points


def phase_caption(points, limit=None):
    """阶段 1：生成真实描述（可断点续跑）"""
    from core.captioner import caption_image
    cache = {}
    if os.path.exists(CAPTION_CACHE):
        cache = json.load(open(CAPTION_CACHE, encoding="utf-8"))
        logger.info("已有描述缓存 %d 条", len(cache))
    todo = [p for p in points if p["path"] not in cache]
    if limit:
        todo = todo[:limit]
    model_path = resolve_model_path(VLM_MODEL)
    logger.info("待生成描述 %d 张", len(todo))
    for i, p in enumerate(todo, 1):
        if not os.path.exists(p["path"]):
            cache[p["path"]] = ""
            continue
        try:
            cap = caption_image(p["path"], model_path)
        except Exception as e:
            logger.warning("描述失败 %s: %s", p["path"], e)
            cap = ""
        cache[p["path"]] = cap
        if i % 10 == 0 or i == len(todo):
            json.dump(cache, open(CAPTION_CACHE, "w", encoding="utf-8"),
                      ensure_ascii=False)
            logger.info("描述进度 %d/%d", i, len(todo))
    json.dump(cache, open(CAPTION_CACHE, "w", encoding="utf-8"), ensure_ascii=False)
    return cache


def phase_embed(points, cache, batch_size=4):
    """阶段 2：图片本体向量化 + 原地更新"""
    from qdrant_client import QdrantClient
    from qdrant_client.models import PointStruct
    client = QdrantClient(url="http://localhost:6333")
    embedder = Qwen3VLEmbedderAdapter(resolve_model_path(EMB_MODEL))

    done = 0
    for start in range(0, len(points), batch_size):
        batch = points[start:start + batch_size]
        valid = [p for p in batch if os.path.exists(p["path"])
                 and (p["metadata"].get("asset_vector") != "image")]
        if not valid:
            continue
        vecs = embedder.encode_images([p["path"] for p in valid])
        upserts = []
        for p, vec in zip(valid, vecs):
            cap = (cache.get(p["path"]) or "").strip()
            prefix = p["text"].split("\n", 1)[0]          # 【文件｜位置】
            new_text = prefix + "\n[图片] " + (cap or "（无描述）")
            md = dict(p["metadata"])
            md["asset_caption"] = cap
            md["asset_vector"] = "image"                  # 标记已用图片本体向量
            md["embedding_dim"] = int(len(vec))
            payload = {
                "text": new_text,
                "source_file": p["source_file"],
                "chunk_index": p["chunk_index"],
                "metadata": md,
            }
            upserts.append(PointStruct(id=p["id"],
                                       vector={"dense": vec.tolist()},
                                       payload=payload))
        client.upsert(COLLECTION, points=upserts, wait=True)
        done += len(valid)
        logger.info("向量更新 %d/%d", done, len(points))
    return done


def main():
    global COLLECTION
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="只处理前 N 张（试跑）")
    ap.add_argument("--missing-only", action="store_true",
                    help="只升级尚未使用图片向量的点（增量）")
    ap.add_argument("--collection", default=COLLECTION)
    args = ap.parse_args()
    COLLECTION = args.collection

    from qdrant_client import QdrantClient
    client = QdrantClient(url="http://localhost:6333")
    points = collect_image_points(client)
    if args.missing_only:
        before = len(points)
        points = [p for p in points if p["metadata"].get("asset_vector") != "image"]
        logger.info("增量模式：%d → %d 个待升级", before, len(points))
    logger.info("图片资产点 %d 个", len(points))

    cache = phase_caption(points, limit=args.limit)
    # 让出内存：描述模型卸载后再加载嵌入模型（24GB 机器装不下两个 2B）
    from core import captioner
    captioner.unload()
    if args.limit:
        points = [p for p in points if p["path"] in cache][:args.limit]
    n = phase_embed(points, cache)
    logger.info("完成：%d 个图片点已更新为图片向量 + 真实描述", n)


if __name__ == "__main__":
    main()
