"""
add_caption_points.py — 为图片资产补一个「描述文本点」

背景：图片点的向量是图片本体，擅长视觉语义（"白板上的手绘图表"），
但匹配不了图内具体文字（"横轴写的是什么"）。补一个同源文本点：
    - 图片点   → 图片向量（视觉检索）
    - 描述点   → 描述文本向量（细节检索）
两者 source_file / chunk_index 相同，检索结果按 (文件, 块号) 去重。
"""

import os
import sys
import logging
import hashlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import core  # noqa: F401

from core.embedder import Qwen3VLEmbedderAdapter, resolve_model_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

from config import QDRANT_URL, EMBEDDING_MODEL as EMB_MODEL, COLLECTION as _CFG_COLLECTION

COLLECTION = _CFG_COLLECTION



def pid_for(source_file: str, chunk_index: int) -> str:
    h = hashlib.md5(f"caption|{source_file}|{chunk_index}".encode()).hexdigest()
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


def main():
    global COLLECTION
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", default=COLLECTION)
    args = ap.parse_args()
    COLLECTION = args.collection

    from qdrant_client import QdrantClient
    from qdrant_client.models import PointStruct

    client = QdrantClient(url=QDRANT_URL)
    points, offset = [], None
    existing = set()
    all_points = []
    while True:
        pts, offset = client.scroll(COLLECTION, limit=300, offset=offset,
                                    with_payload=True, with_vectors=False)
        all_points.extend(pts)
        if offset is None:
            break
    for p in all_points:
        md = p.payload.get("metadata") or {}
        if md.get("point_kind") == "image_caption":
            existing.add((p.payload.get("source_file"), p.payload.get("chunk_index")))
    for p in all_points:
        md = p.payload.get("metadata") or {}
        if md.get("asset_type") == "image" and md.get("asset_caption"):
            if (p.payload.get("source_file"), p.payload.get("chunk_index")) in existing:
                continue
            points.append(p)
    logger.info("待补描述点的图片 %d 个", len(points))

    embedder = Qwen3VLEmbedderAdapter(resolve_model_path(EMB_MODEL))
    B = 16
    done = 0
    for start in range(0, len(points), B):
        batch = points[start:start + B]
        caps = [(p.payload.get("metadata") or {})["asset_caption"] for p in batch]
        vecs = embedder.encode(caps)
        upserts = []
        for p, cap, vec in zip(batch, caps, vecs):
            src = p.payload.get("source_file", "")
            idx = p.payload.get("chunk_index", 0)
            md = dict(p.payload.get("metadata") or {})
            md["point_kind"] = "image_caption"
            md["embedding_dim"] = int(len(vec))
            prefix = p.payload.get("text", "").split("\n", 1)[0]
            upserts.append(PointStruct(
                id=pid_for(src, idx),
                vector={"dense": vec.tolist()},
                payload={
                    "text": prefix + "\n[图片描述] " + cap,
                    "source_file": src,
                    "chunk_index": idx,
                    "metadata": md,
                },
            ))
        client.upsert(COLLECTION, points=upserts, wait=True)
        done += len(batch)
        if done % 160 == 0 or done == len(points):
            logger.info("描述点进度 %d/%d", done, len(points))
    logger.info("完成：新增描述点 %d 个", done)


if __name__ == "__main__":
    main()
