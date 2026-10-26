"""backfill_payload.py — 补齐点的顶层 payload 字段（图片点/描述点升级时丢失的）"""

import os
import sys
import hashlib
import datetime
import logging
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import core  # noqa: F401

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

COLLECTION = "CEO-files"
_HASH_CHUNK = 1024 * 1024


def file_hash(path):
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(_HASH_CHUNK), b""):
                h.update(block)
        return h.hexdigest()
    except OSError:
        return ""


def main():
    from qdrant_client import QdrantClient
    from qdrant_line.vectorize_pipeline import PipelineConfig, pipeline_fingerprint

    fp = pipeline_fingerprint(PipelineConfig(qa_augment=True))
    client = QdrantClient(url="http://localhost:6333")

    by_file = defaultdict(list)
    offset = None
    while True:
        pts, offset = client.scroll(COLLECTION, limit=500, offset=offset,
                                    with_payload=True, with_vectors=False)
        for p in pts:
            by_file[p.payload.get("source_file", "")].append(p)
        if offset is None:
            break
    logger.info("文件 %d 个，点 %d 个", len(by_file), sum(len(v) for v in by_file.values()))

    now = datetime.datetime.now().isoformat()
    fixed = 0
    for path, pts in by_file.items():
        if not path:
            continue
        need_hash = any(not p.payload.get("file_hash") for p in pts)
        try:
            st = os.stat(path)
            mtime = datetime.datetime.fromtimestamp(st.st_mtime).isoformat()
            size = int(st.st_size)
        except OSError:
            mtime, size = "", 0
        fhash = file_hash(path) if need_hash else ""

        def fill(field, value, batch=500):
            ids = [p.id for p in pts if not p.payload.get(field)]
            for i in range(0, len(ids), batch):
                client.set_payload(COLLECTION, payload={field: value},
                                   points=ids[i:i + batch], wait=False)
            return len(ids)

        n = 0
        n += fill("file_type", os.path.splitext(path)[1].lower())
        n += fill("file_mtime", mtime)
        if fhash:
            n += fill("file_hash", fhash)
        n += fill("pipeline_fingerprint", fp)
        n += fill("ingested_at", now)
        fixed += n

        # 逐点补 section_label（从上下文头 "【文件｜标签】" 解析）
        for p in pts:
            if p.payload.get("section_label"):
                continue
            text = p.payload.get("text", "")
            label = ""
            if text.startswith("【") and "｜" in text.split("\n", 1)[0]:
                head = text.split("\n", 1)[0]
                label = head.split("｜", 1)[1].rstrip("】")
            if label:
                client.set_payload(COLLECTION, payload={"section_label": label},
                                   points=[p.id], wait=False)

    logger.info("回填完成：涉及 %d 个点，指纹统一为缺失项 %s", fixed, fp)


if __name__ == "__main__":
    main()
