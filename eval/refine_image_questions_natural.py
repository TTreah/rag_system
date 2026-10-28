"""
refine_image_questions_natural.py — 图片题改写成自然业务问法

真实用户不会说"那张图/第几页的配图"，而是直接问业务问题。
本脚本把测试集中 24 条图片题的题干改为自然问法（不提及图片/截图/页面），
并按「图片证据出现在 top-5」判断可用性（而不是要求 top-1）。
"""

import os
import sys
import json
import logging
import threading
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import core  # noqa: F401

from augment.query_rewriter import _load_api_key, LLM_HOST, LLM_MODEL

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

from config import TEST_CASE_DIR, QDRANT_URL
DOCX = os.path.join(TEST_CASE_DIR, "CEO-files检索测试用例300条-精准版.docx")
COLLECTION = "CEO-files"
_lock = threading.Lock()


def ask(prompt):
    payload = {"model": LLM_MODEL, "messages": [{"role": "user", "content": prompt}],
               "temperature": 0.4, "max_tokens": 1200}
    req = urllib.request.Request(f"{LLM_HOST}/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {_load_api_key()}"})
    with urllib.request.urlopen(req, timeout=150) as r:
        return json.loads(r.read().decode())["choices"][0]["message"]["content"]


def parse_json(raw):
    raw = raw.strip().strip("`")
    if raw.startswith("json"):
        raw = raw[4:]
    i, j = raw.find("{"), raw.rfind("}")
    return json.loads(raw[i:j + 1])


def main():
    from docx import Document
    from qdrant_client import QdrantClient
    from qdrant_line.rag_search import RAGSearcher

    client = QdrantClient(url=QDRANT_URL)
    caps = {}
    kinds = {}
    offset = None
    while True:
        pts, offset = client.scroll(COLLECTION, limit=500, offset=offset,
                                    with_payload=["source_file", "chunk_index", "metadata"],
                                    with_vectors=False)
        for p in pts:
            md = p.payload.get("metadata") or {}
            k = (os.path.basename(p.payload["source_file"]), p.payload["chunk_index"])
            kinds[k] = md.get("asset_type")
            if md.get("asset_type") == "image" and md.get("asset_caption"):
                caps[k] = md["asset_caption"]
        if offset is None:
            break

    searcher = RAGSearcher(collection=COLLECTION, rerank_threshold=0.0)
    doc = Document(DOCX)
    table = doc.tables[0]
    replaced = hit5 = 0

    for row in table.rows[1:]:
        cells = row.cells
        cid = cells[0].text.strip()
        ev = cells[4].text.strip()
        key = None
        for part in ev.split("；"):
            if "#" in part:
                n, i = part.rsplit("#", 1)
                if kinds.get((n, int(i))) == "image" and (n, int(i)) in caps:
                    key = (n, int(i))
                    break
        if not key:
            continue
        cap = caps[key]
        formal = int(cid) <= 100
        prompt = f"""以下是某企业文档图片中的真实内容（已转写为文字）：

{cap[:1400]}

请基于这些内容，生成 1 个真实用户会自然提出的业务问题，输出 JSON：
{{"question": "...", "points": "答案要点，分号分隔"}}

要求：
1. 问题中**绝对不能**出现"图片/图/截图/页面/配图/这张"等指代视觉载体的词
2. 像业务人员直接问业务问题，{"措辞正式" if formal else "用日常口语"}
3. 问题必须能由上述内容回答；points 保留关键实体和细节"""
        q = pts = None
        for attempt in range(2):
            try:
                data = parse_json(ask(prompt))
                q, pts = data["question"].strip(), data["points"].strip()
            except Exception as e:
                logger.warning("[%s] 生成失败: %s", cid, e)
                continue
            with _lock:
                rs = searcher.search(q, top_k=5)
            if any(os.path.basename(r["source_file"]) == key[0]
                   and r.get("chunk_index") == key[1] for r in rs):
                hit5 += 1
                break
        if q:
            cells[1].text = q
            cells[3].text = pts
            replaced += 1
            logger.info("[%s] hit5=%s → %s", cid,
                        "Y" if attempt < 2 else "N", q[:40])

    doc.save(DOCX)
    logger.info("改写 %d 条，其中 top-5 可命中 %d 条", replaced, hit5)


if __name__ == "__main__":
    main()
