"""refresh_image_questions.py — 用真实图片描述重写测试集中的垃圾图片题"""

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
REPLACE_IDS = {"25", "26", "56", "57",
               "236", "237", "238", "240", "241", "243", "244", "245",
               "246", "247", "248", "249", "250"}
_lock = threading.Lock()


def ask(prompt):
    payload = {"model": LLM_MODEL, "messages": [{"role": "user", "content": prompt}],
               "temperature": 0.5, "max_tokens": 1500}
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
    caps_by_name = {}
    offset = None
    while True:
        pts, offset = client.scroll(COLLECTION, limit=500, offset=offset,
                                    with_payload=["source_file", "chunk_index", "metadata"],
                                    with_vectors=False)
        for p in pts:
            md = p.payload.get("metadata") or {}
            if md.get("asset_type") == "image" and md.get("asset_caption"):
                name = os.path.basename(p.payload["source_file"])
                idx = p.payload["chunk_index"]
                caps[(name, idx)] = md["asset_caption"]
                caps_by_name[name] = (idx, md["asset_caption"])
        if offset is None:
            break
    logger.info("可用图片描述 %d 条", len(caps))

    searcher = RAGSearcher(collection=COLLECTION, rerank_threshold=0.0)
    doc = Document(DOCX)
    table = doc.tables[0]
    replaced = 0

    for row_idx, row in enumerate(table.rows[1:], 1):
        cells = row.cells
        cid = cells[0].text.strip()
        if cid not in REPLACE_IDS:
            continue
        ev = cells[4].text.strip()
        key = None
        for part in ev.split("；"):
            if "#" in part:
                n, i = part.rsplit("#", 1)
                key = (n, int(i))
                if key in caps:
                    break
                # 证据块号可能与实际不一致（历史生成遗留）：按文件名纠正
                if n in caps_by_name:
                    key = (n, caps_by_name[n][0])
                    break
        if not key or key not in caps:
            logger.info("[%s] 无描述，跳过", cid)
            continue
        cap = caps[key]
        formal = int(cid) <= 100
        prompt = f"""这是一张企业文档图片的真实内容描述：
{cap[:1200]}

请基于描述生成 1 道{"正式的" if formal else "一线员工口语化的"}检索测试题，
输出 JSON：{{"question": "...", "points": "答案要点，分号分隔，必须来自描述"}}
要求：问题不要出现"图片描述中""根据描述"等元表述；points 保留关键实体和细节。"""
        try:
            data = parse_json(ask(prompt))
            q, pts = data["question"].strip(), data["points"].strip()
        except Exception as e:
            logger.warning("[%s] 生成失败: %s", cid, e)
            continue
        with _lock:
            rs = searcher.search(q, top_k=10)
        hit = any(os.path.basename(r["source_file"]) == key[0]
                  and r.get("chunk_index") == key[1] for r in rs)
        if not hit:
            # 题目合法（基于真实描述），是否命中是系统能力问题，不应因此丢弃
            logger.info("[%s] 当前系统未命中（保留题目）: %s", cid, q[:30])
        cells[1].text = q
        cells[3].text = pts
        cells[4].text = f"{key[0]}#{key[1]}"   # 纠正依据块号
        replaced += 1
        logger.info("[%s] 已替换 → %s", cid, q[:40])

    doc.save(DOCX)
    logger.info("共替换 %d 条", replaced)


if __name__ == "__main__":
    main()
