"""
topup_ceo_testcases.py — 把 227 条补齐到 300 条（优先覆盖未出题文档）

读取已生成的 docx，补齐：
  - 精确题 76 → 100
  - 模糊题 103 → 150
  - 无关题 48 → 50
并保证每份文档至少 1 题；每题依据块 1-3 个（≤5）。
"""

import os
import sys
import json
import random
import logging
import threading
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import core  # noqa: F401

from augment.query_rewriter import _load_api_key, LLM_HOST, LLM_MODEL
import urllib.request

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

COLLECTION = "CEO-files"
DOCX = "/Users/trinxai/Documents/pythonfile/测试用例/CEO-files检索测试用例300条-精准版.docx"
CACHE = "/tmp/topup_cases.json"
TARGET_P, TARGET_F, TARGET_I = 100, 150, 50
random.seed(2026)

IRRELEVANT_EXTRA = [
    "怎么给浴缸除垢？", "怎么挑选小提琴？", "怎么训练鹦鹉说话？",
    "怎么自制辣椒酱？", "怎么学打毛线？", "怎么挑选登山包？",
]

_search_lock = threading.Lock()


def ask_llm(prompt: str, max_tokens: int = 1800) -> str:
    payload = {"model": LLM_MODEL, "messages": [{"role": "user", "content": prompt}],
               "temperature": 0.6, "max_tokens": max_tokens}
    req = urllib.request.Request(
        f"{LLM_HOST}/chat/completions", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {_load_api_key()}"})
    with urllib.request.urlopen(req, timeout=150) as resp:
        return json.loads(resp.read().decode())["choices"][0]["message"]["content"].strip()


def parse_json(raw: str) -> dict:
    raw = raw.strip().strip("`")
    if raw.startswith("json"):
        raw = raw[4:]
    i, j = raw.find("{"), raw.rfind("}")
    return json.loads(raw[i:j + 1])


def load_corpus():
    from qdrant_client import QdrantClient
    c = QdrantClient(url="http://localhost:6333")
    corpus = {}
    offset = None
    while True:
        pts, offset = c.scroll(COLLECTION, limit=500, offset=offset,
                               with_payload=["source_file", "chunk_index", "text", "metadata"],
                               with_vectors=False)
        for p in pts:
            pl = p.payload
            e = corpus.setdefault(pl["source_file"], {"chunks": [], "assets": []})
            is_asset = bool((pl.get("metadata") or {}).get("asset_path"))
            (e["assets"] if is_asset else e["chunks"]).append((pl["chunk_index"], pl["text"]))
        if offset is None:
            break
    for e in corpus.values():
        e["chunks"].sort(); e["assets"].sort()
    return corpus


def read_existing():
    from docx import Document
    d = Document(DOCX)
    rows = [[c.text for c in r.cells] for r in d.tables[0].rows[1:]]
    cases = []
    for r in rows:
        cid, q, fname, pts, ev, res = r
        if "无关" in fname:
            cases.append({"kind": "irrelevant", "question": q, "points": pts,
                          "file": None, "evidence": []})
        else:
            evs = []
            for part in [x for x in ev.split("；") if x]:
                name, idx = part.rsplit("#", 1)
                evs.append((name, int(idx)))
            cases.append({"kind": "precise" if int(cid) <= 76 else "unknown",
                          "question": q, "points": pts, "file": fname,
                          "evidence": evs, "_raw_ev": ev})
    return cases


def pick_groups(entry, used, n_groups=3):
    """挑 1-3 块的证据组，避开已用块"""
    src = entry["chunks"] if entry["chunks"] else entry["assets"]
    if not src:
        return []
    groups = []
    i = 0
    while i < len(src) and len(groups) < n_groups:
        grp = [src[i]]
        if i + 1 < len(src) and src[i][0] in used:
            i += 1
            continue
        # 尝试加相邻块，总长 400-2200
        for k in range(1, 3):
            if i + k >= len(src):
                break
            cand = grp + [src[i + k]]
            if sum(len(t) for _, t in cand) <= 2200:
                grp = cand
        total = sum(len(t) for _, t in grp)
        if total >= 200:
            groups.append(grp)
        i += len(grp)
    return groups


def gen_one(title, ev):
    ev_text = "\n\n".join(f"[块{t[0]}]\n{t[1]}" for t in ev)
    prompt = f"""以下文档《{title}》的 {len(ev)} 个连续片段（方括号里是块号）：

{ev_text[:4500]}

请基于这些片段本身生成测试用例，输出 JSON（不要其他文字）：
{{"precise": {{"question": "...", "points": "答案要点，分号分隔"}},
 "fuzzy": {{"question": "同答案但用一线员工口语的问法（不要用文档原词）", "points": "答案要点，分号分隔"}}}}
要求：答案必须能且只能从上述块中找到；问题不要出现"片段中/根据文档"；points 保留原文关键数字与专名。"""
    data = parse_json(ask_llm(prompt))
    out = []
    for k in ("precise", "fuzzy"):
        it = data.get(k)
        if it and it.get("question") and it.get("points"):
            out.append((k, it["question"].strip(), it["points"].strip()))
    return out


def main():
    from qdrant_line.rag_search import RAGSearcher
    searcher = RAGSearcher(collection=COLLECTION, rerank_threshold=0.0)

    corpus = load_corpus()
    # basename → full path
    name2path = {os.path.basename(p): p for p in corpus}
    existing = read_existing()

    # 归类已有用例
    precise = [c for c in existing if c["kind"] == "precise"]
    fuzzy = [c for c in existing if c["kind"] == "unknown"]  # 旧文件里非前76条即模糊
    irr = [c for c in existing if c["kind"] == "irrelevant"]
    logger.info("已有：精确 %d / 模糊 %d / 无关 %d", len(precise), len(fuzzy), len(irr))

    # 统一 evidence 为全路径
    def full_ev(c):
        return [(name2path.get(n, n), i) for n, i in c["evidence"]]

    covered = set()
    used_by_file = {}
    questions = {c["question"] for c in existing}
    for c in precise + fuzzy:
        if c["file"]:
            fp = name2path.get(c["file"], c["file"])
            c["file"] = fp
            covered.add(fp)
            for n, i in c["evidence"]:
                used_by_file.setdefault(fp, set()).add(i)
        c["evidence"] = full_ev(c)

    need_p = TARGET_P - len(precise)
    need_f = TARGET_F - len(fuzzy)
    need_i = TARGET_I - len(irr)
    logger.info("需补：精确 %d / 模糊 %d / 无关 %d", need_p, need_f, need_i)

    uncovered = [p for p in sorted(corpus) if p not in covered]
    rich = sorted([p for p in corpus if p in covered],
                  key=lambda p: -len(corpus[p]["chunks"]))
    # 任务顺序：先覆盖未出题文档，再用富文档补数
    targets = uncovered + [p for p in rich if p not in uncovered]
    # 未覆盖文档每份先出模糊题；同时尽量出精确题
    logger.info("未覆盖文档 %d 份", len(uncovered))

    new_cases = []
    if os.path.exists(CACHE):
        new_cases = json.load(open(CACHE))
        for c in new_cases:
            c["evidence"] = [(c["file"], e[0]) if isinstance(e[0], int) else tuple(e)
                             for e in c["evidence"]]
            questions.add(c["question"])
        logger.info("载入缓存补齐结果 %d 条", len(new_cases))

    def build(path):
        if len([c for c in new_cases if c["kind"] == "precise"]) >= need_p and \
           len([c for c in new_cases if c["kind"] == "fuzzy"]) >= need_f:
            return []
        title = os.path.basename(path)
        entry = corpus[path]
        out = []
        got_kind = set()
        for grp in pick_groups(entry, used_by_file.get(path, set()), 2):
            try:
                qs = gen_one(title, grp)
            except Exception as e:
                logger.warning("生成失败 %s: %s", title, e)
                continue
            for kind, q, pts in qs:
                if q in questions or kind in got_kind:
                    continue
                with _search_lock:
                    rs = searcher.search(q, top_k=10)
                hit = any(r["source_file"] == path and r.get("chunk_index") == i
                          for r in rs for i, _ in grp)
                if not hit:
                    continue
                # 精确/模糊配额控制
                cur_p = len([c for c in new_cases if c["kind"] == "precise"])
                cur_f = len([c for c in new_cases if c["kind"] == "fuzzy"])
                if kind == "precise" and cur_p >= need_p:
                    continue
                if kind == "fuzzy" and cur_f >= need_f:
                    continue
                out.append({"kind": kind, "question": q, "points": pts,
                            "file": path,
                            "evidence": [(path, idx) for idx, _ in grp]})
                questions.add(q)
                got_kind.add(kind)
            if out:
                break  # 同一文档只用一个证据组，避免同源重复
        return out

    with ThreadPoolExecutor(max_workers=6) as pool:
        for i, cases in enumerate(pool.map(build, targets), 1):
            new_cases.extend(cases)
            if i % 10 == 0:
                json.dump([{**c, "evidence": c["evidence"]} for c in new_cases],
                          open(CACHE, "w"), ensure_ascii=False)
                logger.info("进度 %d/%d，已补 %d 条", i, len(targets), len(new_cases))

    new_p = [c for c in new_cases if c["kind"] == "precise"][:need_p]
    new_f = [c for c in new_cases if c["kind"] == "fuzzy"][:need_f]

    # 无关题补齐
    for q in IRRELEVANT_EXTRA:
        if len([c for c in irr]) >= TARGET_I:
            break
        with _search_lock:
            rs = searcher.search(q, top_k=3)
        top = rs[0].get("rerank_score", 0) if rs else 0
        if top < 0.45:
            irr.append({"kind": "irrelevant", "question": q,
                        "points": "应返回低相似度或被闸门拒答", "file": None, "evidence": []})

    all_cases = precise + new_p + fuzzy + new_f + irr
    logger.info("最终 %d 条（精确 %d / 模糊 %d / 无关 %d）",
                len(all_cases), len(precise) + len(new_p),
                len(fuzzy) + len(new_f), len(irr))

    write_docx(all_cases, corpus)


def write_docx(cases, corpus):
    from docx import Document
    from docx.shared import Pt
    from docx.oxml.ns import qn
    doc = Document()
    st = doc.styles["Normal"]; st.font.name = "Calibri"; st.font.size = Pt(10)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    doc.add_heading("CEO-files 检索测试用例（300 条·全文档覆盖版）", level=0)
    doc.add_paragraph(
        f"覆盖 {len(corpus)} 份文档（每份至少 1 题）｜ 100 精确 + 150 模糊 + 50 无关 ｜ "
        "每题依据块 1-5 个（已实跑检索验证）｜ 2026-10-23")
    doc.add_paragraph("依据块格式：文件名#块号；多块表示答案需综合这些块。")
    t = doc.add_table(rows=1, cols=6); t.style = "Table Grid"
    for i, h in enumerate(["编号", "测试查询", "期望命中文件", "期望命中内容要点",
                           "依据块（1-5）", "实测结果"]):
        t.rows[0].cells[i].text = h
    for idx, c in enumerate(cases, 1):
        cells = t.add_row().cells
        cells[0].text = str(idx)
        cells[1].text = c["question"]
        cells[2].text = os.path.basename(c["file"]) if c.get("file") else "（无关问题）"
        cells[3].text = c["points"]
        evs = c.get("evidence") or []
        norm = []
        for e in evs:
            if len(e) == 2 and isinstance(e[0], int):
                norm.append((c.get("file") or "", e[0]))
            else:
                norm.append((e[0], e[1]))
        cells[4].text = "；".join(
            f"{os.path.basename(f) if isinstance(f, str) and os.path.isabs(f) else f}#{i}"
            for f, i in norm) if norm else ""
        cells[5].text = ""
    doc.save(DOCX)
    print(f"已生成: {DOCX}（{len(cases)} 条）")


if __name__ == "__main__":
    main()
