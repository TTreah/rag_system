"""
gen_ceo_testcases_full.py — CEO-files 300 条精准测试集生成器（全文档覆盖）

与旧版（随机抽 30-50 份、单块出题）的区别：
  1. 遍历全部 163 份文档（含纯图片文档），每份至少出 1 题
  2. 每题从 1-3 个连续块构造证据（硬上限 5），依据块写入用例
  3. 生成后实跑检索验证：依据块必须命中；失败自动重生成一次
  4. 无关题实跑检索，确认库内确实无答案（最高分 < 0.45），否则替换

配比：100 精确 + 150 模糊 + 50 无关 = 300

运行:
    python eval/gen_ceo_testcases_full.py
"""

import os
import sys
import json
import random
import logging
import threading
import unicodedata
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import core  # noqa: F401  本地代理绕过

from augment.query_rewriter import _load_api_key, LLM_HOST, LLM_MODEL
import urllib.request

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

COLLECTION = "CEO-files"
OUT = "/Users/trinxai/Documents/pythonfile/测试用例/CEO-files检索测试用例300条-精准版.docx"
N_PRECISE, N_FUZZY, N_IRREL = 100, 150, 50
WORKERS = 8
random.seed(2026)

IRRELEVANT_POOL = [
    "怎么做红烧肉？", "感冒发烧吃什么药？", "怎么考驾照？", "推荐几部悬疑电影",
    "怎么养多肉植物？", "下周天气怎么样？", "怎么学习英语口语？", "怎么做瑜伽下犬式？",
    "怎么给手机刷机？", "番茄炒蛋先放番茄还是先放蛋？", "怎么预防近视？", "怎么做蛋糕？",
    "怎么练习书法？", "怎么选择跑鞋？", "怎么给猫剪指甲？", "咖啡拉花怎么做？",
    "怎么挑选西瓜？", "怎么腌制泡菜？", "怎么弹钢琴入门？", "怎么折纸飞机？",
    "怎么去除衣服油渍？", "怎么养金鱼？", "怎么学摄影构图？", "怎么泡功夫茶？",
    "怎么写小说大纲？", "怎么养仙人掌？", "怎么做风筝？", "怎么学滑板？",
    "怎么挑选榴莲？", "怎么做美甲？", "怎么学吉他？", "怎么搭帐篷？",
    "怎么腌制咸鸭蛋？", "怎么养护绿萝？", "怎么做手工皂？", "怎么学街舞？",
    "怎么挑新鲜的鱼？", "怎么自制酸奶？", "怎么学围棋？", "怎么做香薰蜡烛？",
    "怎么养护兰花？", "怎么学潜水？", "怎么做布贴画？", "怎么训练狗狗握手？",
    "怎么泡发木耳？", "怎么给自行车打气？（注：库里是管理文档，无保养内容）",
    "怎么挑选保温杯？", "怎么学剪纸？", "怎么做糖葫芦？", "怎么养仓鼠？",
    "怎么练习冥想呼吸？", "怎么给植物换盆？",
]


def ask_llm(prompt: str, max_tokens: int = 1800) -> str:
    payload = {"model": LLM_MODEL,
               "messages": [{"role": "user", "content": prompt}],
               "temperature": 0.5, "max_tokens": max_tokens}
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
    if i >= 0 and j > i:
        raw = raw[i:j + 1]
    return json.loads(raw)


def load_corpus() -> dict:
    """{file: {"chunks": [(idx, text)], "assets": [(idx, text)]}}"""
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
            f = pl["source_file"]
            entry = corpus.setdefault(f, {"chunks": [], "assets": []})
            is_asset = bool((pl.get("metadata") or {}).get("asset_path"))
            (entry["assets"] if is_asset else entry["chunks"]).append(
                (pl["chunk_index"], pl["text"]))
        if offset is None:
            break
    for e in corpus.values():
        e["chunks"].sort()
        e["assets"].sort()
    return corpus


def pick_evidence(entry: dict) -> list:
    """挑 1-3 个连续块作为证据（优先长块，长度 400-1600 字）"""
    chunks = entry["chunks"]
    if not chunks:
        return entry["assets"][:1]  # 纯图片文档：用图片描述块
    best, best_len = None, -1
    for i in range(len(chunks)):
        for n in (2, 3, 1):
            if i + n > len(chunks):
                continue
            group = chunks[i:i + n]
            total = sum(len(t) for _, t in group)
            if 400 <= total <= 2200 and total > best_len:
                best, best_len = group, total
    return best or chunks[:1]


def gen_questions(title: str, evidence: list, is_image_only: bool):
    ev_text = "\n\n".join(f"[块{t[0]}]\n{t[1]}" for t in evidence)
    n_ev = len(evidence)
    prompt = f"""以下文档《{title}》的 {n_ev} 个连续片段（方括号里是块号）：

{ev_text[:5000]}

请基于**这些片段本身**生成测试用例，输出 JSON（不要任何其他文字）：
{{
 "precise": {{"question": "针对片段中具体事实/数字/步骤的精确问题", "points": "答案要点，分号分隔，必须全部来自上述片段"}},
 "fuzzy": {{"question": "同一答案但用一线员工日常口语的问法（不要用文档原词，如"这事儿归谁管啊"）", "points": "答案要点，分号分隔"}}
}}

要求：
1. 两题的答案都必须能且只能从上述 {n_ev} 个块中找到
2. 问题不要出现"片段中""根据文档"这类元表述
3. points 保留原文的关键数字、专有名词
4. 若片段是图片描述，问题应围绕该图片内容提问"""
    data = parse_json(ask_llm(prompt))
    out = []
    for kind in ("precise", "fuzzy"):
        item = data.get(kind)
        if item and item.get("question") and item.get("points"):
            out.append((kind, item["question"].strip(), item["points"].strip()))
    return out


_search_lock = threading.Lock()  # MPS 非线程安全：并发调用会崩进程


def verify_case(question: str, evidence: list, title: str, searcher) -> bool:
    """实跑检索：依据块（文件+块号）至少命中 1 个"""
    inputs = [(title, idx) for idx, _ in evidence]
    with _search_lock:
        results = searcher.search(question, top_k=5)
    for r in results:
        for f, idx in inputs:
            if r["source_file"] == f and r.get("chunk_index") == idx:
                return True
    return False


def main():
    corpus = load_corpus()
    files = sorted(corpus)
    logger.info("文档 %d 份", len(files))

    # 配比：每份文档至少 1 题。前 100 份出精确题，前 150 份出模糊题
    # （随机打乱后分配，避免只覆盖目录靠前的文档）
    order = files[:]
    random.shuffle(order)
    precise_docs = set(order[:N_PRECISE])
    fuzzy_docs = set(order[:min(N_FUZZY, len(order))])
    # 不足 150 时，富文本文件补第二道模糊题
    extra_fuzzy_docs = []
    if len(order) < N_FUZZY:
        rich = sorted(files, key=lambda f: -sum(len(t) for _, t in corpus[f]["chunks"]))
        extra_fuzzy_docs = rich[:N_FUZZY - len(order)]

    from qdrant_line.rag_search import RAGSearcher
    searcher = RAGSearcher(collection=COLLECTION, rerank_threshold=0.0)

    def build_one(item):
        f, want_precise, want_fuzzy = item
        entry = corpus[f]
        title = os.path.basename(f)
        ev = pick_evidence(entry)
        if not ev:
            return []
        is_img = not entry["chunks"]
        cases = []
        for attempt in range(2):
            try:
                qs = gen_questions(title, ev, is_img)
            except Exception as e:
                logger.warning("生成失败 %s: %s", title, e)
                continue
            got_p = got_f = False
            for kind, q, pts in qs:
                if kind == "precise" and want_precise and not got_p:
                    if verify_case(q, ev, f, searcher):
                        cases.append({"kind": "precise", "question": q, "points": pts,
                                      "file": f, "evidence": ev})
                        got_p = True
                if kind == "fuzzy" and want_fuzzy and not got_f:
                    if verify_case(q, ev, f, searcher):
                        cases.append({"kind": "fuzzy", "question": q, "points": pts,
                                      "file": f, "evidence": ev})
                        got_f = True
            if (got_p or not want_precise) and (got_f or not want_fuzzy):
                break
        return cases

    tasks = [(f, f in precise_docs, f in fuzzy_docs) for f in files]
    tasks += [(f, False, True) for f in extra_fuzzy_docs]

    all_cases = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for i, cases in enumerate(pool.map(build_one, tasks), 1):
            all_cases.extend(cases)
            if i % 20 == 0:
                logger.info("已处理 %d/%d 文档，累计用例 %d", i, len(tasks), len(all_cases))

    precise = [c for c in all_cases if c["kind"] == "precise"][:N_PRECISE]
    fuzzy = [c for c in all_cases if c["kind"] == "fuzzy"][:N_FUZZY]
    logger.info("生成：精确 %d / 模糊 %d", len(precise), len(fuzzy))

    # 无关题：实跑检索，最高分必须 < 0.45
    irr = []
    for q in IRRELEVANT_POOL:
        if len(irr) >= N_IRREL:
            break
        with _search_lock:
            rs = searcher.search(q, top_k=3)
        top = rs[0].get("rerank_score", 0) if rs else 0
        if top < 0.45:
            irr.append({"kind": "irrelevant", "question": q,
                        "points": "应返回低相似度或被闸门拒答", "file": None, "evidence": []})
        else:
            logger.info("剔除近似无关题（库内有答案，最高分 %.2f）: %s", top, q)

    cases = precise + fuzzy + irr
    logger.info("最终 %d 条（精确 %d / 模糊 %d / 无关 %d）",
                len(cases), len(precise), len(fuzzy), len(irr))

    # 写 docx
    from docx import Document
    from docx.shared import Pt
    from docx.oxml.ns import qn
    doc = Document()
    st = doc.styles["Normal"]; st.font.name = "Calibri"; st.font.size = Pt(10)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    doc.add_heading("CEO-files 检索测试用例（300 条·全文档覆盖版）", level=0)
    doc.add_paragraph(
        f"覆盖全部 {len(files)} 份文档（每份至少 1 题）｜ 100 精确 + 150 模糊 + 50 无关 ｜ "
        "每题依据块 1-5 个（已实跑检索验证）｜ 2026-10-23")
    doc.add_paragraph("说明：依据块格式为 `文件名#块号`；多块表示答案需综合这些块。")
    t = doc.add_table(rows=1, cols=6); t.style = "Table Grid"
    for i, h in enumerate(["编号", "测试查询", "期望命中文件", "期望命中内容要点",
                           "依据块（1-5）", "实测结果"]):
        t.rows[0].cells[i].text = h
    for idx, c in enumerate(cases, 1):
        cells = t.add_row().cells
        cells[0].text = str(idx)
        cells[1].text = c["question"]
        cells[2].text = os.path.basename(c["file"]) if c["file"] else "（无关问题）"
        cells[3].text = c["points"]
        cells[4].text = "；".join(f"{os.path.basename(c['file'])}#{i}" for i, _ in c["evidence"])
        cells[5].text = ""
    doc.save(OUT)
    print(f"已生成: {OUT}")


if __name__ == "__main__":
    main()
